"""
Hari 2 - Fine-tuning Model 7B dengan QLoRA
============================================
QLoRA (Quantized Low-Rank Adaptation) memungkinkan fine-tune model
besar (7B+ parameter) di GPU dengan VRAM terbatas seperti T4 (16GB).

Cara kerjanya:
  1. Quantize model ke 4-bit (hemat ~75% memory)
  2. Tambahkan adapter LoRA kecil di atas layer frozen
  3. Hanya train adapter LoRA (~0.1% dari total parameter)
  4. Merge adapter ke model asli setelah training selesai

REKOMENDASI: Jalankan di Google Colab dengan GPU T4
  Runtime → Change runtime type → T4 GPU

Setup di Colab:
  !pip install transformers trl peft bitsandbytes datasets accelerate -q

Model yang dipakai: TinyLlama-1.1B (lebih kecil dari 7B, cocok untuk
demo dan bisa jalan di T4 tanpa OOM)

Untuk model 7B (Mistral, LLaMA): sama persis, ganti model_name saja.
Butuh T4 16GB atau A100.

Jalankan: python 06_qlora_finetuning.py
"""

# ---------------------------------------------------------------------------
# IMPORT
# ---------------------------------------------------------------------------

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Ganti ke salah satu ini untuk model lebih besar (butuh GPU lebih besar):
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME = "google/gemma-2b-it"

OUTPUT_DIR = "./qlora-output"


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

RAW_DATA = [
    {
        "prompt": "Apa itu overfitting dan bagaimana mengatasinya?",
        "response": (
            "Overfitting terjadi ketika model terlalu hafal data training "
            "sehingga gagal generalisasi ke data baru. Solusinya: "
            "tambah data training, gunakan regularisasi L1/L2, dropout, "
            "early stopping, atau cross-validation untuk monitoring."
        ),
    },
    {
        "prompt": "Jelaskan cara kerja attention mechanism.",
        "response": (
            "Attention mechanism menghitung seberapa relevan setiap token "
            "terhadap token lain. Setiap token punya tiga representasi: "
            "Query (apa yang dicari), Key (apa yang ditawarkan), Value "
            "(isi informasi). Skor = softmax(Q·Kᵀ/√d) · V. "
            "Token relevan mendapat bobot lebih besar di output."
        ),
    },
    {
        "prompt": "Apa perbedaan precision dan recall?",
        "response": (
            "Precision = seberapa akurat prediksi positif kita "
            "(TP / (TP+FP)). Recall = seberapa lengkap kita menangkap "
            "semua positif sesungguhnya (TP / (TP+FN)). "
            "Gunakan precision ketika false positive mahal (spam filter), "
            "recall ketika false negative berbahaya (deteksi penyakit)."
        ),
    },
    {
        "prompt": "Bagaimana cara kerja gradient descent?",
        "response": (
            "Gradient descent mengoptimalkan parameter model dengan "
            "menghitung gradien loss terhadap setiap parameter, "
            "lalu menggerakkan parameter ke arah yang menurunkan loss. "
            "Update: θ = θ - α·∇L(θ), dengan α = learning rate. "
            "Diulang ratusan ribu kali sampai loss konvergen."
        ),
    },
    {
        "prompt": "Apa itu embedding dalam NLP?",
        "response": (
            "Embedding adalah representasi kata/token sebagai vektor angka "
            "berdimensi tinggi. Kata bermakna mirip memiliki vektor "
            "berdekatan dalam ruang embedding. Properti menarik: "
            "raja - pria + wanita ≈ ratu. Embedding dipelajari otomatis "
            "saat training, bukan dirancang manual."
        ),
    },
]


def build_dataset(data: list) -> Dataset:
    """Format ke chat template TinyLlama/Mistral."""
    formatted = []
    for item in data:
        # Format chat template standar untuk instruction-tuned models
        text = (
            f"<|system|>\nKamu adalah asisten AI yang membantu "
            f"menjelaskan konsep data science dan machine learning.</s>\n"
            f"<|user|>\n{item['prompt']}</s>\n"
            f"<|assistant|>\n{item['response']}</s>"
        )
        formatted.append({"text": text})
    return Dataset.from_list(formatted)


# ---------------------------------------------------------------------------
# 1. QUANTIZATION CONFIG (4-bit)
# ---------------------------------------------------------------------------

def get_bnb_config() -> BitsAndBytesConfig:
    """
    Konfigurasi 4-bit quantization dengan bitsandbytes.

    nf4 (NormalFloat4): format quantization khusus untuk weight
    yang terdistribusi normal — lebih akurat dari INT4 biasa.

    double_quant: quantize juga konstanta quantization itu sendiri
    — hemat tambahan ~0.4 bit per parameter.

    Hasil: model 7B yang biasa butuh ~14GB VRAM (fp16)
           jadi hanya butuh ~4GB VRAM (4-bit).
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ---------------------------------------------------------------------------
# 2. LORA CONFIG
# ---------------------------------------------------------------------------

def get_lora_config() -> LoraConfig:
    """
    Konfigurasi LoRA adapter.

    r (rank): dimensi matriks adapter. Lebih besar = lebih ekspresif
    tapi lebih banyak parameter. Nilai umum: 8, 16, 32, 64.

    alpha: scaling factor. Biasanya diset = r atau 2×r.
    Efektif learning rate LoRA ∝ alpha/r.

    target_modules: layer mana yang diberi adapter.
    q_proj, v_proj = proyeksi Query dan Value di attention.
    Bisa tambahkan k_proj, o_proj, gate_proj, dll untuk coverage lebih.

    dropout: regularisasi untuk adapter LoRA.

    Jumlah trainable params dengan r=16:
    Tiap layer: 2 × (d_model × r + r × d_model) = 4 × d_model × r
    TinyLlama 1.1B, d_model=2048, 22 layers:
    ≈ 22 × 4 × 2048 × 16 ≈ 2.9M params (0.26% dari total)
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# 3. TRAINING
# ---------------------------------------------------------------------------

def train():
    print("=" * 56)
    print(f"QLoRA Fine-tuning: {MODEL_NAME}")
    print("=" * 56)

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model dalam 4-bit
    print("Loading model dalam 4-bit quantization...")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",        # otomatis distribusi ke GPU/CPU
        torch_dtype=torch.float16,
    )

    # Persiapkan untuk kbit training
    model = prepare_model_for_kbit_training(model)

    # Tambah LoRA adapter
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Lihat berapa parameter yang ditraining
    model.print_trainable_parameters()

    # Dataset
    print("\nMempersiapkan dataset...")
    dataset = build_dataset(RAW_DATA)

    # Training config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_seq_length=512,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        fp16=True,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    print("\nMulai training...")
    result = trainer.train()

    print(f"\nTraining selesai!")
    print(f"  Loss akhir : {result.training_loss:.4f}")
    print(f"  Runtime    : {result.metrics['train_runtime']:.1f} detik")

    # Simpan adapter (bukan seluruh model — jauh lebih kecil)
    print(f"\nMenyimpan LoRA adapter ke {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Selesai. Adapter tersimpan (~50MB vs ~4GB full model).")

    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. INFERENCE
# ---------------------------------------------------------------------------

def test_inference(model, tokenizer):
    """Test model hasil fine-tuning."""
    print("\n" + "=" * 56)
    print("Test inference")
    print("=" * 56)

    test_prompts = [
        "Apa itu cross-validation?",
        "Jelaskan perbedaan bagging dan boosting.",
    ]

    model.eval()
    for prompt in test_prompts:
        formatted = (
            f"<|system|>\nKamu adalah asisten AI yang membantu "
            f"menjelaskan konsep data science.</s>\n"
            f"<|user|>\n{prompt}</s>\n"
            f"<|assistant|>\n"
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\nPrompt  : {prompt}")
        print(f"Response: {response}")
        print("-" * 56)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, tokenizer = train()
    test_inference(model, tokenizer)

    print("""
Langkah selanjutnya:
  1. Tambah lebih banyak data domain kamu (asuransi, keuangan, dll)
  2. Naikkan r=16 ke r=32/64 untuk kapasitas lebih besar
  3. Tambahkan target_modules: k_proj, o_proj, gate_proj
  4. Gunakan DPO setelah SFT untuk alignment (07_dpo.py — coming soon)
  5. Merge adapter ke model penuh:
       from peft import PeftModel
       base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
       merged = PeftModel.from_pretrained(base, OUTPUT_DIR).merge_and_unload()
    """)
