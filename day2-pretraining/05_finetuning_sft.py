"""
Hari 2 - Fine-tuning SFT dengan HuggingFace + TRL
====================================================
Fine-tune model kecil (GPT-2) pada dataset instruksi sederhana
menggunakan Supervised Fine-Tuning (SFT).

REKOMENDASI: Jalankan di Google Colab (GPU T4 gratis)
  https://colab.research.google.com/

Setup di Colab (cell pertama):
  !pip install transformers trl datasets accelerate -q

Estimasi waktu:
  - GPT-2 (124M param) + 100 contoh: ~2-3 menit di T4
  - Mistral 7B + QLoRA: ~20-30 menit di T4 (lihat 06_qlora_finetuning.py)

Jalankan: python 05_finetuning_sft.py
"""

# ---------------------------------------------------------------------------
# INSTALL (uncomment jika perlu)
# ---------------------------------------------------------------------------
# import subprocess
# subprocess.run(["pip", "install", "transformers", "trl", "datasets",
#                 "accelerate", "-q"])


# ---------------------------------------------------------------------------
# IMPORT
# ---------------------------------------------------------------------------

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# 1. DATASET — pasangan (prompt, response)
# ---------------------------------------------------------------------------

# Dataset instruksi sederhana bertema data science / LLM
# Di produksi: bisa load dari HuggingFace datasets atau file JSON/CSV
RAW_DATA = [
    {
        "prompt": "Apa itu overfitting dalam machine learning?",
        "response": (
            "Overfitting terjadi ketika model terlalu 'hafal' data training "
            "sehingga performanya buruk pada data baru. Model belajar noise "
            "dan detail spesifik data training, bukan pola umum yang bisa "
            "digeneralisasi. Solusi: tambah data, gunakan regularisasi "
            "(L1/L2), dropout, atau early stopping."
        ),
    },
    {
        "prompt": "Jelaskan perbedaan supervised dan unsupervised learning.",
        "response": (
            "Supervised learning menggunakan data berlabel — setiap input "
            "punya jawaban yang benar untuk dijadikan target training. "
            "Contoh: klasifikasi email spam, prediksi harga rumah. "
            "Unsupervised learning tidak punya label — model menemukan "
            "pola sendiri dari data mentah. Contoh: clustering pelanggan, "
            "dimensionality reduction dengan PCA."
        ),
    },
    {
        "prompt": "Apa itu gradient descent?",
        "response": (
            "Gradient descent adalah algoritma optimasi yang dipakai untuk "
            "melatih model ML. Cara kerjanya: hitung gradien (turunan) dari "
            "loss function terhadap setiap parameter, lalu gerakkan parameter "
            "ke arah yang menurunkan loss. Diulang berkali-kali sampai loss "
            "konvergen ke nilai minimum. Learning rate mengontrol seberapa "
            "besar langkah setiap iterasi."
        ),
    },
    {
        "prompt": "Apa fungsi softmax dalam neural network?",
        "response": (
            "Softmax mengubah vektor angka mentah (logits) menjadi distribusi "
            "probabilitas yang jumlahnya tepat 1.0. Dipakai di output layer "
            "untuk klasifikasi multi-kelas — setiap nilai output jadi "
            "probabilitas untuk kelas tersebut. Rumus: "
            "softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ). "
            "Nilai terbesar mendapat probabilitas tertinggi."
        ),
    },
    {
        "prompt": "Jelaskan apa itu attention mechanism dalam transformer.",
        "response": (
            "Attention mechanism memungkinkan setiap token dalam sekuens "
            "untuk 'memperhatikan' token lain secara berbobot. Setiap token "
            "diproyeksikan jadi tiga vektor: Query (apa yang dicari), "
            "Key (apa yang ditawarkan), Value (isi informasi). "
            "Skor attention = softmax(Q·Kᵀ / √d) · V. "
            "Token yang lebih relevan mendapat bobot lebih besar, "
            "sehingga representasi output setiap token sudah mengandung "
            "konteks dari token lain."
        ),
    },
    {
        "prompt": "Apa perbedaan batch normalization dan layer normalization?",
        "response": (
            "Batch normalization menormalisasi aktivasi per-fitur di seluruh "
            "mini-batch — bergantung pada ukuran batch dan kurang cocok "
            "untuk sekuens variable-length. "
            "Layer normalization menormalisasi per-sampel di seluruh fitur "
            "— tidak bergantung pada batch size, lebih stabil untuk RNN "
            "dan Transformer. LLM modern hampir selalu pakai LayerNorm."
        ),
    },
    {
        "prompt": "Apa itu embedding dalam konteks NLP?",
        "response": (
            "Embedding adalah representasi kata atau token sebagai vektor "
            "angka berdimensi tinggi. Kata dengan makna mirip memiliki "
            "vektor yang berdekatan dalam ruang embedding. Contoh properti "
            "embedding yang baik: vektor(raja) - vektor(pria) + "
            "vektor(wanita) ≈ vektor(ratu). Embedding dipelajari otomatis "
            "saat training, bukan dirancang manual."
        ),
    },
    {
        "prompt": "Bagaimana cara mencegah vanishing gradient?",
        "response": (
            "Beberapa teknik untuk mencegah vanishing gradient: "
            "1) Residual connections (skip connections) seperti di ResNet "
            "dan Transformer — gradien bisa mengalir langsung tanpa melewati "
            "banyak layer. "
            "2) Normalisasi (BatchNorm/LayerNorm) — menjaga skala aktivasi. "
            "3) Aktivasi ReLU dan variannya — tidak saturasi untuk nilai "
            "positif. "
            "4) Gradient clipping — batasi nilai maksimum gradien. "
            "5) Inisialisasi bobot yang tepat (Xavier, He initialization)."
        ),
    },
    {
        "prompt": "Apa itu tokenisasi dalam NLP?",
        "response": (
            "Tokenisasi adalah proses memecah teks menjadi unit-unit kecil "
            "yang disebut token. Token bisa berupa kata, subkata, atau "
            "karakter. LLM modern umumnya pakai subword tokenization seperti "
            "BPE (Byte Pair Encoding) — kata umum disimpan utuh, kata langka "
            "dipecah jadi subword. Contoh: 'unhappiness' → ['un', 'happiness']. "
            "Setiap token diubah jadi ID angka yang menjadi input model."
        ),
    },
    {
        "prompt": "Jelaskan perbedaan precision dan recall.",
        "response": (
            "Precision = TP / (TP + FP): dari semua yang diprediksi positif, "
            "berapa persen yang benar-benar positif. Penting ketika false "
            "positive mahal (spam filter — jangan sampai email penting masuk "
            "spam). "
            "Recall = TP / (TP + FN): dari semua yang benar-benar positif, "
            "berapa persen yang berhasil terdeteksi. Penting ketika false "
            "negative berbahaya (deteksi kanker — jangan sampai ada yang "
            "terlewat). F1 score adalah harmonic mean keduanya."
        ),
    },
]


def build_dataset(data: list, tokenizer) -> Dataset:
    """
    Format dataset ke chat template dan tokenize.
    Template: <|user|>\\n{prompt}\\n<|assistant|>\\n{response}
    """
    formatted = []
    for item in data:
        text = (
            f"<|user|>\n{item['prompt']}\n"
            f"<|assistant|>\n{item['response']}"
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    return dataset


# ---------------------------------------------------------------------------
# 2. LOAD MODEL & TOKENIZER
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = "gpt2"):
    """
    Load model dan tokenizer.
    GPT-2 (124M) dipilih karena:
    - Cukup kecil untuk dilatih di CPU/T4 gratis
    - Sudah ada di HuggingFace, tidak perlu download besar
    - Representasi yang baik untuk memahami konsep

    Untuk model lebih besar, lihat 06_qlora_finetuning.py
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 tidak punya pad token — tambahkan
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameter: {total_params:,} ({total_params/1e6:.0f}M)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. TRAINING CONFIG
# ---------------------------------------------------------------------------

def get_training_args(output_dir: str = "./sft-gpt2-output") -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,

        # Jumlah epoch — 3 cukup untuk dataset kecil
        num_train_epochs=3,

        # Batch size kecil karena GPU T4 memory terbatas
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch = 2×4 = 8

        # Learning rate
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # Logging
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,

        # Panjang maksimum sekuens
        max_seq_length=256,

        # Format output
        report_to="none",   # ganti "wandb" jika ingin tracking
    )


# ---------------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------------

def train(model_name: str = "gpt2"):
    print("=" * 55)
    print("SFT Fine-tuning GPT-2")
    print("=" * 55)

    # Load
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Dataset
    print(f"\nMempersiapkan dataset ({len(RAW_DATA)} contoh)...")
    dataset = build_dataset(RAW_DATA, tokenizer)
    print(f"Contoh formatted:\n{dataset[0]['text'][:200]}...\n")

    # Training args
    args = get_training_args()

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    # Train
    print("Mulai training...")
    train_result = trainer.train()

    print(f"\nTraining selesai!")
    print(f"  Loss akhir    : {train_result.training_loss:.4f}")
    print(f"  Steps         : {train_result.global_step}")
    print(f"  Runtime       : {train_result.metrics['train_runtime']:.1f} detik")

    # Simpan model
    print(f"\nMenyimpan model ke {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("Model tersimpan.")

    return trainer, tokenizer


# ---------------------------------------------------------------------------
# 5. INFERENCE — TEST MODEL SETELAH FINE-TUNING
# ---------------------------------------------------------------------------

def test_inference(model_dir: str = "./sft-gpt2-output"):
    """
    Uji model hasil fine-tuning dengan beberapa prompt.
    Bandingkan output sebelum dan sesudah fine-tuning untuk
    melihat apakah model mulai mengikuti format instruksi.
    """
    import torch

    print("\n" + "=" * 55)
    print("Test inference model fine-tuned")
    print("=" * 55)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    test_prompts = [
        "Apa itu regularisasi dalam machine learning?",
        "Jelaskan apa itu cross-validation.",
    ]

    for prompt in test_prompts:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(formatted, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode hanya token baru (bukan prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\nPrompt  : {prompt}")
        print(f"Response: {response}")
        print("-" * 55)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Training
    trainer, tokenizer = train(model_name="gpt2")

    # Test
    test_inference(model_dir="./sft-gpt2-output")

    print("""
Tips lanjutan:
  1. Tambah lebih banyak data → kualitas output naik signifikan
  2. Coba model lebih besar (gpt2-medium, gpt2-large) jika punya GPU
  3. Untuk model 7B+ pakai QLoRA (lihat 06_qlora_finetuning.py)
  4. Monitor loss dengan: tensorboard --logdir ./sft-gpt2-output
    """)
