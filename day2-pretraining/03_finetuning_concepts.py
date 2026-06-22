"""
Hari 2 - Fine-tuning: SFT dan DPO
====================================
Demonstrasi konsep Supervised Fine-Tuning (SFT) dan Direct Preference
Optimization (DPO). Kode ini menunjukkan struktur dataset dan training loop
tanpa benar-benar melatih model besar (butuh GPU).

Untuk menjalankan fine-tuning sungguhan:
  pip install transformers trl datasets peft accelerate

Jalankan demo ini (tanpa training): python 03_finetuning_concepts.py
"""

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# 1. FORMAT DATASET SFT
# ---------------------------------------------------------------------------

SFT_DATASET = [
    {
        "prompt": "Ringkas teks berikut dalam satu kalimat: "
                  "Transformer adalah arsitektur neural network yang "
                  "memproses seluruh token input secara paralel menggunakan "
                  "mekanisme attention.",
        "response": "Transformer adalah arsitektur yang memproses token secara "
                    "paralel lewat attention, menggantikan RNN yang sekuensial.",
    },
    {
        "prompt": "Apa itu embedding dalam konteks LLM?",
        "response": "Embedding adalah representasi kata sebagai vektor angka "
                    "berdimensi tinggi, di mana kata bermakna mirip memiliki "
                    "vektor yang berdekatan dalam ruang tersebut.",
    },
    {
        "prompt": "Jelaskan perbedaan encoder-only dan decoder-only transformer.",
        "response": "Encoder-only (seperti BERT) melihat seluruh konteks "
                    "bidireksional, cocok untuk klasifikasi. Decoder-only "
                    "(seperti GPT) hanya melihat token sebelumnya, cocok "
                    "untuk generasi teks.",
    },
]

# Format chat template — inilah yang sebenarnya masuk ke model saat SFT
CHAT_TEMPLATE = "<|user|>\n{prompt}\n<|assistant|>\n{response}"


# ---------------------------------------------------------------------------
# 2. FORMAT DATASET DPO
# ---------------------------------------------------------------------------

DPO_DATASET = [
    {
        "prompt": "Bagaimana cara belajar machine learning?",
        "chosen": (                          # respons yang LEBIH baik
            "Mulai dari matematika dasar (linear algebra, statistik), "
            "lalu pelajari Python dan library seperti scikit-learn. "
            "Praktik langsung dengan dataset nyata di Kaggle."
        ),
        "rejected": (                        # respons yang KURANG baik
            "Belajar machine learning itu susah. "
            "Kamu perlu banyak belajar."
        ),
    },
    {
        "prompt": "Apa itu overfitting?",
        "chosen": (
            "Overfitting terjadi ketika model terlalu hafal data training "
            "sehingga performanya buruk pada data baru. Solusinya: "
            "regularisasi, dropout, early stopping, atau tambah data."
        ),
        "rejected": (
            "Overfitting adalah masalah dalam machine learning "
            "yang harus dihindari."
        ),
    },
]


# ---------------------------------------------------------------------------
# 3. PSEUDO-CODE TRAINING LOOP
# ---------------------------------------------------------------------------

SFT_TRAINING_PSEUDOCODE = """
# SFT Training Loop (dengan HuggingFace TRL)
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./sft-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",   # kolom berisi formatted prompt+response
    max_seq_length=512,
    tokenizer=tokenizer,
)

trainer.train()  # update weight model supaya cocok dengan pasangan (prompt, response)
"""

DPO_TRAINING_PSEUDOCODE = """
# DPO Training Loop (dengan HuggingFace TRL)
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-7,     # LR lebih kecil dari SFT — fine adjustment
    beta=0.1,               # hyperparameter DPO: seberapa ketat constraint
)

trainer = DPOTrainer(
    model=model,            # model setelah SFT
    ref_model=ref_model,    # snapshot model SFT — jadi referensi constraint
    args=dpo_config,
    train_dataset=dataset,  # kolom: prompt, chosen, rejected
    tokenizer=tokenizer,
)

trainer.train()
# DPO loss: naikkan P(chosen | prompt), turunkan P(rejected | prompt)
# relatif terhadap ref_model — supaya tidak terlalu jauh dari baseline
"""


# ---------------------------------------------------------------------------
# 4. DEMO: FORMAT TOKEN SFT
# ---------------------------------------------------------------------------

def demo_sft_tokenization():
    tok = AutoTokenizer.from_pretrained("gpt2")
    sample = SFT_DATASET[0]
    formatted = CHAT_TEMPLATE.format(**sample)

    print("="*60)
    print("Contoh input SFT setelah formatting:")
    print("="*60)
    print(formatted)

    tokens = tok.encode(formatted)
    print(f"\nJumlah token: {len(tokens)}")
    print(f"Token IDs (10 pertama): {tokens[:10]}")
    print("\nCatatan: saat training SFT, loss HANYA dihitung pada bagian")
    print("respons (setelah <|assistant|>), bukan pada bagian prompt.")
    print("Ini supaya model belajar menghasilkan respons, bukan menghafal prompt.")


def main():
    print("\n" + "="*60)
    print("1. Struktur dataset SFT")
    print("="*60)
    for i, ex in enumerate(SFT_DATASET, 1):
        print(f"\nContoh {i}:")
        print(f"  Prompt  : {ex['prompt'][:60]}...")
        print(f"  Response: {ex['response'][:60]}...")

    print("\n" + "="*60)
    print("2. Struktur dataset DPO")
    print("="*60)
    for i, ex in enumerate(DPO_DATASET, 1):
        print(f"\nContoh {i}:")
        print(f"  Prompt  : {ex['prompt']}")
        print(f"  Chosen  : {ex['chosen'][:60]}...")
        print(f"  Rejected: {ex['rejected'][:60]}...")

    demo_sft_tokenization()

    print("\n" + "="*60)
    print("3. SFT training pseudocode")
    print("="*60)
    print(SFT_TRAINING_PSEUDOCODE)

    print("="*60)
    print("4. DPO training pseudocode")
    print("="*60)
    print(DPO_TRAINING_PSEUDOCODE)

    print("="*60)
    print("5. Kapan pakai apa?")
    print("="*60)
    print("""
    SFT   → task spesifik dengan output jelas (klasifikasi, ekstraksi,
            format tertentu). Dataset lebih mudah dibuat.

    DPO   → alignment: model harus tahu respons mana lebih baik secara
            kualitas/keamanan/relevansi. Butuh dataset preferensi (chosen
            vs rejected). Lebih stabil dari RLHF.

    RLHF  → alignment skala besar (GPT-4, Claude). Kompleks, butuh reward
            model terpisah dan RL training. Biasanya dilakukan lab besar.

    Urutan yang umum dipakai:
    Base model → SFT → DPO → deployed model
    """)


if __name__ == "__main__":
    main()
