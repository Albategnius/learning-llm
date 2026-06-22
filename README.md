# Learning LLM

Catatan dan kode latihan praktis dari proses belajar LLM (Large Language Model) —
dari fondasi konsep Transformer hingga aplikasi dan optimasi.

Target: 7 hari, dari konsep hingga LLMOps.

## Struktur folder

```
learning-llm/
├── day1-foundations/      # Tokenisasi, embedding, self-attention
│   ├── 01_tokenization.py
│   ├── 02_model_architecture.py
│   ├── 03_attention_weights.py
│   ├── 04_attention_from_scratch.py
│   └── requirements.txt
└── day2-pretraining/      # Pre-training, fine-tuning, scaling laws
    ├── 01_perplexity_demo.py
    ├── 02_data_pipeline.py
    ├── 03_finetuning_concepts.py
    ├── 04_scaling_laws.py
    └── requirements.txt
```

Folder untuk hari-hari berikutnya (`day3-prompting`, `day4-rag`,
`day5-finetuning`, `day6-optimization`, `day7-llmops`) akan ditambahkan
seiring progres belajar.

## Hari 1 — Fondasi LLM & Arsitektur Transformer

Topik: kenapa Transformer menggantikan RNN, mekanisme self-attention
(Q, K, V), multi-head attention, embedding, dan positional encoding.

### Cara menjalankan

```bash
cd day1-foundations
pip install -r requirements.txt

python 01_tokenization.py           # Bandingkan tokenizer BPE vs WordPiece
python 02_model_architecture.py     # Lihat susunan layer GPT-2
python 03_attention_weights.py      # Forward pass + lihat attention weights mentah
python 04_attention_from_scratch.py # Demo manual matrix attention (numpy saja)
```

### Ringkasan konsep

- **Embedding**: representasi kata sebagai vektor di ruang banyak-dimensi,
  di mana jarak antar vektor mencerminkan kemiripan makna
  (`raja - pria + wanita ≈ ratu`).
- **Self-attention**: tiap token diproyeksikan jadi Query (apa yang dicari),
  Key (apa yang ditawarkan), dan Value (isi informasi), lalu dihitung
  `softmax(Q·Kᵀ / √d) · V` untuk menghasilkan representasi kontekstual.
- **Kenapa bukan RNN**: RNN memproses token secara sekuensial (lambat,
  rawan vanishing gradient untuk sekuens panjang). Transformer memproses
  semua token paralel lewat attention.

## Hari 2 — Pre-training, Fine-tuning & Data

Topik: siklus hidup model dari data mentah, causal language modeling,
data pipeline, scaling laws (Chinchilla), SFT, RLHF, dan DPO.

### Cara menjalankan

```bash
cd day2-pretraining
pip install -r requirements.txt

python 01_perplexity_demo.py    # Perplexity: ukuran "kebingungan" model
python 02_data_pipeline.py      # Simulasi filter + dedup data pre-training
python 03_finetuning_concepts.py # Struktur dataset SFT dan DPO
python 04_scaling_laws.py       # Kalkulator Chinchilla scaling laws
```

### Ringkasan konsep

- **Pre-training**: model dilatih memprediksi token berikutnya pada
  triliunan token. Tugas sederhana tapi skala besar memaksa model belajar
  semua aspek bahasa secara implisit.
- **Perplexity**: `e^loss` — ukuran seberapa "bingung" model terhadap
  suatu teks. Makin rendah = teks makin masuk akal menurut model.
- **Scaling laws (Chinchilla)**: token optimal ≈ 20 × jumlah parameter.
  Model lebih kecil + data lebih banyak seringkali lebih efisien.
- **SFT**: supervised fine-tuning pada pasangan (prompt, respons ideal).
- **DPO**: langsung optimize dari preferensi (chosen vs rejected) tanpa
  reward model terpisah. Alternatif RLHF yang lebih stabil.

## Lisensi

Untuk keperluan belajar pribadi.
