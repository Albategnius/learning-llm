# Learning LLM

Catatan dan kode latihan praktis dari proses belajar LLM (Large Language Model) —
dari fondasi konsep Transformer hingga aplikasi dan optimasi.

Target: 7 hari, dari konsep hingga LLMOps.

## Struktur folder

```
learning-llm/
└── day1-foundations/      # Tokenisasi, embedding, self-attention
    ├── 01_tokenization.py
    ├── 02_model_architecture.py
    ├── 03_attention_weights.py
    ├── 04_attention_from_scratch.py
    └── requirements.txt
```

Folder untuk hari-hari berikutnya (`day2-pretraining`, `day3-prompting`,
`day4-rag`, `day5-finetuning`, `day6-optimization`, `day7-llmops`) akan
ditambahkan seiring progres belajar.

## Hari 1 — Fondasi LLM & Arsitektur Transformer

Topik: kenapa Transformer menggantikan RNN, mekanisme self-attention
(Q, K, V), multi-head attention, embedding, dan positional encoding.

### Cara menjalankan

```bash
cd day1-foundations
pip install -r requirements.txt

python 01_tokenization.py          # Bandingkan tokenizer BPE vs WordPiece
python 02_model_architecture.py    # Lihat susunan layer GPT-2
python 03_attention_weights.py     # Forward pass + lihat attention weights mentah
python 04_attention_from_scratch.py  # Demo manual matrix attention (numpy saja)
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

## Lisensi

Untuk keperluan belajar pribadi.
