"""
Hari 2 - Simulasi Data Pipeline Pre-training
==============================================
Simulasi sederhana bagaimana data mentah diproses sebelum dipakai
untuk pre-training: language detection, quality filter, deduplication,
dan tokenization.

Di dunia nyata pipeline ini berjalan pada petabytes data dengan
Apache Spark / Dask. Di sini kita simulasikan logikanya pada skala kecil.

Jalankan: python 02_data_pipeline.py
Requirement: pip install transformers
"""

import re
import hashlib
from transformers import AutoTokenizer


# --- Contoh data mentah simulasi (seperti hasil crawl) ---
RAW_DOCUMENTS = [
    "The transformer architecture has revolutionized natural language processing.",
    "Buy cheap products now!!! Click here!!! Best deals!!!",
    "Kucing itu duduk di atas tikar sambil berjemur di bawah sinar matahari.",
    "The transformer architecture has revolutionized natural language processing.",  # duplikat
    "asjdhaskjdhaksjdh random gibberish asdkjasd",
    "Python is a high-level programming language known for its readability.",
    "!!!! #### @@@@",  # noise
    "Machine learning models learn patterns from data through optimization.",
    "Kucing itu duduk di atas tikar sambil berjemur di bawah sinar matahari.",  # duplikat
    "Deep learning has enabled significant advances in computer vision and NLP.",
]


def step1_language_filter(docs: list[str], target_lang: str = "en") -> list[str]:
    """
    Deteksi bahasa sederhana berdasarkan karakter ASCII.
    Di produksi: pakai fastText language detection.
    """
    def is_english(text):
        ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
        return ascii_ratio > 0.85

    filtered = [d for d in docs if is_english(d)]
    print(f"[Language filter] {len(docs)} → {len(filtered)} dokumen")
    return filtered


def step2_quality_filter(docs: list[str]) -> list[str]:
    """
    Filter dokumen berkualitas rendah berdasarkan beberapa heuristik:
    - terlalu pendek
    - terlalu banyak tanda seru / karakter aneh
    - rasio huruf terlalu rendah
    """
    def is_quality(text):
        if len(text.split()) < 5:
            return False
        exclamation_ratio = text.count("!") / max(len(text), 1)
        if exclamation_ratio > 0.05:
            return False
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.6:
            return False
        return True

    filtered = [d for d in docs if is_quality(d)]
    print(f"[Quality filter]  {len(docs)} → {len(filtered)} dokumen")
    return filtered


def step3_deduplication(docs: list[str]) -> list[str]:
    """
    Buang dokumen duplikat menggunakan hashing.
    Di produksi: MinHash / SimHash untuk near-duplicate detection.
    """
    seen = set()
    unique = []
    for doc in docs:
        h = hashlib.md5(doc.strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    print(f"[Deduplication]   {len(docs)} → {len(unique)} dokumen")
    return unique


def step4_tokenize(docs: list[str], tokenizer) -> list[list[int]]:
    """
    Ubah teks jadi token IDs.
    Di produksi: dikemas jadi fixed-length chunks (context_length token).
    """
    tokenized = [tokenizer.encode(doc) for doc in docs]
    total_tokens = sum(len(t) for t in tokenized)
    print(f"[Tokenization]    {len(docs)} dokumen → {total_tokens} total token")
    return tokenized


def main():
    print("="*60)
    print("Simulasi data pipeline pre-training")
    print("="*60)
    print(f"\nData mentah: {len(RAW_DOCUMENTS)} dokumen\n")

    tok = AutoTokenizer.from_pretrained("gpt2")

    # Jalankan pipeline step by step
    docs = RAW_DOCUMENTS[:]
    docs = step1_language_filter(docs)
    docs = step2_quality_filter(docs)
    docs = step3_deduplication(docs)
    tokenized = step4_tokenize(docs, tok)

    print("\n" + "="*60)
    print("Dokumen final yang lolos pipeline:")
    print("="*60)
    for i, (doc, tokens) in enumerate(zip(docs, tokenized), 1):
        print(f"\n{i}. [{len(tokens)} token] {doc}")

    print("\n" + "="*60)
    print("Catatan skala nyata:")
    print("="*60)
    print("""
    CommonCrawl dump mentah : ~400 TB per bulan
    Setelah language filter : ~40 TB (hanya teks Inggris)
    Setelah quality filter  : ~10-20 TB
    Setelah dedup           : ~5-10 TB
    → setara ~1-3 triliun token untuk satu dataset training

    LLaMA-3 dilatih pada 15 triliun token dari berbagai sumber:
    CommonCrawl, GitHub, Wikipedia, buku, ArXiv, Stack Exchange, dll.
    """)


if __name__ == "__main__":
    main()
