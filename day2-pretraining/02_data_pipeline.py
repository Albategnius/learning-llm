"""
Hari 2 - Data Pipeline Pre-training (Lengkap)
===============================================
Simulasi pipeline data pre-training dari dokumen mentah hingga token
siap masuk training loop. Mencakup:

  1. Language filter    — buang dokumen bukan bahasa target
  2. Quality filter     — buang spam, terlalu pendek, terlalu repetitif
  3. Exact dedup        — hash MD5, buang dokumen identik
  4. Near-dedup         — MinHash + Jaccard similarity, buang dokumen mirip
  5. Tokenization       — ubah teks ke token IDs (butuh transformers)
  6. Packing            — kemas token ke fixed-length chunks

Di produksi pipeline ini berjalan pada petabytes data dengan
Apache Spark / Dask di ratusan mesin. Di sini kita simulasikan
logika dan konsepnya pada skala kecil.

Jalankan: python 02_data_pipeline.py
Requirement: pip install transformers
"""

import hashlib


# ---------------------------------------------------------------------------
# DATA MENTAH SIMULASI
# ---------------------------------------------------------------------------

RAW_DOCUMENTS = [
    # Dokumen normal berkualitas
    "The transformer architecture has revolutionized natural language processing. "
    "By replacing recurrent networks with self-attention mechanisms, transformers "
    "enable parallel processing of entire sequences at once.",

    "Machine learning models learn patterns from data through optimization. "
    "Gradient descent iteratively adjusts model weights to minimize a loss function, "
    "allowing the model to improve its predictions over time.",

    "Python is a high-level programming language known for its readability and "
    "versatility. It is widely used in data science, web development, and "
    "automation due to its extensive library ecosystem.",

    "Deep learning has enabled significant advances in computer vision and NLP. "
    "Convolutional neural networks excel at image recognition, while transformers "
    "have become the dominant architecture for language tasks.",

    "Natural language processing involves teaching computers to understand and "
    "generate human language. Tasks include translation, summarization, "
    "question answering, and sentiment analysis.",

    # Bahasa Indonesia — difilter jika target=en
    "Kucing itu duduk di atas tikar sambil berjemur di bawah sinar matahari.",

    # Spam / iklan
    "Buy cheap products now!!! Click here!!! Best deals ever!!!",
    "FREE MONEY!!! You have won $1,000,000!!! Click now!!!",

    # Noise / gibberish
    "asjdhaskjdhaksjdh random gibberish asdkjasd lkjhasd",
    "!!!! #### @@@@ %%%% **** $$$$",

    # Terlalu pendek
    "Hello world.",
    "This is short.",

    # Sangat repetitif
    "word " * 100,

    # DUPLIKAT EXACT — identik dengan dokumen pertama
    "The transformer architecture has revolutionized natural language processing. "
    "By replacing recurrent networks with self-attention mechanisms, transformers "
    "enable parallel processing of entire sequences at once.",

    # NEAR-DUPLIKAT — mirip dokumen pertama, beda beberapa kata
    "The transformer architecture has revolutionized natural language processing! "
    "By replacing recurrent networks with attention mechanisms, transformers "
    "enable parallel processing of entire input sequences at once.",

    # Duplikat dokumen ML
    "Machine learning models learn patterns from data through optimization. "
    "Gradient descent iteratively adjusts model weights to minimize a loss function, "
    "allowing the model to improve its predictions over time.",
]


# ---------------------------------------------------------------------------
# STEP 1: LANGUAGE FILTER
# ---------------------------------------------------------------------------

def step1_language_filter(docs: list, target_lang: str = "en") -> list:
    """
    Deteksi bahasa berdasarkan rasio karakter ASCII.
    Di produksi: fastText language detection (176 bahasa, <1ms/dok).

    Threshold: dokumen lolos jika ascii_ratio >= 0.85
    """
    def is_target_language(text):
        if len(text) == 0:
            return False
        ascii_ratio = sum(c.isascii() for c in text) / len(text)
        return ascii_ratio >= 0.85

    passed = [d for d in docs if is_target_language(d)]
    removed = [d for d in docs if not is_target_language(d)]

    print(f"\n{'─'*56}")
    print(f"STEP 1  Language filter  (target={target_lang})")
    print(f"{'─'*56}")
    print(f"  Input  : {len(docs)} dokumen")
    print(f"  Output : {len(passed)} dokumen")
    print(f"  Dibuang: {len(removed)} dokumen")
    for d in removed:
        print(f"    ✗ '{d[:55]}...'")
    return passed


# ---------------------------------------------------------------------------
# STEP 2: QUALITY FILTER
# ---------------------------------------------------------------------------

def step2_quality_filter(docs: list) -> list:
    """
    Filter heuristik kualitas dokumen. Kriteria:
      - panjang minimal 10 kata (demo; produksi: 50-200 kata)
      - rasio huruf alfabet >= 0.70
      - rasio tanda seru <= 0.02
      - rasio vocabulary (unique/total) >= 0.20  deteksi repetisi

    Di produksi: tambah ML classifier yang membedakan
    Wikipedia/buku (berkualitas) vs spam/SEO (tidak berkualitas).
    """
    def quality_check(text):
        words = text.split()

        if len(words) < 10:
            return False, f"terlalu pendek ({len(words)} kata, min=10)"

        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.70:
            return False, f"rasio huruf rendah ({alpha_ratio:.2f})"

        exclaim_ratio = text.count("!") / max(len(text), 1)
        if exclaim_ratio > 0.02:
            return False, f"terlalu banyak '!' ({exclaim_ratio:.3f})"

        vocab_ratio = len(set(words)) / max(len(words), 1)
        if vocab_ratio < 0.20:
            return False, f"terlalu repetitif (vocab={vocab_ratio:.2f})"

        return True, "ok"

    passed, failed = [], []
    for d in docs:
        ok, reason = quality_check(d)
        if ok:
            passed.append(d)
        else:
            failed.append((d, reason))

    print(f"\n{'─'*56}")
    print(f"STEP 2  Quality filter")
    print(f"{'─'*56}")
    print(f"  Input  : {len(docs)} dokumen")
    print(f"  Output : {len(passed)} dokumen")
    print(f"  Dibuang: {len(failed)} dokumen")
    for d, reason in failed:
        print(f"    ✗ [{reason}]  '{d[:40].strip()}...'")
    return passed


# ---------------------------------------------------------------------------
# STEP 3: EXACT DEDUPLICATION
# ---------------------------------------------------------------------------

def step3_exact_dedup(docs: list) -> list:
    """
    Buang dokumen identik menggunakan MD5 hash.
    Kompleksitas O(n) — cepat, cukup untuk duplikat exact.
    """
    seen, unique, dupes = set(), [], []

    for doc in docs:
        h = hashlib.md5(doc.strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
        else:
            dupes.append(doc)

    print(f"\n{'─'*56}")
    print(f"STEP 3  Exact deduplication (MD5)")
    print(f"{'─'*56}")
    print(f"  Input  : {len(docs)} dokumen")
    print(f"  Output : {len(unique)} dokumen")
    print(f"  Dibuang: {len(dupes)} duplikat exact")
    for d in dupes:
        print(f"    ✗ '{d[:55]}...'")
    return unique


# ---------------------------------------------------------------------------
# STEP 4: NEAR-DUPLICATE DETECTION (MinHash)
# ---------------------------------------------------------------------------

def get_ngrams(text, n=5):
    """Hasilkan set karakter n-gram dari teks."""
    text = text.lower().strip()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def minhash_signature(ngrams, num_hashes=64):
    """
    Simulasi MinHash signature.
    Di produksi: pakai library datasketch yang jauh lebih efisien.

    MinHash mengubah set n-gram menjadi vektor angka fixed-size
    yang bisa dipakai untuk estimasi Jaccard similarity antar dokumen
    tanpa membandingkan semua pasangan secara brute force.

    Properti kunci: P(sig[i] sama untuk A dan B) == Jaccard(A, B)
    """
    sig = []
    for seed in range(num_hashes):
        min_val = float("inf")
        for gram in ngrams:
            h = int(hashlib.md5(f"{seed}:{gram}".encode()).hexdigest(), 16)
            min_val = min(min_val, h)
        sig.append(min_val)
    return sig


def estimate_jaccard(sig1, sig2):
    """
    Estimasi Jaccard similarity dari dua MinHash signature.
    Jaccard(A,B) = |A ∩ B| / |A ∪ B|
    """
    matches = sum(a == b for a, b in zip(sig1, sig2))
    return matches / len(sig1)


def step4_near_dedup(docs: list, threshold: float = 0.7) -> list:
    """
    Buang dokumen mirip (near-duplicate) menggunakan MinHash.
    Threshold: Jaccard >= threshold dianggap near-duplicate.

    Di produksi: Locality Sensitive Hashing (LSH) menghindari
    O(n²) comparison — hanya bandingkan pasangan yang kemungkinan
    mirip berdasarkan bucket LSH yang sama.

    Threshold umum di paper besar: 0.7 - 0.8
    """
    print(f"\n{'─'*56}")
    print(f"STEP 4  Near-dedup (MinHash, threshold={threshold})")
    print(f"{'─'*56}")
    print("  Menghitung signatures...", end=" ", flush=True)

    signatures = []
    for doc in docs:
        ngrams = get_ngrams(doc, n=5)
        sig = minhash_signature(ngrams, num_hashes=64)
        signatures.append(sig)
    print("selesai")

    # Bandingkan semua pasangan O(n²) — ok untuk dataset kecil
    removed = set()
    for i in range(len(docs)):
        if i in removed:
            continue
        for j in range(i + 1, len(docs)):
            if j in removed:
                continue
            sim = estimate_jaccard(signatures[i], signatures[j])
            if sim >= threshold:
                removed.add(j)
                print(f"  ✗ near-dup terdeteksi (Jaccard ≈ {sim:.2f})")
                print(f"      keep   : '{docs[i][:48]}...'")
                print(f"      remove : '{docs[j][:48]}...'")

    unique = [docs[i] for i in range(len(docs)) if i not in removed]
    print(f"  Input  : {len(docs)} dokumen")
    print(f"  Output : {len(unique)} dokumen")
    print(f"  Dibuang: {len(removed)} near-duplikat")
    return unique


# ---------------------------------------------------------------------------
# STEP 5: TOKENIZATION
# ---------------------------------------------------------------------------

def step5_tokenize(docs: list, tokenizer) -> list:
    """
    Ubah teks menjadi token IDs menggunakan tokenizer BPE.
    Butuh: pip install transformers
    """
    tokenized = [tokenizer.encode(doc) for doc in docs]
    total = sum(len(t) for t in tokenized)

    print(f"\n{'─'*56}")
    print(f"STEP 5  Tokenization (BPE)")
    print(f"{'─'*56}")
    print(f"  Input  : {len(docs)} dokumen")
    print(f"  Total  : {total:,} token")
    print(f"  Rata2  : {total // max(len(docs), 1):,} token/dokumen")
    for doc, toks in zip(docs, tokenized):
        print(f"    [{len(toks):3d} tok] '{doc[:48]}...'")
    return tokenized


# ---------------------------------------------------------------------------
# STEP 6: PACKING KE FIXED-LENGTH CHUNKS
# ---------------------------------------------------------------------------

def step6_pack_chunks(token_lists: list, context_length: int = 128,
                      sep_token: int = 50256) -> list:
    """
    Kemas token dari banyak dokumen ke chunks fixed-length.

    Kenapa penting:
    - GPU butuh input ukuran sama (fixed batch size)
    - Menggabungkan dokumen pendek hemat padding overhead
    - Dokumen panjang dipotong ke context_length

    context_length kecil (128) untuk demo.
    Produksi: 2048 - 8192 token per chunk.
    sep_token 50256 = <|endoftext|> di GPT-2 vocabulary.

    Di produksi chunks disimpan ke file .bin + .idx untuk
    streaming efisien saat training tanpa load semua ke RAM.
    """
    buffer = []
    chunks = []

    for tokens in token_lists:
        buffer.extend(tokens + [sep_token])
        while len(buffer) >= context_length:
            chunks.append(buffer[:context_length])
            buffer = buffer[context_length:]

    # Pad sisa buffer jika lebih dari setengah penuh
    if len(buffer) > context_length // 2:
        buffer.extend([sep_token] * (context_length - len(buffer)))
        chunks.append(buffer)

    print(f"\n{'─'*56}")
    print(f"STEP 6  Packing (context_length={context_length})")
    print(f"{'─'*56}")
    print(f"  Output : {len(chunks)} chunks × {context_length} token")
    print(f"  Total  : {len(chunks) * context_length:,} token siap training")
    print(f"  Contoh chunk[0]: {chunks[0][:12]}...")
    return chunks


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 56)
    print("DATA PIPELINE PRE-TRAINING — SIMULASI LENGKAP")
    print("=" * 56)
    print(f"Data mentah: {len(RAW_DOCUMENTS)} dokumen")

    docs = RAW_DOCUMENTS[:]
    docs = step1_language_filter(docs, target_lang="en")
    docs = step2_quality_filter(docs)
    docs = step3_exact_dedup(docs)
    docs = step4_near_dedup(docs, threshold=0.7)

    # Step 5-6 butuh transformers — coba load, skip jika tidak ada
    try:
        from transformers import AutoTokenizer
        print("\nLoading tokenizer GPT-2...")
        tok = AutoTokenizer.from_pretrained("gpt2")
        token_lists = step5_tokenize(docs, tok)
        chunks = step6_pack_chunks(token_lists, context_length=128)
    except Exception:
        print("\n[Skip step 5-6: transformers/internet tidak tersedia]")
        print("Jalankan: pip install transformers  lalu coba lagi.")
        chunks = []

    # Ringkasan akhir
    print(f"\n{'='*56}")
    print("RINGKASAN PIPELINE")
    print(f"{'='*56}")
    print(f"  Raw documents   : {len(RAW_DOCUMENTS):>4}")
    print(f"  Setelah lang    : {len(RAW_DOCUMENTS)-1:>4}  (-1 non-English)")
    print(f"  Setelah quality : sesuai output step 2 di atas")
    print(f"  Dokumen final   : {len(docs):>4}")
    if chunks:
        print(f"  Chunks training : {len(chunks):>4}  × 128 token")

    print(f"""
Skala produksi nyata:
  400 TB raw  →  40 TB (lang filter)
             →  15 TB (quality filter)
             →   8 TB (dedup)
             →   1-3 triliun token (tokenize + pack)

  Pipeline yang sama, skala ~10⁹ kali lebih besar. :)
    """)


if __name__ == "__main__":
    main()
