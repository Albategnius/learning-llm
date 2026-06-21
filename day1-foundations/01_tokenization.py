"""
Hari 1 - Eksplorasi Tokenisasi
================================
Membandingkan bagaimana tokenizer berbeda (BPE vs WordPiece, English-centric
vs Indonesia-aware) memecah teks bahasa Indonesia jadi subword units.

Jalankan: python 01_tokenization.py
Requirement: pip install transformers
"""

from transformers import AutoTokenizer


def explore_tokenizer(model_name: str, sentence: str):
    print(f"\n{'='*60}")
    print(f"Tokenizer: {model_name}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(model_name)

    tokens = tok.tokenize(sentence)
    ids = tok.convert_tokens_to_ids(tokens)

    print(f"Kalimat asli : {sentence}")
    print(f"Jumlah token : {len(tokens)}")
    print(f"Tokens       : {tokens}")
    print(f"Token IDs    : {ids}")

    # Decode balik untuk verifikasi round-trip
    decoded = tok.decode(tok.encode(sentence))
    print(f"Decode balik : {decoded}")


def subword_splitting_demo(model_name: str = "gpt2"):
    """Tunjukkan bagaimana kata jarang dipecah jadi subword unit."""
    tok = AutoTokenizer.from_pretrained(model_name)

    words = ["unhappiness", "bermalas-malasan", "tokenization", "transformer"]

    print(f"\n{'='*60}")
    print("Demo subword splitting")
    print(f"{'='*60}")
    for w in words:
        tokens = tok.tokenize(w)
        ids = tok.convert_tokens_to_ids(tokens)
        print(f"{w:20s} -> tokens={tokens}  ids={ids}")


if __name__ == "__main__":
    sentence = "Kucing itu duduk di atas tikar sambil bermalas-malasan."

    # BPE, dilatih dominan korpus Inggris
    explore_tokenizer("gpt2", sentence)

    # WordPiece, dilatih khusus korpus bahasa Indonesia
    explore_tokenizer("indobenchmark/indobert-base-p1", sentence)

    subword_splitting_demo("gpt2")
