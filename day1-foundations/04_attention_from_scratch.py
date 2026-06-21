"""
Hari 1 - Demo Manual: Embedding & Self-Attention dari Nol
=============================================================
Mendemonstrasikan secara manual (tanpa library deep learning):
1. Embedding sebagai koordinat di ruang vektor (analogi raja-ratu-kucing)
2. Self-attention dengan proyeksi W_Q, W_K, W_V acak (bukan identitas)

Jalankan: python 04_attention_from_scratch.py
Requirement: pip install numpy
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)


def embedding_analogy_demo():
    """Buktikan raja - pria + wanita ~= ratu, dan kata sejenis berdekatan."""
    print(f"\n{'='*60}")
    print("Demo 1: Embedding sebagai koordinat di ruang vektor")
    print(f"{'='*60}")

    words = ["raja", "ratu", "pria", "wanita", "kucing", "anjing", "kelinci"]
    E = np.array([
        [4.0, 1.0],   # raja
        [4.0, 4.0],   # ratu
        [1.0, 1.0],   # pria
        [1.0, 4.0],   # wanita
        [-3.0, 2.0],  # kucing
        [-2.5, 2.5],  # anjing
        [-3.2, 1.6],  # kelinci
    ])

    for w, v in zip(words, E):
        print(f"  {w:8s} = {v}")

    analogy_result = E[0] - E[2] + E[3]  # raja - pria + wanita
    print(f"\nraja - pria + wanita = {analogy_result}")
    print(f"vektor ratu asli     = {E[1]}")

    def dist(a, b):
        return round(float(np.linalg.norm(a - b)), 2)

    print(f"\nJarak raja <-> ratu     : {dist(E[0], E[1])}  (sama kelompok: kerajaan)")
    print(f"Jarak kucing <-> anjing : {dist(E[4], E[5])}  (sama kelompok: hewan)")
    print(f"Jarak raja <-> kucing   : {dist(E[0], E[4])}  (beda kelompok)")


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def self_attention_demo():
    """Self-attention 3 token dengan proyeksi W_Q, W_K, W_V acak (non-identitas)."""
    print(f"\n{'='*60}")
    print("Demo 2: Self-attention dengan proyeksi W_Q, W_K, W_V sungguhan")
    print(f"{'='*60}")

    tokens = ["Kucing", "duduk", "diam"]
    # d_model = 4. "duduk" dan "diam" sengaja dibuat mirip (sama2 kata keadaan)
    X = np.array([
        [1.0, 0.0, 0.5, 0.0],   # Kucing
        [0.0, 1.0, 0.0, 0.5],   # duduk
        [0.0, 0.8, 0.0, 0.4],   # diam
    ])

    np.random.seed(7)
    W_Q = np.round(np.random.randn(4, 4) * 0.5, 2)
    W_K = np.round(np.random.randn(4, 4) * 0.5, 2)
    W_V = np.round(np.random.randn(4, 4) * 0.5, 2)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    print(f"Tokens: {tokens}")
    print(f"\nQ =\n{Q}")
    print(f"\nK =\n{K}")
    print(f"\nV =\n{V}")

    d_k = X.shape[1]
    scores = Q @ K.T
    scaled = scores / np.sqrt(d_k)
    weights = softmax(scaled)

    print(f"\nScores (Q @ Kt) =\n{scores}")
    print(f"\nAttention weights (setelah scale + softmax) =\n{weights}")
    print("\nCatatan: tiap baris berjumlah 1.0 -> itu distribusi attention satu token")
    for i, t in enumerate(tokens):
        print(f"  jumlah baris '{t}': {weights[i].sum():.4f}")

    output = weights @ V
    print(f"\nOutput (weights @ V) =\n{output}")
    print("\nOutput berukuran sama dengan X -> tiap token kini representasi")
    print("kontekstual, hasil campuran berbobot dari semua token lain.")


if __name__ == "__main__":
    embedding_analogy_demo()
    self_attention_demo()
