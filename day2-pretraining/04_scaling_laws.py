"""
Hari 2 - Scaling Laws (Chinchilla)
=====================================
Kalkulator sederhana berdasarkan Chinchilla scaling laws (Hoffmann et al., 2022).
Temuan kunci: untuk model N parameter, token training optimal ≈ 20 × N.

Paper: "Training Compute-Optimal Large Language Models"
       DeepMind, 2022

Jalankan: python 04_scaling_laws.py
Requirement: tidak ada (pure Python)
"""


def optimal_tokens(num_params: float) -> float:
    """Jumlah token training optimal untuk model dengan num_params parameter."""
    return 20 * num_params


def compute_flops(num_params: float, num_tokens: float) -> float:
    """
    Estimasi total FLOPs training.
    Rumus approx: FLOPs ≈ 6 × N × D
    (6 karena: 2 untuk forward, 4 untuk backward — approx rule of thumb)
    """
    return 6 * num_params * num_tokens


def gpu_days(flops: float, gpu_tflops: float = 312e12) -> float:
    """
    Estimasi waktu training dalam GPU-days.
    Default: A100 80GB ≈ 312 TFLOPS (bf16).
    Asumsi utilization GPU 40% (realistic).
    """
    utilization = 0.4
    effective_flops_per_sec = gpu_tflops * utilization
    seconds = flops / effective_flops_per_sec
    return seconds / (60 * 60 * 24)


def format_number(n: float) -> str:
    """Format angka besar jadi readable."""
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    return str(n)


def main():
    print("="*65)
    print("Chinchilla Scaling Laws — berapa token optimal untuk model LLM")
    print("="*65)
    print("Rumus: token_optimal ≈ 20 × jumlah_parameter")
    print()

    models = [
        ("GPT-2 base",        124e6),
        ("GPT-2 XL",          1.5e9),
        ("LLaMA-2 7B",        7e9),
        ("LLaMA-3 8B",        8e9),
        ("LLaMA-2 13B",       13e9),
        ("LLaMA-3 70B",       70e9),
        ("GPT-3",             175e9),
        ("GPT-4 (estimasi)",  1.8e12),
    ]

    header = f"{'Model':<22} {'Params':>8} {'Token Optimal':>14} {'FLOPs':>12} {'GPU-days (A100)':>16}"
    print(header)
    print("-" * 75)

    for name, params in models:
        tokens = optimal_tokens(params)
        flops  = compute_flops(params, tokens)
        days   = gpu_days(flops)

        print(
            f"{name:<22} "
            f"{format_number(params):>8} "
            f"{format_number(tokens):>14} "
            f"{format_number(flops):>12} "
            f"{days:>14.0f}d"
        )

    print()
    print("="*65)
    print("Catatan penting dari paper Chinchilla:")
    print("="*65)
    print("""
    Sebelum Chinchilla (2022), banyak model dilatih dengan paradigma:
    "buat model sebesar mungkin, latih dengan data seadanya."
    → GPT-3 175B hanya dilatih pada 300B token (underfitted secara Chinchilla)

    Chinchilla membuktikan: model lebih kecil + data lebih banyak
    menghasilkan performa yang sama atau lebih baik, SEKALIGUS:
    - lebih murah saat inference (lebih sedikit parameter = lebih cepat)
    - lebih mudah di-deploy di hardware terbatas

    Ini mengubah industri: LLaMA (Meta) adalah implementasi publik
    pertama yang benar-benar serius menerapkan Chinchilla optimal ratio.
    LLaMA-3 8B dilatih pada 15T token — jauh melampaui minimum Chinchilla
    (160B token) karena inference efficiency lebih penting dari training cost.
    """)

    print("="*65)
    print("Simulasi: berapa lama melatih GPT-2 di MacBook Pro kamu?")
    print("="*65)
    params = 124e6          # GPT-2 base
    tokens = optimal_tokens(params)
    flops  = compute_flops(params, tokens)

    macbook_tflops = 11e12  # M2 Pro Neural Engine, fp16, no utilization discount
    macbook_days = flops / (macbook_tflops * 86400)

    a100_days = gpu_days(flops)

    print(f"Model   : GPT-2 ({format_number(params)} params)")
    print(f"Tokens  : {format_number(tokens)} (Chinchilla optimal)")
    print(f"FLOPs   : {format_number(flops)}")
    print(f"  A100 GPU      : ~{a100_days:.1f} GPU-days")
    print(f"  MacBook M2 Pro: ~{macbook_days:.0f} days (~{macbook_days/365:.1f} tahun)")
    print("\nItulah kenapa pre-training tidak dilakukan di laptop. :)")


if __name__ == "__main__":
    main()
