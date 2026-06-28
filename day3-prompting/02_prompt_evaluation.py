"""
Hari 3 - Evaluasi Prompt Secara Sistematis
============================================
Framework untuk mengukur dan membandingkan performa beberapa versi
prompt secara kuantitatif — bukan cuma feeling.

Pendekatan ini mirip A/B testing di ML: buat test set dengan ground
truth, jalankan tiap prompt versi, hitung metrik, pilih yang terbaik.

Setup:
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."

Jalankan: python 02_prompt_evaluation.py
"""

import os
import json
import time
from dataclasses import dataclass, field
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# TEST DATASET — sentimen klasifikasi dengan ground truth
# ---------------------------------------------------------------------------

TEST_CASES = [
    # Positif jelas
    {"input": "Produk bagus, pengiriman cepat, penjual responsif!", "expected": "Positif"},
    {"input": "Kualitas premium, worth every penny, sangat puas.", "expected": "Positif"},
    {"input": "Persis seperti foto, kondisi mulus, bintang 5!", "expected": "Positif"},

    # Negatif jelas
    {"input": "Barang rusak saat diterima, sangat kecewa.", "expected": "Negatif"},
    {"input": "Tidak sesuai deskripsi sama sekali, minta refund.", "expected": "Negatif"},
    {"input": "Pengiriman lama 3 minggu, produk sudah expired.", "expected": "Negatif"},

    # Netral / ambigu
    {"input": "Ukurannya standar, tidak lebih tidak kurang.", "expected": "Netral"},
    {"input": "Sesuai harga yang dibayar, biasa saja.", "expected": "Netral"},
    {"input": "Sudah sampai, packaging oke.", "expected": "Netral"},

    # Edge cases
    {"input": "Lumayan sih, tapi ada yang lebih bagus.", "expected": "Netral"},
    {"input": "Harganya murah tapi kualitas juga murahan.", "expected": "Negatif"},
    {"input": "Surprisingly bagus untuk harga segini!", "expected": "Positif"},
]


# ---------------------------------------------------------------------------
# VERSI PROMPT YANG AKAN DIBANDINGKAN
# ---------------------------------------------------------------------------

PROMPT_VERSIONS = {
    "v1_zero_shot": {
        "description": "Zero-shot tanpa format instruksi",
        "template": "Klasifikasikan sentimen: '{input}'",
        "system": None,
    },
    "v2_format_specified": {
        "description": "Zero-shot dengan format output eksplisit",
        "template": (
            "Klasifikasikan sentimen ulasan berikut.\n"
            "Jawab HANYA dengan satu kata: Positif / Negatif / Netral.\n\n"
            "Ulasan: '{input}'"
        ),
        "system": None,
    },
    "v3_few_shot": {
        "description": "Few-shot dengan 3 contoh",
        "template": (
            "Klasifikasikan sentimen. Jawab HANYA: Positif / Negatif / Netral.\n\n"
            "Ulasan: 'Produk bagus, sangat puas!'\n"
            "Sentimen: Positif\n\n"
            "Ulasan: 'Barang rusak, sangat kecewa.'\n"
            "Sentimen: Negatif\n\n"
            "Ulasan: 'Sesuai deskripsi, biasa saja.'\n"
            "Sentimen: Netral\n\n"
            "Ulasan: '{input}'\n"
            "Sentimen:"
        ),
        "system": None,
    },
    "v4_role_few_shot": {
        "description": "Role prompting + few-shot",
        "template": (
            "Klasifikasikan sentimen ulasan produk e-commerce.\n"
            "Jawab HANYA dengan satu kata: Positif / Negatif / Netral.\n\n"
            "Contoh:\n"
            "- 'Produk bagus!' → Positif\n"
            "- 'Barang rusak.' → Negatif\n"
            "- 'Sesuai deskripsi.' → Netral\n\n"
            "Ulasan: '{input}'"
        ),
        "system": (
            "Kamu adalah analis sentimen e-commerce yang sangat teliti. "
            "Berikan klasifikasi yang konsisten dan objektif."
        ),
    },
}


# ---------------------------------------------------------------------------
# EVALUASI
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    version: str
    description: str
    correct: int = 0
    total: int = 0
    errors: list = field(default_factory=list)
    latency_ms: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    @property
    def avg_latency(self) -> float:
        return sum(self.latency_ms) / max(len(self.latency_ms), 1)


def normalize_output(raw: str) -> str:
    """Normalisasi output model untuk perbandingan yang adil."""
    raw = raw.strip().lower()
    if "positif" in raw:
        return "Positif"
    if "negatif" in raw:
        return "Negatif"
    if "netral" in raw:
        return "Netral"
    return raw.capitalize()


def evaluate_prompt(version_name: str, config: dict) -> EvalResult:
    """Jalankan satu versi prompt pada semua test case."""
    result = EvalResult(
        version=version_name,
        description=config["description"],
        total=len(TEST_CASES),
    )

    for case in TEST_CASES:
        prompt = config["template"].format(input=case["input"])
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": MODEL,
            "max_tokens": 20,  # klasifikasi cukup pendek
            "messages": messages,
        }
        if config.get("system"):
            kwargs["system"] = config["system"]

        start = time.time()
        try:
            response = client.messages.create(**kwargs)
            elapsed_ms = (time.time() - start) * 1000
            result.latency_ms.append(elapsed_ms)

            raw_output = response.content[0].text
            predicted = normalize_output(raw_output)
            expected = case["expected"]

            if predicted == expected:
                result.correct += 1
            else:
                result.errors.append({
                    "input": case["input"],
                    "expected": expected,
                    "predicted": predicted,
                    "raw_output": raw_output.strip(),
                })
        except Exception as e:
            result.errors.append({
                "input": case["input"],
                "error": str(e),
            })

        # Rate limit — jangan terlalu cepat
        time.sleep(0.3)

    return result


def print_report(results: list):
    """Tampilkan laporan perbandingan."""
    print(f"\n{'='*65}")
    print("LAPORAN EVALUASI PROMPT")
    print(f"{'='*65}")
    print(f"Test set: {len(TEST_CASES)} kasus")
    print(f"Model   : {MODEL}\n")

    # Tabel ringkasan
    header = f"{'Versi':<22} {'Desc':<28} {'Acc':>6} {'Latency':>9}"
    print(header)
    print("─" * 68)

    results.sort(key=lambda r: r.accuracy, reverse=True)
    for r in results:
        bar = "█" * int(r.accuracy * 20)
        print(
            f"{r.version:<22} "
            f"{r.description[:27]:<28} "
            f"{r.accuracy*100:5.1f}% "
            f"{r.avg_latency:7.0f}ms"
        )

    # Detail error versi terbaik
    best = results[0]
    print(f"\n{'─'*65}")
    print(f"Detail error — {best.version} (akurasi terbaik):")
    if not best.errors:
        print("  Tidak ada error! 🎉")
    else:
        for err in best.errors:
            print(
                f"  ✗ '{err.get('input', '')[:40]}...'\n"
                f"      expected={err.get('expected')} "
                f"predicted={err.get('predicted')} "
                f"raw='{err.get('raw_output', '')}'"
            )

    print(f"\n{'─'*65}")
    print("Rekomendasi:")
    print(f"  Gunakan '{best.version}' — akurasi {best.accuracy*100:.1f}%")
    if len(results) > 1:
        worst = results[-1]
        improvement = (best.accuracy - worst.accuracy) * 100
        print(
            f"  Improvement vs '{worst.version}': "
            f"+{improvement:.1f} percentage points"
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Mengevaluasi semua versi prompt...")
    print(f"Total API calls: {len(PROMPT_VERSIONS) * len(TEST_CASES)}\n")

    results = []
    for version_name, config in PROMPT_VERSIONS.items():
        print(f"Testing {version_name}: {config['description']}...")
        result = evaluate_prompt(version_name, config)
        results.append(result)
        print(
            f"  Selesai: {result.correct}/{result.total} benar "
            f"({result.accuracy*100:.1f}%)"
        )

    print_report(results)
