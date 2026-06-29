"""
Hari 4 - Evaluasi RAG dengan RAGAS Metrics
===========================================
Framework evaluasi untuk mengukur kualitas RAG pipeline secara
kuantitatif. Empat metrik utama RAGAS diimplementasikan dengan LLM-as-judge.

Metrik:
  1. Faithfulness      — apakah jawaban bisa dibuktikan dari konteks?
  2. Answer Relevancy  — apakah jawaban relevan dengan pertanyaan?
  3. Context Precision — apakah chunk yang diambil memang berguna?
  4. Context Recall    — apakah chunk mencakup semua info yang dibutuhkan?

Setup:
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."

Jalankan: python 03_rag_evaluation.py
"""

import os
import json
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# TEST SET — pertanyaan + jawaban ideal + konteks yang seharusnya diambil
# ---------------------------------------------------------------------------

EVAL_DATASET = [
    {
        "question": "Apa saja dokumen yang dibutuhkan untuk klaim jiwa?",
        "ground_truth": (
            "Dokumen yang dibutuhkan: akta kematian asli, polis asuransi asli, "
            "KTP tertanggung, KTP pemohon, dan surat keterangan dokter. "
            "Pengajuan maksimal 60 hari sejak kejadian."
        ),
        "retrieved_context": (
            "Untuk mengajukan klaim asuransi jiwa, tertanggung atau ahli waris "
            "harus menyiapkan: akta kematian asli, polis asuransi asli, KTP "
            "tertanggung, KTP pemohon, dan surat keterangan dokter. "
            "Pengajuan maksimal 60 hari sejak kejadian."
        ),
        "generated_answer": (
            "Dokumen yang diperlukan untuk klaim jiwa adalah: akta kematian "
            "asli, polis asuransi asli, KTP tertanggung, KTP pemohon klaim, "
            "dan surat keterangan dokter atau rumah sakit."
        ),
    },
    {
        "question": "Apakah kematian akibat olahraga ekstrem ditanggung?",
        "ground_truth": (
            "Klaim tidak dibayarkan untuk kematian akibat olahraga ekstrem "
            "yang tidak didaftarkan ke perusahaan asuransi."
        ),
        "retrieved_context": (
            "Klaim tidak dibayarkan untuk: kematian bunuh diri dalam 2 tahun "
            "pertama, kematian akibat pelanggaran hukum, kematian akibat perang, "
            "olahraga ekstrem tidak terdaftar, dan kondisi pre-existing yang "
            "tidak diungkapkan."
        ),
        "generated_answer": (
            "Kematian akibat olahraga ekstrem TIDAK ditanggung jika olahraga "
            "tersebut tidak didaftarkan sebelumnya ke perusahaan asuransi. "
            "Hal ini termasuk dalam daftar pengecualian klaim."
        ),
    },
    {
        "question": "Berapa bunga yang dikenakan untuk tunggakan premi reinstatement?",
        "ground_truth": (
            "Informasi besaran bunga tunggakan tidak disebutkan secara spesifik "
            "dalam dokumen kebijakan yang tersedia."
        ),
        "retrieved_context": (
            "Polis lapse dapat dipulihkan dalam 2 tahun sejak lapse. "
            "Syarat: lunasi semua tunggakan premi beserta bunga, isi ulang "
            "formulir kesehatan, mungkin diperlukan pemeriksaan medis baru."
        ),
        "generated_answer": (
            "Dokumen yang saya miliki menyebutkan bahwa tunggakan premi harus "
            "dilunasi beserta bunganya, namun besaran persentase bunga yang "
            "dikenakan tidak disebutkan secara spesifik. Silakan hubungi "
            "customer service untuk informasi lebih lanjut."
        ),
    },
]


# ---------------------------------------------------------------------------
# LLM-AS-JUDGE EVALUASI
# ---------------------------------------------------------------------------

def evaluate_faithfulness(answer: str, context: str) -> dict:
    """
    Faithfulness: apakah setiap klaim dalam jawaban dapat dibuktikan
    dari konteks? Nilai 0-1.

    0 = jawaban berisi informasi yang tidak ada di konteks (hallucination)
    1 = semua klaim dalam jawaban ada dukungannya di konteks
    """
    prompt = f"""Evaluasi apakah JAWABAN sepenuhnya didukung oleh KONTEKS.

KONTEKS:
{context}

JAWABAN:
{answer}

Instruksi:
- Periksa tiap klaim/pernyataan dalam jawaban
- Apakah ada pernyataan yang tidak ada dukungannya di konteks?
- Skor 1.0 = semua klaim didukung konteks
- Skor 0.0 = ada klaim yang tidak ada di konteks (hallucination)

Jawab HANYA JSON:
{{"score": 0.0-1.0, "reason": "penjelasan singkat", "hallucinated_claims": []}}"""

    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ]
    )
    try:
        return json.loads("{" + resp.content[0].text)
    except Exception:
        return {"score": None, "reason": "parse error"}


def evaluate_answer_relevancy(question: str, answer: str) -> dict:
    """
    Answer relevancy: seberapa relevan jawaban terhadap pertanyaan?
    Jawaban yang relevan tapi salah tetap mendapat skor rendah.
    """
    prompt = f"""Evaluasi seberapa relevan JAWABAN terhadap PERTANYAAN.

PERTANYAAN:
{question}

JAWABAN:
{answer}

Instruksi:
- Skor 1.0 = jawaban langsung menjawab pertanyaan
- Skor 0.5 = jawaban sebagian relevan, ada yang tidak perlu
- Skor 0.0 = jawaban sama sekali tidak relevan atau menghindari pertanyaan

Jawab HANYA JSON:
{{"score": 0.0-1.0, "reason": "penjelasan singkat"}}"""

    resp = client.messages.create(
        model=MODEL, max_tokens=200,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ]
    )
    try:
        return json.loads("{" + resp.content[0].text)
    except Exception:
        return {"score": None, "reason": "parse error"}


def evaluate_context_precision(question: str, context: str) -> dict:
    """
    Context precision: apakah semua chunk yang diambil memang berguna
    untuk menjawab pertanyaan? Chunk tidak relevan menurunkan skor.
    """
    prompt = f"""Evaluasi apakah KONTEKS yang diambil relevan untuk menjawab PERTANYAAN.

PERTANYAAN:
{question}

KONTEKS:
{context}

Instruksi:
- Skor 1.0 = semua bagian konteks relevan untuk menjawab pertanyaan
- Skor 0.5 = sebagian konteks relevan, sebagian tidak perlu diambil
- Skor 0.0 = konteks tidak relevan sama sekali dengan pertanyaan

Jawab HANYA JSON:
{{"score": 0.0-1.0, "reason": "penjelasan singkat"}}"""

    resp = client.messages.create(
        model=MODEL, max_tokens=200,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ]
    )
    try:
        return json.loads("{" + resp.content[0].text)
    except Exception:
        return {"score": None, "reason": "parse error"}


def evaluate_context_recall(question: str, context: str,
                             ground_truth: str) -> dict:
    """
    Context recall: apakah konteks yang diambil mencakup semua informasi
    yang dibutuhkan untuk menjawab pertanyaan secara lengkap?
    """
    prompt = f"""Evaluasi apakah KONTEKS mengandung semua informasi yang dibutuhkan
untuk menjawab PERTANYAAN sesuai JAWABAN IDEAL.

PERTANYAAN:
{question}

JAWABAN IDEAL (ground truth):
{ground_truth}

KONTEKS YANG DIAMBIL:
{context}

Instruksi:
- Skor 1.0 = semua informasi dalam jawaban ideal ada di konteks
- Skor 0.5 = sebagian informasi penting ada, sebagian hilang
- Skor 0.0 = konteks tidak mengandung informasi yang dibutuhkan

Jawab HANYA JSON:
{{"score": 0.0-1.0, "reason": "penjelasan singkat", "missing_info": []}}"""

    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ]
    )
    try:
        return json.loads("{" + resp.content[0].text)
    except Exception:
        return {"score": None, "reason": "parse error"}


# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------

def run_evaluation():
    print("=" * 60)
    print("Evaluasi RAG Pipeline — RAGAS Metrics")
    print("=" * 60)
    print(f"Test set: {len(EVAL_DATASET)} kasus\n")

    all_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for i, case in enumerate(EVAL_DATASET, 1):
        print(f"{'─'*60}")
        print(f"Kasus {i}: {case['question'][:55]}...")

        # Evaluasi 4 metrik
        faith = evaluate_faithfulness(
            case["generated_answer"], case["retrieved_context"]
        )
        relev = evaluate_answer_relevancy(
            case["question"], case["generated_answer"]
        )
        prec  = evaluate_context_precision(
            case["question"], case["retrieved_context"]
        )
        rec   = evaluate_context_recall(
            case["question"], case["retrieved_context"], case["ground_truth"]
        )

        scores = {
            "faithfulness":     faith.get("score"),
            "answer_relevancy": relev.get("score"),
            "context_precision": prec.get("score"),
            "context_recall":   rec.get("score"),
        }

        for metric, score in scores.items():
            if score is not None:
                all_scores[metric].append(score)
            bar = "█" * int((score or 0) * 10)
            print(f"  {metric:<22} {score:.2f}  {bar}")

        # Tampilkan alasan untuk skor rendah
        if (faith.get("score") or 1) < 0.8:
            print(f"  ⚠ Faithfulness: {faith.get('reason', '')[:60]}")
        if faith.get("hallucinated_claims"):
            print(f"  ⚠ Hallucinated: {faith['hallucinated_claims']}")

    # Ringkasan
    print(f"\n{'='*60}")
    print("RINGKASAN — Rata-rata skor")
    print(f"{'='*60}")
    for metric, scores_list in all_scores.items():
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            bar = "█" * int(avg * 20)
            status = "✓" if avg >= 0.8 else "⚠" if avg >= 0.6 else "✗"
            print(f"  {status} {metric:<22} {avg:.3f}  {bar}")

    print(f"""
Interpretasi:
  ≥ 0.8 = bagus, siap produksi
  0.6–0.8 = perlu tuning (chunk size, top_k, prompt)
  < 0.6   = ada masalah fundamental (data, embedding, atau prompt)

Cara improve skor rendah:
  Faithfulness rendah     → perbaiki prompt, tambah instruksi "hanya dari konteks"
  Answer relevancy rendah → perbaiki prompt, tambah few-shot
  Context precision rendah → kurangi top_k atau tambah reranker
  Context recall rendah   → perkecil chunk size atau tambah top_k
    """)


if __name__ == "__main__":
    run_evaluation()
