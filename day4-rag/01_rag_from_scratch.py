"""
Hari 4 - RAG dari Scratch (tanpa library eksternal)
=====================================================
Implementasi RAG minimal menggunakan numpy untuk vector similarity
dan Anthropic API untuk embedding + generation.

Tujuan: pahami mekanisme RAG tanpa abstraksi library,
sebelum pakai LangChain/LlamaIndex di file berikutnya.

Dokumen contoh: kebijakan asuransi fiktif (relevan dengan
environment asuransi yang terlihat di path kamu)

Setup:
  pip install anthropic numpy
  export ANTHROPIC_API_KEY="sk-ant-..."

Jalankan: python 01_rag_from_scratch.py
"""

import os
import json
import numpy as np
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
EMBED_MODEL = "voyage-3"          # Anthropic embedding model
GEN_MODEL   = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# 1. DOKUMEN — corpus asuransi fiktif
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "id": "pol-001",
        "title": "Syarat Pengajuan Klaim Asuransi Jiwa",
        "content": (
            "Untuk mengajukan klaim asuransi jiwa, tertanggung atau ahli waris "
            "harus menyiapkan dokumen berikut: akta kematian asli, polis asuransi "
            "asli, KTP tertanggung, KTP pemohon klaim, dan surat keterangan dokter "
            "atau rumah sakit. Pengajuan klaim dilakukan maksimal 60 hari sejak "
            "tanggal kejadian. Klaim yang melewati batas waktu akan ditolak kecuali "
            "ada alasan force majeure yang dapat dibuktikan."
        ),
    },
    {
        "id": "pol-002",
        "title": "Pengecualian Klaim Asuransi Jiwa",
        "content": (
            "Klaim tidak akan dibayarkan dalam kondisi berikut: kematian akibat "
            "bunuh diri dalam 2 tahun pertama polis aktif, kematian akibat "
            "perbuatan melawan hukum, kematian akibat perang atau kerusuhan sipil, "
            "kematian akibat olahraga ekstrem yang tidak didaftarkan, dan kematian "
            "yang disebabkan kondisi pre-existing yang tidak diungkapkan saat "
            "pengajuan polis. Semua pengecualian ini tercantum dalam Pasal 7 polis."
        ),
    },
    {
        "id": "pol-003",
        "title": "Premi dan Manfaat Asuransi Jiwa Berjangka",
        "content": (
            "Premi asuransi jiwa berjangka dihitung berdasarkan usia, jenis kelamin, "
            "status merokok, dan uang pertanggungan yang dipilih. Premi bulanan "
            "untuk uang pertanggungan Rp 500 juta, usia 35 tahun, non-perokok "
            "berkisar Rp 350.000 hingga Rp 500.000 tergantung masa pertanggungan. "
            "Manfaat kematian dibayarkan sekaligus kepada ahli waris yang ditunjuk "
            "dalam polis. Tidak ada nilai tunai untuk produk jiwa berjangka."
        ),
    },
    {
        "id": "pol-004",
        "title": "Prosedur Investigasi Klaim",
        "content": (
            "Tim investigasi klaim akan memeriksa semua klaim di atas Rp 200 juta. "
            "Proses investigasi mencakup verifikasi dokumen, wawancara ahli waris, "
            "koordinasi dengan rumah sakit atau dokter yang merawat, dan pemeriksaan "
            "riwayat medis tertanggung. Investigasi diselesaikan dalam 30 hari kerja. "
            "Jika diperlukan investigasi lapangan, waktu dapat diperpanjang 15 hari "
            "dengan pemberitahuan tertulis kepada pemohon."
        ),
    },
    {
        "id": "pol-005",
        "title": "Reinstatement Polis yang Lapse",
        "content": (
            "Polis yang lapse karena tidak membayar premi dapat dipulihkan "
            "(reinstatement) dalam waktu 2 tahun sejak tanggal lapse. "
            "Syarat reinstatement: melunasi semua tunggakan premi beserta bunga, "
            "mengisi ulang formulir kesehatan, dan mungkin diperlukan pemeriksaan "
            "medis baru tergantung usia dan uang pertanggungan. Setelah reinstatement, "
            "masa tunggu pengecualian bunuh diri mulai dihitung ulang dari awal."
        ),
    },
    {
        "id": "pol-006",
        "title": "Klaim Cacat Total dan Permanen",
        "content": (
            "Manfaat cacat total dan permanen (CTP) dibayarkan jika tertanggung "
            "mengalami kehilangan fungsi tubuh permanen yang mencegahnya bekerja "
            "seumur hidup. Kondisi yang memenuhi syarat CTP: kehilangan kedua tangan "
            "atau kedua kaki, kehilangan satu tangan dan satu kaki, kebutaan total "
            "permanen pada kedua mata, atau kondisi medis permanen lain yang "
            "ditetapkan tim dokter independen. Manfaat CTP dibayarkan sekaligus "
            "dan polis berakhir setelah pembayaran."
        ),
    },
    {
        "id": "pol-007",
        "title": "Asuransi Kesehatan — Rawat Inap",
        "content": (
            "Manfaat rawat inap mencakup biaya kamar dan makan sesuai kelas yang "
            "dipilih (kelas 1, 2, atau VIP), biaya dokter, biaya obat-obatan yang "
            "diresepkan dokter, biaya laboratorium dan radiologi, dan biaya tindakan "
            "medis. Batas tahunan manfaat rawat inap mulai dari Rp 100 juta hingga "
            "Rp 1 miliar tergantung paket yang dipilih. Rumah sakit rekanan tersedia "
            "di lebih dari 3.000 jaringan nasional dengan sistem cashless."
        ),
    },
    {
        "id": "pol-008",
        "title": "Fraud Detection dan Sanksi",
        "content": (
            "Perusahaan menerapkan sistem deteksi fraud berbasis machine learning "
            "untuk mengidentifikasi pola klaim yang mencurigakan. Klaim yang terindikasi "
            "fraud akan diinvestigasi lebih lanjut oleh tim khusus. Jika terbukti "
            "melakukan fraud, polis dibatalkan, klaim ditolak, dan pemohon dapat "
            "dilaporkan ke pihak berwenang. Data fraud sharing antar perusahaan "
            "asuransi dilakukan melalui database AAUI untuk mencegah fraud berulang."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. CHUNKING
# ---------------------------------------------------------------------------

def chunk_documents(
    docs: list,
    chunk_size: int = 300,
    overlap: int = 50
) -> list:
    """
    Potong dokumen jadi chunks dengan overlap.

    chunk_size: karakter per chunk (bukan token — perlu dikali ~4 untuk token)
    overlap: berapa karakter yang diulang antara chunk berurutan.
             Overlap penting supaya informasi di batas potongan tidak hilang.
    """
    chunks = []
    for doc in docs:
        text = doc["content"]
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": f"{doc['id']}-chunk{chunk_idx}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "text": chunk_text,
                "start_char": start,
            })

            chunk_idx += 1
            if end >= len(text):
                break
            start = end - overlap  # mundur overlap karakter untuk kontinuitas

    return chunks


# ---------------------------------------------------------------------------
# 3. EMBEDDING
# ---------------------------------------------------------------------------

def embed_texts(texts: list) -> np.ndarray:
    """
    Ubah list teks jadi matrix embedding menggunakan Voyage AI
    (via Anthropic API).

    Voyage adalah embedding model khusus — berbeda dari Claude yang
    generative. Embedding model dioptimasi untuk retrieval: menghasilkan
    vektor yang menempatkan teks bermakna mirip berdekatan dalam ruang.

    Output: numpy array shape [len(texts), embedding_dim]
    embedding_dim = 1024 untuk voyage-3
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    embeddings = [e.embedding for e in response.data]
    return np.array(embeddings, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Hitung cosine similarity antara satu query vector dan
    semua document vectors sekaligus.

    cos(q, d) = (q · d) / (|q| × |d|)

    Output: array skor similarity [num_docs], range -1 sampai 1.
    Makin tinggi = makin mirip.
    """
    # Normalisasi supaya dot product = cosine similarity
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms  = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


# ---------------------------------------------------------------------------
# 4. VECTOR DATABASE (sederhana — in-memory)
# ---------------------------------------------------------------------------

class SimpleVectorDB:
    """
    Vector database minimal: simpan chunks + embedding dalam memori.

    Di produksi pakai: FAISS (Meta), Chroma, Pinecone, Weaviate, Qdrant.
    Perbedaan utama: produksi mendukung persist ke disk, update inkremental,
    jutaan vektor, approximate nearest neighbor (ANN) yang jauh lebih cepat
    dari brute-force search yang kita pakai di sini.
    """

    def __init__(self):
        self.chunks    = []
        self.embeddings = None  # numpy array [n_chunks, dim]

    def add(self, chunks: list, embeddings: np.ndarray):
        self.chunks     = chunks
        self.embeddings = embeddings
        print(f"  Vector DB: {len(chunks)} chunks, dim={embeddings.shape[1]}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list:
        """Cari top_k chunk paling relevan dengan query."""
        scores   = cosine_similarity(query_embedding, self.embeddings)
        top_idxs = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idxs:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
            })
        return results


# ---------------------------------------------------------------------------
# 5. RAG PIPELINE
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, vector_db: SimpleVectorDB):
        self.db = vector_db

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Embed query lalu cari chunk paling relevan."""
        query_embedding = embed_texts([query])[0]
        return self.db.search(query_embedding, top_k=top_k)

    def build_prompt(self, query: str, retrieved: list) -> str:
        """Gabungkan retrieved chunks + query jadi prompt."""
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            context_parts.append(
                f"[Sumber {i}: {r['chunk']['title']} "
                f"(relevansi: {r['score']:.3f})]\n"
                f"{r['chunk']['text']}"
            )
        context = "\n\n".join(context_parts)

        return f"""Kamu adalah asisten asuransi yang membantu menjawab pertanyaan
berdasarkan dokumen kebijakan yang diberikan.

Jawab pertanyaan HANYA berdasarkan konteks di bawah.
Jika informasi tidak ada dalam konteks, katakan "Informasi tidak tersedia
dalam dokumen yang saya miliki."

KONTEKS:
{context}

PERTANYAAN:
{query}

JAWABAN:"""

    def answer(self, query: str, top_k: int = 3) -> dict:
        """End-to-end: query → retrieve → generate → answer."""
        # Retrieve
        retrieved = self.retrieve(query, top_k=top_k)

        # Build prompt
        prompt = self.build_prompt(query, retrieved)

        # Generate
        response = client.messages.create(
            model=GEN_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "query": query,
            "answer": response.content[0].text.strip(),
            "sources": [
                {
                    "title": r["chunk"]["title"],
                    "score": r["score"],
                    "text_preview": r["chunk"]["text"][:100] + "...",
                }
                for r in retrieved
            ],
        }


# ---------------------------------------------------------------------------
# 6. EVALUASI RAG — faithfulness check
# ---------------------------------------------------------------------------

def evaluate_faithfulness(answer: str, retrieved_chunks: list) -> dict:
    """
    Evaluasi sederhana: apakah jawaban bisa didukung oleh konteks?
    Di produksi pakai RAGAS framework untuk evaluasi lebih komprehensif.

    Metrik RAGAS:
    - faithfulness: apakah klaim dalam jawaban ada di konteks?
    - answer_relevancy: apakah jawaban relevan dengan pertanyaan?
    - context_recall: apakah chunk yang diambil mencakup info yang dibutuhkan?
    - context_precision: apakah chunk yang diambil memang relevan semua?
    """
    context_combined = " ".join(r["chunk"]["text"] for r in retrieved_chunks)

    eval_prompt = f"""Periksa apakah jawaban berikut sepenuhnya didukung oleh konteks.

Konteks: {context_combined[:600]}

Jawaban: {answer}

Jawab HANYA dengan JSON:
{{"faithful": true/false, "confidence": 0.0-1.0, "reason": "penjelasan singkat"}}"""

    response = client.messages.create(
        model=GEN_MODEL,
        max_tokens=200,
        messages=[
            {"role": "user", "content": eval_prompt},
            {"role": "assistant", "content": "{"},  # prefill JSON
        ],
    )

    try:
        return json.loads("{" + response.content[0].text)
    except Exception:
        return {"faithful": None, "confidence": None, "reason": "parse error"}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RAG dari Scratch — Dokumen Kebijakan Asuransi")
    print("=" * 60)

    # INDEXING PHASE
    print("\n[1/3] Chunking dokumen...")
    chunks = chunk_documents(DOCUMENTS, chunk_size=400, overlap=60)
    print(f"  {len(DOCUMENTS)} dokumen → {len(chunks)} chunks")

    print("\n[2/3] Embedding semua chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    db = SimpleVectorDB()
    db.add(chunks, embeddings)

    rag = RAGPipeline(db)

    # QUERYING PHASE
    test_queries = [
        "Apa saja dokumen yang dibutuhkan untuk mengajukan klaim?",
        "Apakah klaim akan dibayar jika kematian akibat olahraga ekstrem?",
        "Berapa lama proses investigasi klaim berlangsung?",
        "Bagaimana cara memulihkan polis yang sudah lapse?",
        "Apakah ada sistem deteksi fraud?",
    ]

    print("\n[3/3] Menjawab pertanyaan...\n")
    for i, query in enumerate(test_queries, 1):
        print(f"{'─'*60}")
        print(f"Q{i}: {query}")

        result = rag.answer(query, top_k=3)
        print(f"\nJawaban:\n{result['answer']}")

        print(f"\nSumber yang dipakai:")
        for src in result["sources"]:
            print(f"  [{src['score']:.3f}] {src['title']}")
            print(f"          '{src['text_preview']}'")

        # Evaluasi faithfulness
        retrieved = rag.retrieve(query, top_k=3)
        faith = evaluate_faithfulness(result["answer"], retrieved)
        print(f"\nFaithfulness: {faith.get('faithful')} "
              f"(confidence={faith.get('confidence')}) "
              f"— {faith.get('reason', '')[:60]}")

    print(f"\n{'='*60}")
    print("Selesai! Coba tambah dokumen baru dan lihat apakah RAG")
    print("bisa menjawab pertanyaan tentang dokumen tersebut.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
