"""
Hari 4 - RAG dengan LangChain + ChromaDB
==========================================
Versi production-ready menggunakan LangChain sebagai orchestration
framework dan ChromaDB sebagai vector database yang persist ke disk.

Keunggulan vs 01_rag_from_scratch.py:
- ChromaDB: persist ke disk, tidak hilang saat restart
- LangChain: abstraksi yang mudah diganti (swap Chroma → Pinecone
  hanya dengan ganti satu baris)
- RecursiveCharacterTextSplitter: chunking yang lebih cerdas
- ConversationalRetrievalChain: support multi-turn conversation
- PDF loader: bisa langsung load dari file PDF

Setup:
  pip install langchain langchain-anthropic langchain-chroma
              chromadb pypdf anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."

Jalankan: python 02_rag_langchain.py
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma

# Untuk embedding kita pakai HuggingFace (gratis, offline)
# Alternatif: VoyageAIEmbeddings (lebih baik tapi berbayar)
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = "./chroma_db"   # Vector DB tersimpan di sini
COLLECTION_NAME    = "asuransi_docs"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB, offline


# ---------------------------------------------------------------------------
# DOKUMEN
# ---------------------------------------------------------------------------

RAW_DOCS = [
    ("Syarat Klaim Jiwa",
     "Untuk mengajukan klaim asuransi jiwa, tertanggung atau ahli waris "
     "harus menyiapkan: akta kematian asli, polis asuransi asli, KTP tertanggung, "
     "KTP pemohon, dan surat keterangan dokter. Pengajuan maksimal 60 hari sejak "
     "kejadian. Klaim melewati batas waktu akan ditolak kecuali force majeure."),

    ("Pengecualian Klaim",
     "Klaim tidak dibayarkan untuk: kematian bunuh diri dalam 2 tahun pertama, "
     "kematian akibat pelanggaran hukum, kematian akibat perang atau kerusuhan, "
     "olahraga ekstrem tidak terdaftar, dan kondisi pre-existing yang tidak "
     "diungkapkan saat pengajuan polis."),

    ("Premi Asuransi Jiwa Berjangka",
     "Premi dihitung dari usia, jenis kelamin, status merokok, dan uang "
     "pertanggungan. Untuk UP Rp 500 juta, usia 35 tahun non-perokok: "
     "Rp 350.000-500.000/bulan tergantung masa pertanggungan. "
     "Manfaat kematian dibayar sekaligus. Tidak ada nilai tunai."),

    ("Prosedur Investigasi Klaim",
     "Semua klaim di atas Rp 200 juta diinvestigasi oleh tim khusus. "
     "Proses: verifikasi dokumen, wawancara ahli waris, koordinasi dengan "
     "rumah sakit, pemeriksaan riwayat medis. Selesai dalam 30 hari kerja, "
     "bisa diperpanjang 15 hari dengan pemberitahuan tertulis."),

    ("Reinstatement Polis Lapse",
     "Polis lapse dapat dipulihkan dalam 2 tahun sejak lapse. "
     "Syarat: lunasi semua tunggakan premi + bunga, isi ulang formulir "
     "kesehatan, dan mungkin pemeriksaan medis baru. Setelah reinstatement, "
     "masa tunggu pengecualian bunuh diri mulai ulang dari awal."),

    ("Asuransi Kesehatan Rawat Inap",
     "Manfaat rawat inap: biaya kamar sesuai kelas (1, 2, VIP), biaya dokter, "
     "obat-obatan yang diresepkan, laboratorium, radiologi, dan tindakan medis. "
     "Batas tahunan Rp 100 juta hingga Rp 1 miliar. Lebih dari 3.000 RS rekanan "
     "dengan sistem cashless."),

    ("Fraud Detection",
     "Perusahaan menggunakan machine learning untuk mendeteksi pola klaim "
     "mencurigakan. Klaim terindikasi fraud diinvestigasi lebih lanjut. "
     "Jika terbukti: polis dibatalkan, klaim ditolak, pemohon dilaporkan "
     "ke pihak berwenang. Data fraud sharing antar perusahaan via database AAUI."),
]


# ---------------------------------------------------------------------------
# 1. LOAD DAN CHUNK DOKUMEN
# ---------------------------------------------------------------------------

def load_documents() -> list:
    """
    Konversi RAW_DOCS ke format LangChain Document.
    Di produksi bisa pakai:
      PyPDFLoader("dokumen.pdf")
      DirectoryLoader("./docs/", glob="*.pdf")
      WebBaseLoader("https://example.com/policy")
    """
    docs = []
    for title, content in RAW_DOCS:
        docs.append(Document(
            page_content=content,
            metadata={"title": title, "source": "kebijakan_asuransi"},
        ))
    return docs


def chunk_documents(docs: list) -> list:
    """
    RecursiveCharacterTextSplitter: coba potong di paragraf dulu,
    kalau masih terlalu panjang potong di kalimat, lalu di kata.
    Lebih natural dari fixed-size chunking biasa.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(docs)} dokumen → {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# 2. BUAT ATAU LOAD VECTOR DB
# ---------------------------------------------------------------------------

def get_vector_db(chunks: list = None, force_rebuild: bool = False):
    """
    Load ChromaDB dari disk jika ada, atau buat baru.
    ChromaDB persist otomatis — tidak perlu rebuild setiap run.
    """
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(CHROMA_PERSIST_DIR) and not force_rebuild:
        print(f"  Memuat vector DB dari {CHROMA_PERSIST_DIR}...")
        db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        print(f"  {db._collection.count()} chunks ter-index")
    else:
        print(f"  Membangun vector DB baru...")
        if chunks is None:
            raise ValueError("chunks diperlukan untuk membangun DB baru")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        print(f"  {db._collection.count()} chunks tersimpan ke disk")

    return db


# ---------------------------------------------------------------------------
# 3. RAG CHAIN
# ---------------------------------------------------------------------------

def build_rag_chain(db):
    """
    Bangun RAG chain dengan LangChain LCEL (LangChain Expression Language).

    Alur:
    query → retriever (ambil top-3 chunk) → prompt builder → LLM → output
    """
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        max_tokens=512,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template("""
Kamu adalah asisten asuransi yang menjawab berdasarkan dokumen kebijakan.
Jawab HANYA berdasarkan konteks. Jika tidak ada di konteks, katakan
"Informasi tidak tersedia dalam dokumen."

Konteks:
{context}

Pertanyaan: {question}

Jawaban:""")

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('title', 'Dokumen')}]\n{doc.page_content}"
            for doc in docs
        )

    # LCEL chain: query → retrieve + format → prompt → LLM → parse
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ---------------------------------------------------------------------------
# 4. CHUNKING STRATEGY COMPARISON
# ---------------------------------------------------------------------------

def compare_chunking_strategies(text: str):
    """
    Bandingkan tiga strategi chunking pada teks yang sama.
    Berguna untuk memilih strategi yang tepat untuk dokumen kamu.
    """
    print("\n" + "─" * 55)
    print("Perbandingan chunking strategy")
    print("─" * 55)

    strategies = {
        "Fixed-size (500 char)": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0, separators=[" "]
        ),
        "Semantic (paragraph)": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", ". "]
        ),
        "Small chunks + overlap": RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        ),
    }

    doc = Document(page_content=text)
    for name, splitter in strategies.items():
        chunks = splitter.split_documents([doc])
        print(f"\n{name}:")
        print(f"  {len(chunks)} chunks")
        for i, c in enumerate(chunks[:2]):
            print(f"  Chunk {i+1} ({len(c.page_content)} char): "
                  f"'{c.page_content[:60]}...'")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("RAG dengan LangChain + ChromaDB")
    print("=" * 55)

    # Indexing
    print("\n[1/3] Load dan chunk dokumen...")
    docs   = load_documents()
    chunks = chunk_documents(docs)

    print("\n[2/3] Membangun vector DB...")
    db = get_vector_db(chunks, force_rebuild=True)

    print("\n[3/3] Menjawab pertanyaan...\n")
    chain, retriever = build_rag_chain(db)

    test_queries = [
        "Apa saja dokumen yang dibutuhkan untuk klaim jiwa?",
        "Apakah kematian karena olahraga ekstrem ditanggung?",
        "Berapa lama investigasi klaim berlangsung?",
        "Bagaimana cara pulihkan polis yang lapse?",
        "Apakah ada manfaat rawat jalan?",  # tidak ada di dokumen
    ]

    for query in test_queries:
        print(f"{'─'*55}")
        print(f"Q: {query}")

        # Jawaban dari chain
        answer = chain.invoke(query)
        print(f"A: {answer}")

        # Lihat chunk yang diambil
        retrieved = retriever.invoke(query)
        print(f"\nSumber ({len(retrieved)} chunk):")
        for r in retrieved:
            print(f"  • {r.metadata.get('title')} — '{r.page_content[:60]}...'")

    # Demo chunking comparison
    sample_text = (
        "Untuk mengajukan klaim, siapkan dokumen berikut.\n\n"
        "Pertama, akta kematian asli dari kelurahan.\n"
        "Kedua, polis asuransi yang masih aktif.\n\n"
        "Pengajuan dilakukan dalam 60 hari sejak kejadian."
    )
    compare_chunking_strategies(sample_text)

    print(f"\n{'='*55}")
    print(f"Vector DB tersimpan di: {CHROMA_PERSIST_DIR}/")
    print("Run ulang — DB tidak perlu rebuild, langsung query.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
