"""Build the ChromaDB vector store from data/iso27001.txt.

Run from the project root:
    python -m backend.rag_setup
"""

import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ---------- Paths & constants (shared with backend/main.py) ----------

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_PATH = BASE_DIR / "data" / "iso27001.txt"
DB_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "iso27001_controls"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 600


def load_document() -> str:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"iso27001.txt not found at {DATA_PATH}")
    return DATA_PATH.read_text(encoding="utf-8")


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """
    Split the document into chunks, one per section (sections are separated
    by blank lines and start with a title line such as
    "A.7.7 Clear desk and clear screen (Physical controls)").

    A section longer than max_chars is split on sentence boundaries, and the
    title is repeated on every piece so each chunk still says which control
    it belongs to.
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks: list[dict] = []

    for block in blocks:
        title, _, body = block.partition("\n")
        title = title.strip()

        if len(block) <= max_chars:
            chunks.append({"title": title, "text": block})
            continue

        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        current = ""
        for sentence in sentences:
            if current and len(title) + len(current) + len(sentence) + 2 > max_chars:
                chunks.append({"title": title, "text": f"{title}\n{current.strip()}"})
                current = ""
            current += sentence + " "
        if current.strip():
            chunks.append({"title": title, "text": f"{title}\n{current.strip()}"})

    return chunks


def control_id(title: str) -> str:
    """Return the Annex A control number from a title, e.g. 'A.7.7', or ''."""
    match = re.match(r"(A\.\d+\.\d+)", title)
    return match.group(1) if match else ""


def build_vector_store() -> None:
    print(f"🔹 Loading document from {DATA_PATH} ...")
    text = load_document()

    print("🔹 Chunking document ...")
    chunks = chunk_text(text)
    print(f"   → {len(chunks)} chunks created")

    print("🔹 Loading embedding model ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("🔹 Creating ChromaDB client ...")
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Rebuild from scratch every run so the store always matches the text file.
    existing = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print("   → Deleting existing collection ...")
        client.delete_collection(COLLECTION_NAME)

    # Cosine distance suits sentence-transformer embeddings.
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    print("🔹 Embedding and inserting chunks ...")
    documents = [c["text"] for c in chunks]
    embeddings = model.encode(documents).tolist()

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {"source": "iso27001", "title": c["title"], "control": control_id(c["title"])}
            for c in chunks
        ],
    )

    print("✅ Done. Vector store created at:", DB_PATH)
    print("   Collection name:", COLLECTION_NAME)
    print("   Total documents in collection:", collection.count())


def quick_test(query: str = "What is clean desk policy?") -> None:
    """Retrieval smoke test, using the same embedding model as the API."""
    print("🔹 Running quick retrieval test ...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    results = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=3,
    )

    print(f"   Query: {query}")
    for i, (doc, dist) in enumerate(
        zip(results["documents"][0], results["distances"][0]), start=1
    ):
        print(f"\n--- Result {i} (cosine distance {dist:.3f}) ---")
        print(doc)


if __name__ == "__main__":
    build_vector_store()
    quick_test()
