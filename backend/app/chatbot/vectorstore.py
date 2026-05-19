"""
vectorstore.py — Build & load FAISS vector index.

Uses FAISS (local) instead of Pinecone (cloud) — no API key needed.
Index is stored in backend/vectorstore/faiss_index/.
"""

from langchain_community.vectorstores import FAISS
import os

# Resolve path relative to backend/ root
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_default_path = os.path.join(_BACKEND_ROOT, "vectorstore", "faiss_index")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", _default_path)


def build_faiss(chunks, embeddings):
    """Build and save FAISS index from document chunks."""
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(VECTORSTORE_PATH) or ".", exist_ok=True)
    db.save_local(VECTORSTORE_PATH)
    print(f"FAISS index saved to {VECTORSTORE_PATH}")
    return db


def load_faiss(embeddings):
    """Load pre-built FAISS index."""
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{VECTORSTORE_PATH}'. "
            "Run 'python scripts/build_embeddings.py' from the backend/ directory first."
        )
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
