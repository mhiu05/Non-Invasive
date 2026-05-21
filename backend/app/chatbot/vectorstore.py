"""
vectorstore.py — Build, load, and update FAISS vector index.

Uses FAISS local storage at backend/vectorstore/faiss_index/. Supports
rebuilds and incremental updates for internal document sources.
"""

from langchain_community.vectorstores import FAISS
import os
import shutil

# Resolve path relative to backend/ root
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_default_path = os.path.join(_BACKEND_ROOT, "vectorstore", "faiss_index")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", _default_path)


def _ensure_index_dir():
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)


def index_exists() -> bool:
    return os.path.exists(VECTORSTORE_PATH) and bool(os.listdir(VECTORSTORE_PATH))


def build_faiss(chunks, embeddings):
    """Build and save FAISS index from document chunks."""
    db = FAISS.from_documents(chunks, embeddings)
    _ensure_index_dir()
    db.save_local(VECTORSTORE_PATH)
    print(f"FAISS index saved to {VECTORSTORE_PATH}")
    return db


def load_faiss(embeddings):
    """Load pre-built FAISS index."""
    if not index_exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{VECTORSTORE_PATH}'. "
            "Run 'python scripts/build_embeddings.py' from the backend/ directory first."
        )
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def rebuild_faiss(chunks, embeddings, force: bool = False):
    """Rebuild FAISS index from scratch."""
    if force and os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
    return build_faiss(chunks, embeddings)


def update_faiss(chunks, embeddings):
    """Incrementally update an existing FAISS index or build a new one."""
    if index_exists():
        db = load_faiss(embeddings)
        db.add_documents(chunks)
        db.save_local(VECTORSTORE_PATH)
        print(f"FAISS index updated with {len(chunks)} new chunks")
        return db
    return build_faiss(chunks, embeddings)
