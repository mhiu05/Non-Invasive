"""
auto_update.py — Rebuild and refresh the chatbot vectorstore.

Provides a centralized rebuild function so future automated index refresh or
file watcher workflows can reuse the same logic.
"""

import os
from typing import Optional
from .loader import load_documents, text_split, get_embeddings
from .vectorstore import rebuild_faiss


def rebuild_index(base_dir: Optional[str] = None, force: bool = True) -> None:
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    documents = load_documents(base_dir)
    if not documents:
        raise ValueError("No chatbot documents found in backend/app/documents/")

    chunks = text_split(documents)
    embeddings = get_embeddings()
    rebuild_faiss(chunks, embeddings, force=force)
