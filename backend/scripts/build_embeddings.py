"""
Build FAISS vectorstore from chatbot documents.

Usage:
    cd backend
    python scripts/build_embeddings.py [--force]

This script loads documents from backend/app/documents/ (PDFs, .md, .txt),
splits them into chunks, and builds a FAISS index for the RAG chatbot.

Run this once before using the chatbot, and again whenever documents change.
"""

import argparse
import sys
import os
import shutil

# backend/ is the project root for this script
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from app.chatbot.loader import load_documents, text_split, get_embeddings
from app.chatbot.vectorstore import build_faiss

# Project root is one level above backend/
PROJECT_ROOT = os.path.dirname(BACKEND_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vectorstore for chatbot.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing FAISS index and rebuild from scratch.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Build FAISS Vectorstore for Non-Invasive Chatbot")
    print("=" * 60)
    print()

    print("Loading documents from backend/app/documents/ ...")
    docs = load_documents(PROJECT_ROOT)
    print(f"   -> {len(docs)} documents loaded")

    if not docs:
        print("X No documents found! Check backend/app/documents/ directory.")
        sys.exit(1)

    print()
    print("Splitting into chunks...")
    chunks = text_split(docs)
    print(f"   -> {len(chunks)} chunks created")

    from app.chatbot.vectorstore import VECTORSTORE_PATH
    if args.force and os.path.exists(VECTORSTORE_PATH):
        print(f"Removing existing FAISS index at {VECTORSTORE_PATH}...")
        shutil.rmtree(VECTORSTORE_PATH)

    print()
    print("Building FAISS index (embedding with HuggingFace all-MiniLM-L6-v2)...")
    embeddings = get_embeddings()
    build_faiss(chunks, embeddings)

    print()
    print("Done! Vectorstore saved to backend/vectorstore/faiss_index/")
    print("   You can now start the backend and use the /chat endpoint.")


if __name__ == "__main__":
    main()
