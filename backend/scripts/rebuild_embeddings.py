"""
Rebuild the chatbot FAISS index using the shared auto-update helper.

Usage:
    cd backend
    python scripts/rebuild_embeddings.py
"""

import sys
import os

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from app.chatbot.auto_update import rebuild_index


def main():
    print("Rebuilding chatbot FAISS index from backend/app/documents/")
    rebuild_index(base_dir=BACKEND_ROOT, force=True)
    print("Rebuild complete.")


if __name__ == "__main__":
    main()
