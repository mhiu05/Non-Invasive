"""
ingest.py — Ingest external text into the internal chatbot document corpus.

This module provides a lightweight helper for adding new knowledge sources to
backend/app/documents/ so they can be included in FAISS build and chatbot answers.
"""

import os
from pathlib import Path
from typing import Optional


def _documents_dir(base_dir: Optional[str] = None) -> Path:
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return Path(base_dir) / "backend" / "app" / "documents"


def ingest_text(name: str, text: str, base_dir: Optional[str] = None) -> Path:
    documents_dir = _documents_dir(base_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    path = documents_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def ingest_file(source_path: str, base_dir: Optional[str] = None) -> Path:
    documents_dir = _documents_dir(base_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path)
    target_path = documents_dir / source_path.name
    target_path.write_bytes(source_path.read_bytes())
    return target_path
