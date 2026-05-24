"""
loader.py — Load & split Markdown/Text/PDF documents for RAG chatbot.

Loads documents from backend/app/documents/ and preserves source metadata so
responses can include transparent document references.
"""

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os


def _document_dir(base_dir: Optional[str] = None) -> str:
    if base_dir:
        base_dir = os.path.abspath(base_dir)
        if os.path.basename(base_dir) == "backend":
            return os.path.join(base_dir, "app", "documents")
        return os.path.join(base_dir, "backend", "app", "documents")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documents"))


def _normalize_source_metadata(doc: Document) -> None:
    metadata = dict(doc.metadata or {})
    source = (
        metadata.get("source")
        or metadata.get("path")
        or metadata.get("file_path")
        or metadata.get("source_file")
    )
    metadata["source"] = os.path.basename(str(source)) if source else "unknown"
    doc.metadata = metadata


def load_documents(base_dir: Optional[str] = None) -> List[Document]:
    """Load all .md, .txt, and .pdf files from documents directory."""
    docs = []
    doc_dir = _document_dir(base_dir)
    
    configs = [
        {"glob": "**/*.md", "loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}},
        {"glob": "**/*.txt", "loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}},
        {"glob": "**/*.pdf", "loader_cls": PyPDFLoader, "loader_kwargs": {}},
    ]

    if os.path.exists(doc_dir):
        for config in configs:
            try:
                loader = DirectoryLoader(
                    doc_dir,
                    glob=config["glob"],
                    loader_cls=config["loader_cls"],
                    loader_kwargs=config["loader_kwargs"],
                )
                documents = loader.load()
                for doc in documents:
                    _normalize_source_metadata(doc)
                docs.extend(documents)
            except Exception as e:
                print(f"Warning: Could not load {config['glob']} from {doc_dir}: {e}")

    return docs


def text_split(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for embedding and preserve source metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    for chunk_index, doc in enumerate(chunks, start=1):
        _normalize_source_metadata(doc)
        metadata = dict(doc.metadata or {})
        metadata.setdefault("chunk_id", str(chunk_index))
        doc.metadata = metadata
    return chunks


def get_embeddings():
    """Get HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
