"""
loader.py — Load & split Markdown/code documents for RAG chatbot.

Replaces PyPDFLoader from medical-chatbot with TextLoader for .md and .py files.
"""

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import os


def load_documents(base_dir: str) -> List[Document]:
    """Load all .md, .txt, and .pdf files from documents directory."""
    docs = []
    
    doc_dir = os.path.join(base_dir, "backend", "app", "documents")
    
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
                docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: Could not load {config['glob']} from {doc_dir}: {e}")

    return docs


def text_split(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def get_embeddings():
    """Get HuggingFace embeddings model (same as medical-chatbot)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
