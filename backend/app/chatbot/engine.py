"""
engine.py — RAG chain engine for the chatbot.

Lazy-loads the RAG chain on first request to avoid slowing down server startup.
Uses Google Gemini as the LLM backend.
"""

import logging
import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from .loader import get_embeddings
from .vectorstore import load_faiss

logger = logging.getLogger(__name__)

# System prompt: rPPG health context + medical disclaimer
SYSTEM_PROMPT = (
    "Bạn là trợ lý AI cho ứng dụng đo sức khỏe không xâm lấn (rPPG). "
    "Hỗ trợ người dùng hiểu kết quả đo (nhịp tim, HRV, blink rate, SNR) "
    "và developer tra cứu kiến trúc hệ thống. "
    "Chỉ dùng thông tin từ tài liệu nội bộ được cung cấp. "
    "Nếu câu hỏi về sức khỏe, hãy thêm cảnh báo: thông tin chỉ mang tính tham khảo, không thay thế bác sĩ. "
    "Nếu không có thông tin, hãy nói rõ là không tìm thấy trong tài liệu nội bộ. "
    "Trả lời ngắn gọn, tối đa 4 câu.\n\n"
    "{context}"
)

_rag_chain = None


def get_rag_chain():
    """Get or initialize the RAG chain (lazy-loaded singleton)."""
    global _rag_chain
    if _rag_chain is None:
        logger.info("🤖 Initializing RAG chain (first request)...")
        embeddings = get_embeddings()
        vectorstore = load_faiss(embeddings)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        from app.core.config import settings
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=settings.chatbot_model,
            google_api_key=settings.gemini_api_key,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm, prompt)
        _rag_chain = create_retrieval_chain(retriever, qa_chain)
        logger.info("✅ RAG chain initialized successfully")
    return _rag_chain


def ask(question: str) -> dict:
    """Ask a question to the RAG chatbot and return answer + sources."""
    chain = get_rag_chain()
    result = chain.invoke({"input": question})
    sources = list({doc.metadata.get("source", "") for doc in result["context"]})
    return {"answer": result["answer"], "sources": sources}
