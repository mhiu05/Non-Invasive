"""
engine.py — RAG chain engine for the chatbot.

Lazy-loads the RAG chain on first request to avoid slowing down server startup.
Uses Google Gemini as the LLM backend.

Fallback strategy:
  1. Try RAG (internal docs) first.
  2. If no relevant docs found → call Gemini directly with general knowledge.
  3. Response includes `from_internal_docs` so the frontend can show the source.
"""

import os
os.environ["HF_HOME"] = "/tmp/huggingface"

import logging
from typing import Any, List

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from .loader import get_embeddings
from .vectorstore import load_faiss, load_bm25

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI cho ứng dụng đo sức khỏe không xâm lấn (rPPG). "
    "Hỗ trợ người dùng hiểu kết quả đo (nhịp tim, HRV, SNR) "
    "và developer tra cứu kiến trúc hệ thống. "
    "Chỉ dùng thông tin từ tài liệu nội bộ được cung cấp. "
    "Nếu câu hỏi về sức khỏe, hãy thêm cảnh báo: thông tin chỉ mang tính tham khảo, không thay thế bác sĩ. "
    "Nếu không có thông tin, hãy nói rõ là không tìm thấy trong tài liệu nội bộ và để gợi ý cách tìm thêm. "
    "{context}"
)

_rag_chain = None
_rag_llm = None


def _extract_sources(result: Any) -> List[str]:
    """Extract source metadata from the RAG chain result."""
    sources = []
    candidates = []

    if isinstance(result, dict):
        if "source_documents" in result:
            candidates = result["source_documents"]
        elif "context" in result:
            candidates = result["context"]
        elif "sources" in result:
            candidates = result["sources"]

    for item in candidates or []:
        metadata = None
        if isinstance(item, dict):
            metadata = item.get("metadata", {})
        else:
            metadata = getattr(item, "metadata", {}) if item is not None else {}

        if metadata:
            source = metadata.get("source") or metadata.get("source_file") or metadata.get("path")
            if source:
                sources.append(str(source))

    normalized = []
    for source in sources:
        if source:
            normalized.append(source.split("|")[-1].strip())
    return sorted(set(normalized))


def reset_rag_chain() -> None:
    """Reset the cached RAG chain so a rebuilt index will be reloaded."""
    global _rag_chain
    _rag_chain = None


def _build_gemini_llm(model: str = None):
    from app.core.config import settings
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm_model = model or settings.chatbot_model
    return ChatGoogleGenerativeAI(
        model=llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        phrase in text
        for phrase in [
            "resource_exhausted",
            "quota exceeded",
            "429",
            "free_tier",
            "rate-limit",
            "rate limit",
        ]
    )


def _get_fallback_model():
    from app.core.config import settings
    return settings.chatbot_model_fallback or None


def get_rag_chain():
    """Get or initialize the RAG chain (lazy-loaded singleton)."""
    global _rag_chain, _rag_llm
    if _rag_chain is None:
        logger.info("🤖 Initializing RAG chain (first request)...")
        embeddings = get_embeddings()
        vectorstore = load_faiss(embeddings)
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )

        bm25_retriever = load_bm25()
        if bm25_retriever:
            bm25_retriever.k = 10
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.3, 0.7]
            )
        else:
            retriever = faiss_retriever

        _rag_llm = _build_gemini_llm()
        
        # Query Rewriting
        retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=_rag_llm
        )
        
        # Disabled Re-ranking to save RAM (avoids OOM on Render Free tier / Vercel)
        # and to speed up response time.

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(_rag_llm, prompt)
        _rag_chain = create_retrieval_chain(retriever, qa_chain)
        logger.info("✅ RAG chain initialized successfully")
    return _rag_chain


def _build_llm_answer(question: str, llm) -> dict:
    """Ask the LLM directly to answer when internal docs lack coverage.

    This returns a short answer and an explicit source marker indicating
    the response came from the LLM's general knowledge (not local docs).
    For health-related questions the prompt includes a disclaimer.
    """
    prompt_text = (
        "Tài liệu nội bộ không có thông tin đủ để trả lời câu hỏi sau. "
        "Sử dụng kiến thức tổng quát của bạn để trả lời ngắn gọn, chính xác, "
        "và luôn kèm cảnh báo y tế khi câu hỏi liên quan sức khỏe. "
        "Nêu rõ khi bạn đang trả lời dựa trên kiến thức chung chứ không phải từ tài liệu nội bộ.\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời bằng tiếng Việt, tối đa 4 câu. Nếu không chắc chắn, ghi rõ rằng bạn không chắc."
    )

    try:
        # ChatGoogleGenerativeAI.invoke() expects a string or list of
        # messages and returns an AIMessage object with a .content attribute.
        ai_message = llm.invoke(prompt_text)
        answer_text = ai_message.content.strip() if ai_message.content else ""
    except Exception as exc:
        model_name = getattr(llm, "model", "gemini")
        logger.error("LLM invoke failed for %s: %s", model_name, exc, exc_info=True)
        if _is_quota_error(exc):
            fallback_model = _get_fallback_model()
            if fallback_model and fallback_model != model_name:
                fallback_llm = _build_gemini_llm(fallback_model)
                return _build_llm_answer(question, fallback_llm)
            answer_text = (
                f"Gemini ({model_name}) hiện không thể trả lời do hạn mức hoặc billing. "
                "Vui lòng kiểm tra API key, plan, hoặc đổi model trong .env."
            )
        else:
            answer_text = (
                f"Tôi không thể truy cập LLM {model_name} để trả lời lúc này. "
                "Bạn có thể thử lại sau hoặc tìm nguồn y tế uy tín."
            )

    source_model = getattr(llm, "model", "gemini")
    sources = [f"Gemini ({source_model})"]
    return {
        "answer": answer_text,
        "sources": sources,
        "from_internal_docs": False,
    }


def ask(question: str, refresh: bool = False) -> dict:
    """Ask a question to the RAG chatbot and return answer + sources.

    Returns dict with keys: answer, sources, from_internal_docs.
    """
    global _rag_llm
    if refresh:
        reset_rag_chain()

    chain = get_rag_chain()
    try:
        result = chain.invoke({"input": question})
    except Exception as exc:
        if _is_quota_error(exc):
            logger.warning("Gemini quota exceeded on primary model; trying fallback.", exc_info=exc)
            fallback_model = _get_fallback_model()
            if fallback_model:
                fallback_llm = _build_gemini_llm(fallback_model)
                return _build_llm_answer(question, fallback_llm)
        raise

    answer = result.get("answer") or result.get("output_text") or ""
    sources = _extract_sources(result)

    answer_text = answer.strip()
    answer_lower = answer_text.lower()
    missing_answer = (
        "không tìm thấy" in answer_lower
        or ("không có" in answer_lower and "tài liệu" in answer_lower)
        or "không có thông tin" in answer_lower
        or "không tìm được" in answer_lower
    )

    if not sources or missing_answer:
        # Internal docs don't have the answer → fallback to Gemini general knowledge
        logger.info("📡 Tài liệu nội bộ không đủ thông tin, chuyển sang Gemini trực tiếp cho câu hỏi: %s", question)
        if _rag_llm is None:
            _rag_llm = _build_gemini_llm()
        return _build_llm_answer(question, _rag_llm)

    return {
        "answer": answer_text,
        "sources": sources,
        "from_internal_docs": True,
    }
