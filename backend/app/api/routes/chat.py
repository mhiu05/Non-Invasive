"""
chat.py — FastAPI endpoint for the RAG chatbot.

POST /chat  →  { question } → { answer, sources }
POST /chat/feedback  →  { feedback } → { status }
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatFeedbackRequest,
    ChatFeedbackResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chatbot"])


@router.post("", response_model=ChatResponse)
def chat_endpoint(body: ChatRequest):
    """Handle chat questions via RAG pipeline."""
    try:
        from app.chatbot.engine import ask

        result = ask(body.question)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            from_internal_docs=result.get("from_internal_docs", True),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot error: {str(e)}",
        )


@router.post("/feedback", response_model=ChatFeedbackResponse)
def chat_feedback(body: ChatFeedbackRequest):
    """Persist user feedback for future self-learning workflows."""
    try:
        from app.chatbot.feedback_store import save_feedback

        save_feedback(
            question=body.question,
            answer=body.answer,
            sources=body.sources,
            rating=body.rating,
            session_id=body.session_id,
        )
        return ChatFeedbackResponse()
    except Exception as e:
        logger.error(f"Chat feedback error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not save chat feedback: {str(e)}",
        )
