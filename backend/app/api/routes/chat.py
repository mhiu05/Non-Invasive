"""
chat.py — FastAPI endpoint for the RAG chatbot.

POST /chat  →  { question } → { answer, sources }
POST /chat/feedback  →  { feedback } → { status }
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chatbot"])


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    from_internal_docs: bool = True


class ChatFeedbackRequest(BaseModel):
    question: str
    answer: str
    sources: List[str]
    rating: Optional[int] = None
    comment: Optional[str] = None
    session_id: Optional[str] = None


class ChatFeedbackResponse(BaseModel):
    status: str = "ok"
    saved: bool = True


@router.post("", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
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
async def chat_feedback(body: ChatFeedbackRequest):
    """Persist user feedback for future self-learning workflows."""
    try:
        from app.chatbot.feedback_store import save_feedback

        save_feedback(
            question=body.question,
            answer=body.answer,
            sources=body.sources,
            rating=body.rating,
            comment=body.comment,
            session_id=body.session_id,
        )
        return ChatFeedbackResponse()
    except Exception as e:
        logger.error(f"Chat feedback error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not save chat feedback: {str(e)}",
        )
