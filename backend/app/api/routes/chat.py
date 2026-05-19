"""
chat.py — FastAPI endpoint for the RAG chatbot.

POST /chat  →  { question } → { answer, sources }
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


@router.post("", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    """Handle chat questions via RAG pipeline."""
    try:
        from app.chatbot.engine import ask

        result = ask(body.question)
        return ChatResponse(answer=result["answer"], sources=result["sources"])
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
