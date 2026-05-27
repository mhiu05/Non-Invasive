from pydantic import BaseModel
from typing import Optional, List

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
    session_id: Optional[str] = None

class ChatFeedbackResponse(BaseModel):
    status: str = "ok"
    saved: bool = True
