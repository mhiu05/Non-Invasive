from fastapi import APIRouter
from app.api.routes import health, history, video, chat, auth
from app.api.websocket import stream

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(history.router, tags=["history"])
api_router.include_router(video.router, tags=["video"])
api_router.include_router(stream.router, tags=["websocket"])
api_router.include_router(chat.router, tags=["chatbot"])
