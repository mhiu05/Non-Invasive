from fastapi import APIRouter
from app.api.routes import system, history, video, chat, auth
from app.api.websocket import stream

api_router = APIRouter()

api_router.include_router(system.router, tags=["system"])
api_router.include_router(history.router, tags=["history"])
api_router.include_router(video.router, tags=["video"])
api_router.include_router(stream.router, tags=["websocket"])
api_router.include_router(chat.router, tags=["chatbot"])
api_router.include_router(auth.router, tags=["auth"])
