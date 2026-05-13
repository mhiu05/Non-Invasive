from fastapi import APIRouter
from app.api.routes import health, video
from app.api.websocket import stream

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(video.router, tags=["video"])
api_router.include_router(stream.router, tags=["websocket"])
