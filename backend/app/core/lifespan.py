"""
lifespan.py — Application lifecycle management and global state.

Responsibilities:
- Manage the startup and shutdown events of the FastAPI application.
- Initialize and hold global, read-only instances (like AI models) in memory
  so they can be shared efficiently across all API requests.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.services.face_detector import FaceDetector
from app.services.history_store import init_history_db
from app.services.rppg_engine import RPPGEngine

logger = logging.getLogger(__name__)

# Globals — loaded once at startup, shared read-only across requests
engine: RPPGEngine | None = None
face_detector: FaceDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, face_detector

    logger.info("Loading rPPG model from: %s", settings.model_path)
    engine = RPPGEngine(settings.model_path, settings.model_config_path, settings.device)

    logger.info("Initializing MediaPipe face detector")
    face_detector = FaceDetector()

    logger.info("Initializing history DB")
    init_history_db()

    logger.info("Startup complete.")
    yield

    logger.info("Shutdown.")
