"""
config.py — Central configuration settings for the backend.

Responsibilities:
- Load environment variables from the .env file.
- Define application-wide configuration like model paths, database URLs, and API keys.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

_backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(_backend_dir, ".env"), env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Non-Invasive Health API"
    debug: bool = False

    # Model
    model_path: str = "weights/UBFC-rPPG_FactorizePhys_FSAM_Res.onnx"
    model_config_path: str = "weights/model_config.json"
    device: str = "cpu"  # "cpu" hoặc "cuda:0"

    # Signal processing
    fps: int = 30

    # Upload
    max_upload_mb: int = 100

    # Chatbot
    gemini_api_key: str = ""
    chatbot_model: str = "gemini-2.5-flash"
    chatbot_model_fallback: str = "gemini-2.5-pro"

    # Redis & Celery
    redis_url: str = "redis://redis:6379/0"

    # Object Storage (Supabase)
    storage_endpoint: str = ""
    storage_key: str = ""
    storage_secret: str = ""
    storage_bucket: str = "videos"
settings = Settings()
