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
    buffer_size: int = 180  # số frames tích lũy trước khi tính HR
    hr_low_hz: float = 0.75  # 45 BPM
    hr_high_hz: float = 2.5  # 150 BPM

    # Upload
    max_upload_mb: int = 100

    # Chatbot
    gemini_api_key: str = ""
    # Primary Gemini model used for internal fallback when documents are missing
    chatbot_model: str = "gemini-flash-latest"
    # Fallback model to try if the primary model hits quota limits or is unavailable
    chatbot_model_fallback: str = "gemini-pro-latest"
    # NOTE: Google CSE removed — web fallback disabled

    # Redis & Celery
    redis_url: str = "redis://redis:6379/0"

    # Object Storage (Supabase/S3)
    storage_endpoint: str = ""
    storage_key: str = ""
    storage_secret: str = ""
    storage_bucket: str = "videos"
settings = Settings()
