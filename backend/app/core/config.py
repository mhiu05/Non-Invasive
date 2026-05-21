from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Non-Invasive Health API"
    debug: bool = False

    # Model
    model_path: str = "weights/PURE_DeepPhys.onnx"
    model_config_path: str = "weights/model_config.json"
    device: str = "cpu"  # "cpu" hoặc "cuda:0"

    # Signal processing
    fps: int = 30
    buffer_size: int = 180  # số frames tích lũy trước khi tính HR
    hr_low_hz: float = 0.75  # 45 BPM
    hr_high_hz: float = 2.5  # 150 BPM
    blink_low_hz: float = 0.1
    blink_high_hz: float = 0.9

    # Upload
    max_upload_mb: int = 100

    # Chatbot
    gemini_api_key: str = ""
    # Primary Gemini model used for internal fallback when documents are missing
    chatbot_model: str = "gemini-2.0-flash-lite"
    # Fallback model to try if the primary model hits quota limits or is unavailable
    chatbot_model_fallback: str = "gemini-2.0-flash-lite"
    # NOTE: Google CSE removed — web fallback disabled


settings = Settings()
