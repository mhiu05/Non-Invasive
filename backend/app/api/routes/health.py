from fastapi import APIRouter
from app.schemas.vitals import HealthResponse
from app.core.config import settings
import app.core.lifespan as state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=state.engine is not None,
        device=settings.device,
    )
