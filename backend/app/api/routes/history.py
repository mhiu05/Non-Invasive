"""
history.py — FastAPI endpoints for user history.

GET /history  →  { limit, offset, type, ... } → [ HistorySummary ]
GET /history/{history_id}  →  { history_id } → { HistoryDetailResponse }
"""

from fastapi import APIRouter, HTTPException, Query, Depends

from app.schemas.history import HistoryDetailResponse, HistorySummary
from app.services.history_store import get_history_by_id, get_history_list
from app.core.security import get_current_user_required

router = APIRouter()


@router.get("/history", response_model=list[HistorySummary])
def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    type: str | None = Query(None),
    start_at: str | None = Query(None),
    end_at: str | None = Query(None),
    current_user: dict = Depends(get_current_user_required),
):
    return get_history_list(
        limit=limit,
        offset=offset,
        history_type=type,
        start_at=start_at,
        end_at=end_at,
        user_id=current_user["id"],
    )


@router.get("/history/{history_id}", response_model=HistoryDetailResponse)
def get_history_detail(history_id: str, current_user: dict = Depends(get_current_user_required)):
    record = get_history_by_id(history_id, user_id=current_user["id"])
    if record is None:
        raise HTTPException(404, "History record not found")
    return record
