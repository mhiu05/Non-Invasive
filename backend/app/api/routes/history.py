from fastapi import APIRouter, HTTPException, Query

from app.schemas.history import HistoryDetailResponse, HistorySummary
from app.services.history_store import get_history_by_id, get_history_list

router = APIRouter()


@router.get("/history", response_model=list[HistorySummary])
def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    type: str | None = Query(None),
    start_at: str | None = Query(None),
    end_at: str | None = Query(None),
):
    return get_history_list(
        limit=limit,
        offset=offset,
        history_type=type,
        start_at=start_at,
        end_at=end_at,
    )


@router.get("/history/{history_id}", response_model=HistoryDetailResponse)
def get_history_detail(history_id: str):
    record = get_history_by_id(history_id)
    if record is None:
        raise HTTPException(404, "History record not found")
    return record
