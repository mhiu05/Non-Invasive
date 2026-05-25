"""Basic API tests — chạy: pytest tests/"""

import importlib
import os
import sqlite3
import uuid
from datetime import datetime

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest

# Mock engine và face_detector trước khi import app
mock_engine = MagicMock()
mock_engine.img_size = 72
mock_engine.buffer_size = 180

with patch("app.core.lifespan.engine", mock_engine), \
     patch("app.core.lifespan.face_detector", MagicMock()):
    from app.main import app

client = TestClient(app)


def test_health_endpoint():
    with patch("app.core.lifespan.engine", mock_engine):
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_docs_available():
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_upload_no_file():
    resp = client.post("/video/upload")
    assert resp.status_code == 422  # validation error — thiếu file


def test_upload_video_async_creates_job(tmp_path, monkeypatch):
    # Use a temporary DB path for isolated test state
    db_path = tmp_path / "video_jobs.db"
    monkeypatch.setattr("app.api.routes.video.DB_PATH", str(db_path))
    import app.api.routes.video as video_module
    video_module.init_jobs_db()

    def fake_process_video_job(job_id, tmp_path_arg, filename, age):
        # Simulate background completion and cleanup
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            ("done", datetime.utcnow().isoformat(), job_id),
        )
        conn.commit()
        conn.close()
        try:
            os.unlink(tmp_path_arg)
        except OSError:
            pass

    monkeypatch.setattr("app.api.routes.video.process_video_job", fake_process_video_job)

    with patch("app.core.lifespan.engine", mock_engine), patch("app.core.lifespan.face_detector", MagicMock()):
        resp = client.post(
            "/video/upload-async",
            files={"file": ("test.mp4", b"fakevideo", "video/mp4")},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "pending"
    assert "job_id" in data

    job_resp = client.get(f"/video/jobs/{data['job_id']}")
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] in {"pending", "running", "done", "failed"}


def test_history_store_save_and_query(tmp_path, monkeypatch):
    history_db = tmp_path / "history.db"
    monkeypatch.setattr("app.services.history_store.DB_PATH", str(history_db))
    import app.services.history_store as history_store
    history_store.init_history_db()

    history_id = history_store.save_history_record({
        "type": "video",
        "filename": "test.mp4",
        "duration_sec": 1.5,
        "heart_rate": 72.0,

        "snr_db": 5.0,
        "age": 30,
        "age_group": ">= 8 tuổi",
        "bandpass_low_hz": 1.0,
        "bandpass_high_hz": 1.67,
        "hrv_ms": 10.0,
        "sdnn_ms": 20.0,
        "rmssd_ms": 30.0,
        "pnn50": 2.0,
        "peak_count": 10,
        "result": {"summary": "test history"},
    })

    record = history_store.get_history_by_id(history_id)
    assert record is not None
    assert record["filename"] == "test.mp4"
    assert record["type"] == "video"
    assert record["result"]["summary"] == "test history"

    records = history_store.get_history_list()
    assert len(records) == 1


def test_upload_video_saves_history(tmp_path, monkeypatch):
    history_db = tmp_path / "history.db"
    monkeypatch.setattr("app.services.history_store.DB_PATH", str(history_db))
    import app.services.history_store as history_store
    history_store.init_history_db()

    monkeypatch.setattr("app.api.routes.video._process_video", lambda path: ([0.1] * 30, 8.0, 60))

    with patch("app.core.lifespan.engine", mock_engine), patch("app.core.lifespan.face_detector", MagicMock()):
        resp = client.post(
            "/video/upload",
            files={"file": ("test.mp4", b"fakevideo", "video/mp4")},
            data={"age": "25"},
        )

    assert resp.status_code == 200
    items = history_store.get_history_list()
    assert len(items) == 1
    assert items[0]["filename"] == "test.mp4"
    assert items[0]["type"] == "video"


def test_history_endpoint_returns_saved_records(tmp_path, monkeypatch):
    history_db = tmp_path / "history.db"
    monkeypatch.setattr("app.services.history_store.DB_PATH", str(history_db))
    import app.services.history_store as history_store
    history_store.init_history_db()
    history_store.save_history_record({
        "type": "video",
        "filename": "history_test.mp4",
        "duration_sec": 2.5,
        "heart_rate": 75.0,

        "snr_db": 4.5,
        "age": 28,
        "age_group": ">= 8 tuổi",
        "bandpass_low_hz": 1.0,
        "bandpass_high_hz": 1.67,
        "hrv_ms": 15.0,
        "sdnn_ms": 25.0,
        "rmssd_ms": 35.0,
        "pnn50": 3.0,
        "peak_count": 8,
    })

    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.json()[0]["filename"] == "history_test.mp4"


def test_process_video_job_saves_history(tmp_path, monkeypatch):
    history_db = tmp_path / "history.db"
    video_db = tmp_path / "video_jobs.db"
    monkeypatch.setattr("app.services.history_store.DB_PATH", str(history_db))
    import app.services.history_store as history_store
    history_store.init_history_db()

    monkeypatch.setattr("app.api.routes.video.DB_PATH", str(video_db))
    import app.api.routes.video as video_module
    video_module.init_jobs_db()
    monkeypatch.setattr("app.api.routes.video._process_video", lambda path: ([0.1] * 60, 7.0, 120))

    job_id = str(uuid.uuid4())
    video_module.process_video_job(job_id, str(tmp_path / "dummy.mp4"), "async_test.mp4", 35)

    items = history_store.get_history_list()
    assert len(items) == 1
    assert items[0]["filename"] == "async_test.mp4"
    assert items[0]["type"] == "video"
