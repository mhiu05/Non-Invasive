"""Basic API tests — chạy: pytest tests/"""

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
