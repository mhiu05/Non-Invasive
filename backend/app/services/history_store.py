import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "history.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db() -> None:
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            id TEXT PRIMARY KEY,
            username TEXT,
            email TEXT UNIQUE,
            hashed_password TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history(
            id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at TEXT,
            type TEXT,
            filename TEXT,
            session_id TEXT,
            duration_sec REAL,
            heart_rate REAL,
            blink_rate REAL,
            snr_db REAL,
            age INTEGER,
            age_group TEXT,
            bandpass_low_hz REAL,
            bandpass_high_hz REAL,
            hrv_ms REAL,
            sdnn_ms REAL,
            rmssd_ms REAL,
            pnn50 REAL,
            peak_count INTEGER,
            result TEXT
        )
        """
    )
    # Check if user_id column exists (for backward compatibility)
    try:
        conn.execute("ALTER TABLE history ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    conn.commit()
    conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "user_id": row["user_id"] if "user_id" in row.keys() else None,
        "created_at": row["created_at"],
        "type": row["type"],
        "filename": row["filename"],
        "session_id": row["session_id"],
        "duration_sec": row["duration_sec"],
        "heart_rate": row["heart_rate"],
        "snr_db": row["snr_db"],
        "age": row["age"],
        "age_group": row["age_group"],
        "bandpass_low_hz": row["bandpass_low_hz"],
        "bandpass_high_hz": row["bandpass_high_hz"],
        "hrv_ms": row["hrv_ms"],
        "sdnn_ms": row["sdnn_ms"],
        "rmssd_ms": row["rmssd_ms"],
        "pnn50": row["pnn50"],
        "peak_count": row["peak_count"],
        "result": json.loads(row["result"]) if row["result"] else None,
    }


def save_history_record(record: dict) -> str:
    history_id = record.get("id") or str(uuid.uuid4())
    created_at = record.get("created_at") or datetime.utcnow().isoformat()
    result_value = record.get("result")
    user_id = record.get("user_id")
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO history(
            id, user_id, created_at, type, filename, session_id, duration_sec,
            heart_rate, snr_db, age, age_group, bandpass_low_hz,
            bandpass_high_hz, hrv_ms, sdnn_ms, rmssd_ms, pnn50, peak_count, result
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            history_id,
            user_id,
            created_at,
            record.get("type"),
            record.get("filename"),
            record.get("session_id"),
            record.get("duration_sec"),
            record.get("heart_rate"),
            record.get("snr_db"),
            record.get("age"),
            record.get("age_group"),
            record.get("bandpass_low_hz"),
            record.get("bandpass_high_hz"),
            record.get("hrv_ms"),
            record.get("sdnn_ms"),
            record.get("rmssd_ms"),
            record.get("pnn50"),
            record.get("peak_count"),
            json.dumps(result_value) if result_value is not None else None,
        ),
    )
    conn.commit()
    conn.close()
    return history_id


def get_history_list(
    limit: int = 50,
    offset: int = 0,
    history_type: str | None = None,
    start_at: str | None = None,
    end_at: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM history"
    filters: list[str] = []
    params: list[object] = []

    if user_id is not None:
        filters.append("user_id = ?")
        params.append(user_id)
    if history_type is not None:
        filters.append("type = ?")
        params.append(history_type)
    if start_at is not None:
        filters.append("created_at >= ?")
        params.append(start_at)
    if end_at is not None:
        filters.append("created_at <= ?")
        params.append(end_at)

    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    conn = _get_conn()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_row_to_dict(row) for row in rows]


def get_history_by_id(history_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM history WHERE id = ?", (history_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row is not None else None


def get_user_by_email(email: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def create_user(user: dict) -> dict:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO users (id, username, email, hashed_password, created_at) VALUES (?, ?, ?, ?, ?)",
        (user["id"], user["username"], user["email"], user["hashed_password"], user["created_at"])
    )
    conn.commit()
    conn.close()
    return user


# Initialize database when module is imported.
init_history_db()
