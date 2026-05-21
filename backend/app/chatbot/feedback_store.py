"""
feedback_store.py — Simple feedback store for future self-learning workflows.

Stores user feedback in a local SQLite database so the chatbot can later use
validated interactions for corpus improvements.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Optional

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEEDBACK_DB_PATH = os.path.join(BACKEND_ROOT, "chatbot_feedback.db")


def _get_conn():
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_store() -> None:
    conn = _get_conn()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chatbot_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                rating INTEGER,
                comment TEXT,
                session_id TEXT
            )
            """
        )
    conn.close()


def save_feedback(
    question: str,
    answer: str,
    sources: List[str],
    rating: Optional[int] = None,
    comment: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    init_feedback_store()
    conn = _get_conn()
    with conn:
        conn.execute(
            """
            INSERT INTO chatbot_feedback (
                created_at, question, answer, sources, rating, comment, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat() + "Z",
                question,
                answer,
                ",".join(sources) if sources else None,
                rating,
                comment,
                session_id,
            ),
        )
    conn.close()


def list_feedback(limit: int = 100) -> List[sqlite3.Row]:
    init_feedback_store()
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT * FROM chatbot_feedback ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
