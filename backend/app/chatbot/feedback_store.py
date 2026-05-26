"""
feedback_store.py — Simple feedback store for future self-learning workflows.

Stores user feedback in a local SQLite database so the chatbot can later use
validated interactions for corpus improvements.
"""

import os
from datetime import datetime
from typing import List, Optional
import psycopg2
import psycopg2.extras

DB_URL = os.getenv("SUPABASE_DB_URL")


def _get_conn():
    conn = psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.DictCursor)
    return conn


def init_feedback_store() -> None:
    pass


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
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chatbot_feedback (
                created_at, question, answer, sources, rating, comment, session_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
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
    conn.commit()
    conn.close()


def list_feedback(limit: int = 100) -> List[dict]:
    init_feedback_store()
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM chatbot_feedback ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]
