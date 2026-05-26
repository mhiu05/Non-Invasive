import os
import sqlite3
import psycopg2
from psycopg2 import sql

DB_URL = "postgresql://postgres.yoyebrolahzxcvosojun:%40Hl12119520105@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres"

def init_tables(pg_conn):
    with pg_conn.cursor() as cur:
        # Bảng jobs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT,
                created_at TIMESTAMP WITH TIME ZONE,
                updated_at TIMESTAMP WITH TIME ZONE,
                result TEXT,
                error TEXT,
                file_path TEXT
            );
        """)
        
        # Bảng users
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT,
                email TEXT UNIQUE,
                hashed_password TEXT,
                created_at TIMESTAMP WITH TIME ZONE
            );
        """)
        
        # Bảng history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                user_id TEXT REFERENCES users(id),
                created_at TIMESTAMP WITH TIME ZONE,
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
            );
        """)
        
        # Bảng chatbot_feedback
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chatbot_feedback (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                rating INTEGER,
                comment TEXT,
                session_id TEXT
            );
        """)
    pg_conn.commit()

def migrate_data():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    pg_conn = psycopg2.connect(DB_URL)
    init_tables(pg_conn)
    print("Tables initialized.")
    
    cur_pg = pg_conn.cursor()
    
    # Migrate users and history from history.db
    history_db_path = os.path.join(backend_dir, "history.db")
    if os.path.exists(history_db_path):
        sqlite_conn = sqlite3.connect(history_db_path)
        sqlite_conn.row_factory = sqlite3.Row
        cur_lite = sqlite_conn.cursor()
        
        # Users
        try:
            cur_lite.execute("SELECT * FROM users")
            users = cur_lite.fetchall()
            for u in users:
                cur_pg.execute(
                    "INSERT INTO users (id, username, email, hashed_password, created_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (u["id"], u["username"], u["email"], u["hashed_password"], u["created_at"])
                )
            print(f"Migrated {len(users)} users.")
        except sqlite3.OperationalError:
            pass

        # History
        try:
            cur_lite.execute("SELECT * FROM history")
            history = cur_lite.fetchall()
            for h in history:
                cur_pg.execute(
                    """
                    INSERT INTO history (
                        id, user_id, created_at, type, filename, session_id, duration_sec,
                        heart_rate, blink_rate, snr_db, age, age_group, bandpass_low_hz,
                        bandpass_high_hz, hrv_ms, sdnn_ms, rmssd_ms, pnn50, peak_count, result
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        h["id"], h["user_id"] if "user_id" in h.keys() else None, h["created_at"], h["type"], h["filename"],
                        h["session_id"], h["duration_sec"], h["heart_rate"], 
                        h["blink_rate"] if "blink_rate" in h.keys() else None, 
                        h["snr_db"], h["age"], h["age_group"], 
                        h["bandpass_low_hz"], h["bandpass_high_hz"], h["hrv_ms"],
                        h["sdnn_ms"], h["rmssd_ms"], h["pnn50"], h["peak_count"], h["result"]
                    )
                )
            print(f"Migrated {len(history)} history records.")
        except sqlite3.OperationalError:
            pass
            
        sqlite_conn.close()
        
    # Migrate jobs from video_jobs.db
    video_db_path = os.path.join(backend_dir, "video_jobs.db")
    if os.path.exists(video_db_path):
        sqlite_conn = sqlite3.connect(video_db_path)
        sqlite_conn.row_factory = sqlite3.Row
        cur_lite = sqlite_conn.cursor()
        try:
            cur_lite.execute("SELECT * FROM jobs")
            jobs = cur_lite.fetchall()
            for j in jobs:
                cur_pg.execute(
                    """
                    INSERT INTO jobs (id, status, created_at, updated_at, result, error, file_path) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING
                    """,
                    (j["id"], j["status"], j["created_at"], j["updated_at"], j["result"], j["error"], j["file_path"])
                )
            print(f"Migrated {len(jobs)} jobs.")
        except sqlite3.OperationalError:
            pass
        sqlite_conn.close()
        
    # Migrate chatbot feedback
    feedback_db_path = os.path.join(backend_dir, "chatbot_feedback.db")
    if os.path.exists(feedback_db_path):
        sqlite_conn = sqlite3.connect(feedback_db_path)
        sqlite_conn.row_factory = sqlite3.Row
        cur_lite = sqlite_conn.cursor()
        try:
            cur_lite.execute("SELECT * FROM chatbot_feedback")
            feedbacks = cur_lite.fetchall()
            for fb in feedbacks:
                cur_pg.execute(
                    """
                    INSERT INTO chatbot_feedback (id, created_at, question, answer, sources, rating, comment, session_id) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING
                    """,
                    (fb["id"], fb["created_at"], fb["question"], fb["answer"], fb["sources"], fb["rating"], fb["comment"], fb["session_id"])
                )
            
            # Reset sequence if needed
            if feedbacks:
                cur_pg.execute("SELECT setval('chatbot_feedback_id_seq', (SELECT MAX(id) FROM chatbot_feedback));")
                
            print(f"Migrated {len(feedbacks)} feedback records.")
        except sqlite3.OperationalError:
            pass
        sqlite_conn.close()

    pg_conn.commit()
    cur_pg.close()
    pg_conn.close()
    print("Migration finished!")

if __name__ == "__main__":
    migrate_data()
