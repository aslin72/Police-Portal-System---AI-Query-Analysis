import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "complaints.db")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_table():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_text TEXT NOT NULL,
                category TEXT NOT NULL,
                location TEXT,
                incident_time TEXT,
                persons_involved TEXT,
                summary TEXT,
                priority TEXT,
                followup_questions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_complaint(
    complaint_text, category, location, incident_time,
    persons_involved, summary, priority, followup_questions,
):
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO complaints
                (complaint_text, category, location, incident_time,
                 persons_involved, summary, priority, followup_questions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                complaint_text, category, location, incident_time,
                json.dumps(persons_involved), summary, priority,
                json.dumps(followup_questions),
            ),
        )
        conn.commit()
        complaint_id = cursor.lastrowid
    return complaint_id


def get_complaints():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM complaints ORDER BY id DESC"
        ).fetchall()

    complaints = []
    for row in rows:
        complaints.append(
            {
                "id": row["id"],
                "complaint_text": row["complaint_text"],
                "category": row["category"],
                "location": row["location"],
                "incident_time": row["incident_time"],
                "persons_involved": json.loads(row["persons_involved"]),
                "summary": row["summary"],
                "priority": row["priority"],
                "followup_questions": json.loads(row["followup_questions"]),
                "created_at": row["created_at"],
            }
        )
    return complaints
