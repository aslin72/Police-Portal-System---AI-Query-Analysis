import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "complaints.db")
NEW_COLUMNS = {
    "injured": "TEXT", "money_lost": "TEXT", "evidence_available": "TEXT",
    "status": "TEXT DEFAULT 'New'", "assigned_unit": "TEXT", "triage_reason": "TEXT",
    "risk_flags": "TEXT", "recommended_action": "TEXT", "officer_notes": "TEXT",
    "updated_at": "TEXT",
}


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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT, complaint_text TEXT NOT NULL,
                category TEXT NOT NULL, location TEXT, incident_time TEXT,
                persons_involved TEXT, summary TEXT, priority TEXT,
                followup_questions TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(complaints)")}
        for name, definition in NEW_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE complaints ADD COLUMN {name} {definition}")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT, complaint_id INTEGER NOT NULL,
                original_filename TEXT NOT NULL, stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL, content_type TEXT, file_size INTEGER,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (complaint_id) REFERENCES complaints(id)
            )
        """)
        conn.commit()


def save_complaint(payload, ai, triage, questions):
    values = (
        payload["complaint_text"], ai["category"], payload["location"] or ai["location"],
        payload["incident_time"] or ai["incident_time"], json.dumps(ai["persons_involved"]),
        ai["summary"], triage["priority"], json.dumps(questions), payload["injured"],
        payload["money_lost"], payload["evidence_available"], triage["assigned_unit"],
        triage["triage_reason"], json.dumps(triage["risk_flags"]), triage["recommended_action"],
    )
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO complaints (
                complaint_text, category, location, incident_time, persons_involved,
                summary, priority, followup_questions, injured, money_lost,
                evidence_available, status, assigned_unit, triage_reason, risk_flags,
                recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New', ?, ?, ?, ?)
        """, values)
        conn.commit()
        return cursor.lastrowid


def get_complaint(complaint_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()
    return _row(row) if row else None


def get_complaints():
    order = "CASE priority WHEN 'Emergency' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END"
    with get_db() as conn:
        rows = conn.execute(f"SELECT * FROM complaints ORDER BY {order}, id DESC").fetchall()
    return [_row(row) for row in rows]


def update_triage(complaint_id, status, notes):
    with get_db() as conn:
        conn.execute(
            "UPDATE complaints SET status = ?, officer_notes = ?, updated_at = ? WHERE id = ?",
            (status, notes, datetime.now(timezone.utc).isoformat(), complaint_id),
        )
        conn.commit()
    return get_complaint(complaint_id)


def save_evidence(complaint_id, records):
    with get_db() as conn:
        ids = []
        for record in records:
            cursor = conn.execute("""
                INSERT INTO evidence (complaint_id, original_filename, stored_filename,
                    file_path, content_type, file_size) VALUES (?, ?, ?, ?, ?, ?)
            """, (complaint_id, record["original_filename"], record["stored_filename"],
                    record["file_path"], record["content_type"], record["file_size"]))
            ids.append(cursor.lastrowid)
        conn.commit()
    return ids


def get_evidence(complaint_id):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, complaint_id, original_filename, content_type, file_size, uploaded_at
            FROM evidence WHERE complaint_id = ? ORDER BY id DESC
        """, (complaint_id,)).fetchall()
    return [dict(row) for row in rows]


def get_evidence_file(evidence_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
    return dict(row) if row else None


def _row(row):
    data = dict(row)
    for field in ("persons_involved", "followup_questions", "risk_flags"):
        try:
            data[field] = json.loads(data.get(field) or "[]")
        except (json.JSONDecodeError, TypeError):
            data[field] = []
    data["status"] = data.get("status") or "New"
    return data
