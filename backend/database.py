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


def migrate():
    new_columns = [
        ("reporter_name", "TEXT"),
        ("reporter_phone", "TEXT"),
        ("reporter_email", "TEXT"),
        ("citizen_incident_location", "TEXT"),
        ("citizen_incident_time", "TEXT"),
        ("status", "TEXT DEFAULT 'New'"),
        ("assigned_unit", "TEXT"),
        ("triage_reason", "TEXT"),
        ("risk_flags", "TEXT"),
        ("recommended_action", "TEXT"),
        ("officer_notes", "TEXT"),
        ("updated_at", "TIMESTAMP"),
    ]

    with get_db() as conn:
        existing = set(
            row[1] for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
        )
        for col_name, col_def in new_columns:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE complaints ADD COLUMN {col_name} {col_def}")
        conn.commit()


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id INTEGER NOT NULL,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                content_type TEXT,
                file_size INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (complaint_id) REFERENCES complaints(id)
            )
            """
        )
        conn.commit()

    migrate()


def save_complaint(
    complaint_text, category, location, incident_time,
    persons_involved, summary, priority, followup_questions,
    reporter_name=None, reporter_phone=None, reporter_email=None,
    citizen_incident_location=None, citizen_incident_time=None,
    assigned_unit=None, triage_reason=None, risk_flags=None,
    recommended_action=None,
):
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO complaints
                (complaint_text, category, location, incident_time,
                 persons_involved, summary, priority, followup_questions,
                 reporter_name, reporter_phone, reporter_email,
                 citizen_incident_location, citizen_incident_time,
                 status, assigned_unit, triage_reason, risk_flags,
                 recommended_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New', ?, ?, ?, ?)
            """,
            (
                complaint_text, category, location, incident_time,
                json.dumps(persons_involved), summary, priority,
                json.dumps(followup_questions),
                reporter_name, reporter_phone, reporter_email,
                citizen_incident_location, citizen_incident_time,
                assigned_unit, triage_reason,
                json.dumps(risk_flags) if risk_flags else None,
                recommended_action,
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
        complaints.append(_row_to_dict(row))
    return complaints


def get_complaint(complaint_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM complaints WHERE id = ?", (complaint_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def update_triage(complaint_id, status=None, officer_notes=None):
    updates = []
    params = []
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if officer_notes is not None:
        updates.append("officer_notes = ?")
        params.append(officer_notes)
    if not updates:
        return get_complaint(complaint_id)

    from datetime import datetime, timezone
    updates.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())

    params.append(complaint_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE complaints SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
    return get_complaint(complaint_id)


def save_evidence_batch(complaint_id, records):
    with get_db() as conn:
        evidence_ids = []
        try:
            for record in records:
                cursor = conn.execute(
                    """
                    INSERT INTO evidence
                        (complaint_id, original_filename, stored_filename, file_path, content_type, file_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        complaint_id,
                        record["original_filename"],
                        record["stored_filename"],
                        record["file_path"],
                        record["content_type"],
                        record["file_size"],
                    ),
                )
                evidence_ids.append(cursor.lastrowid)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return evidence_ids


def get_evidence_by_complaint(complaint_id):
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, complaint_id, original_filename, stored_filename,
                   content_type, file_size, uploaded_at
            FROM evidence
            WHERE complaint_id = ?
            ORDER BY uploaded_at DESC
            """,
            (complaint_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_evidence_by_id(evidence_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM evidence WHERE id = ?", (evidence_id,)
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def _row_to_dict(row):
    def safe_json(value):
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    return {
        "id": row["id"],
        "complaint_text": row["complaint_text"],
        "category": row["category"],
        "location": row["location"],
        "incident_time": row["incident_time"],
        "persons_involved": safe_json(row["persons_involved"]) or [],
        "summary": row["summary"],
        "priority": row["priority"],
        "followup_questions": safe_json(row["followup_questions"]) or [],
        "created_at": row["created_at"],
        "reporter_name": row["reporter_name"],
        "reporter_phone": row["reporter_phone"],
        "reporter_email": row["reporter_email"],
        "citizen_incident_location": row["citizen_incident_location"],
        "citizen_incident_time": row["citizen_incident_time"],
        "status": row["status"] or "New",
        "assigned_unit": row["assigned_unit"],
        "triage_reason": row["triage_reason"],
        "risk_flags": safe_json(row["risk_flags"]) or [],
        "recommended_action": row["recommended_action"],
        "officer_notes": row["officer_notes"],
        "updated_at": row["updated_at"],
    }
