"""
Database module for storing NLP analysis history.
Uses SQLite — updated schema for PhoBERT NLP results.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

# Database file path — relative to this file (backend/database/)
_BASE_DIR = Path(__file__).resolve().parent.parent  # → backend/
DB_PATH = _BASE_DIR / "resources" / "database" / "history.db"

# Ensure the database file exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
if not DB_PATH.exists():
    # Print a message when creating the database file for the first time
    print(f"📁 Creating new database file at: {DB_PATH}")
    DB_PATH.touch()

def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables — schema mới cho NLP."""
    conn = get_connection()
    cursor = conn.cursor()

    # Drop old table nếu tồn tại (schema đã thay đổi)
    cursor.execute("DROP TABLE IF EXISTS analysis_history")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Input data (NLP)
            equipment TEXT NOT NULL,
            description TEXT NOT NULL,

            -- Output data (PhoBERT results)
            fault_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            severity_score REAL DEFAULT 0.0,
            confidence REAL DEFAULT 0.0,
            keywords TEXT NOT NULL,
            recommendations TEXT NOT NULL,
            summary TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized at: {DB_PATH}")


def save_analysis(
    # Input
    equipment: str,
    description: str,
    # Output (NLP results)
    fault_type: str,
    severity: str,
    severity_score: float,
    confidence: float,
    keywords: list[str],
    recommendations: list[str],
    summary: str
) -> int:
    """
    Save an NLP analysis record to database.
    Returns the ID of the new record.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analysis_history
        (equipment, description, fault_type, severity, severity_score,
         confidence, keywords, recommendations, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        equipment, description,
        fault_type, severity, severity_score, confidence,
        json.dumps(keywords, ensure_ascii=False),
        json.dumps(recommendations, ensure_ascii=False),
        summary
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return record_id


def get_all_history(limit: int = 100, offset: int = 0) -> list[dict]:
    """Get all analysis history records."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM analysis_history
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        record = dict(row)
        record['keywords'] = json.loads(record['keywords'])
        record['recommendations'] = json.loads(record['recommendations'])
        result.append(record)

    return result


def get_history_by_id(record_id: int) -> Optional[dict]:
    """Get a specific analysis record by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        record = dict(row)
        record['keywords'] = json.loads(record['keywords'])
        record['recommendations'] = json.loads(record['recommendations'])
        return record
    return None


def delete_history(record_id: int) -> bool:
    """Delete a specific record."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
    affected = cursor.rowcount

    conn.commit()
    conn.close()

    return affected > 0


def clear_all_history() -> int:
    """Delete all history records. Returns number of deleted records."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM analysis_history")
    affected = cursor.rowcount

    conn.commit()
    conn.close()

    return affected


def get_history_count() -> int:
    """Get total number of history records."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM analysis_history")
    count = cursor.fetchone()[0]

    conn.close()
    return count


# Initialize database when module is imported
init_db()
