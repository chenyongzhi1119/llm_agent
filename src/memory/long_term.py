"""Long-term memory - SQLite persistent storage"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


DB_PATH = Path("data/agent.db")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist"""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id);
        """)


def save_message(session_id: str, role: str, content: str) -> None:
    """Save a single message to the database"""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )


def load_history(session_id: str, limit: int = 50) -> list[dict]:
    """Load recent conversation history for a session"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    # Return in chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_history(session_id: str) -> None:
    """Delete all messages for a session"""
    with _get_conn() as conn:
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))


def set_preference(key: str, value: str) -> None:
    """Save or update a user preference"""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, datetime.now().isoformat()),
        )


def get_preference(key: str, default: str = "") -> str:
    """Get a user preference value"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM user_preferences WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def list_sessions() -> list[str]:
    """List all session IDs"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM conversations ORDER BY session_id"
        ).fetchall()
    return [r["session_id"] for r in rows]
