"""
Общая схема embeddings.db (SQLite):
  - fact_embeddings — текстовые факты и их векторы
  - char_embeddings — персонажи user_001: face_embedding, voice_embedding
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

EMBEDDING_DB_PATH = Path(__file__).resolve().parent / "embeddings.db"
FACT_TABLE = "fact_embeddings"
CHAR_TABLE = "char_embeddings"
LEGACY_TABLE = "embeddings"


def normalize_user_name(name: str) -> str:
    """user_1 / user_01 → user_001."""
    if not name.startswith("user_"):
        return name
    suffix = name[5:]
    if suffix.isdigit():
        return f"user_{int(suffix):03d}"
    return name


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def init_embedding_db(conn: sqlite3.Connection) -> None:
    """Создаёт fact_embeddings и char_embeddings, мигрирует старую таблицу embeddings."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FACT_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            model TEXT NOT NULL,
            source_file TEXT,
            classified_file TEXT,
            created_at TEXT NOT NULL,
            UNIQUE (fact_text, source_file)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CHAR_TABLE} (
            user_name TEXT PRIMARY KEY,
            face_embedding BLOB NOT NULL,
            voice_embedding BLOB,
            created_at REAL NOT NULL
        )
        """
    )
    _migrate_legacy_embeddings(conn)
    _normalize_char_user_names(conn)
    conn.commit()


def _migrate_legacy_embeddings(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, LEGACY_TABLE):
        return

    cols = conn.execute(f"PRAGMA table_info({LEGACY_TABLE})").fetchall()
    col_names = {row[1] for row in cols}
    if "face_embedding" in col_names:
        face_col = "face_embedding"
    elif "embedding" in col_names:
        face_col = "embedding"
    else:
        return

    voice_sel = "voice_embedding" if "voice_embedding" in col_names else "NULL"
    rows = conn.execute(
        f"SELECT user_name, {face_col}, {voice_sel}, created_at FROM {LEGACY_TABLE}"
    ).fetchall()

    for user_name, face_blob, voice_blob, created_at in rows:
        if face_blob is None:
            continue
        norm = normalize_user_name(str(user_name))
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {CHAR_TABLE}
                (user_name, face_embedding, voice_embedding, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (norm, face_blob, voice_blob, float(created_at or 0)),
        )


def _normalize_char_user_names(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        f"SELECT user_name, face_embedding, voice_embedding, created_at FROM {CHAR_TABLE}"
    ).fetchall()
    for old_name, face_blob, voice_blob, created_at in rows:
        new_name = normalize_user_name(str(old_name))
        if new_name == old_name:
            continue
        conn.execute(f"DELETE FROM {CHAR_TABLE} WHERE user_name = ?", (old_name,))
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {CHAR_TABLE}
                (user_name, face_embedding, voice_embedding, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (new_name, face_blob, voice_blob, created_at),
        )
