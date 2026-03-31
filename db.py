import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "triplets.db"


def get_database_path() -> Path:
    return DB_PATH


def _get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with _get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                source_type TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                raw_triplets_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE TABLE IF NOT EXISTS triplets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                extraction_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                FOREIGN KEY (extraction_id) REFERENCES extractions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(subject);
            CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplets(object);
            CREATE INDEX IF NOT EXISTS idx_triplets_extraction_id ON triplets(extraction_id);
            """
        )


def save_triplets_payload(
    *,
    source_name: str,
    source_type: str,
    method: str,
    triplets_payload: Dict[str, Any],
) -> Tuple[int, int]:
    if not isinstance(triplets_payload, dict):
        raise ValueError("Ожидался JSON-объект с ключом 'triplets'.")

    raw_triplets = triplets_payload.get("triplets")
    if not isinstance(raw_triplets, list):
        raise ValueError("В JSON отсутствует список 'triplets'.")

    normalized_triplets = []
    for idx, item in enumerate(raw_triplets, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Триплет #{idx} имеет неверный формат.")

        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()

        if not subject and not predicate and not obj:
            continue

        normalized_triplets.append((subject, predicate, obj))

    timestamp = datetime.now(timezone.utc).isoformat()
    payload_json = json.dumps(triplets_payload, ensure_ascii=False, indent=2)

    with _get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO documents (source_name, source_type, created_at)
            VALUES (?, ?, ?)
            """,
            (source_name, source_type, timestamp),
        )
        document_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO extractions (document_id, method, raw_triplets_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, method, payload_json, timestamp),
        )
        extraction_id = cursor.lastrowid

        cursor.executemany(
            """
            INSERT INTO triplets (extraction_id, subject, predicate, object)
            VALUES (?, ?, ?, ?)
            """,
            [(extraction_id, subject, predicate, obj) for subject, predicate, obj in normalized_triplets],
        )
        connection.commit()

    return extraction_id, len(normalized_triplets)


def get_saved_extractions(
    *,
    search_term: str = "",
    search_field: str = "all",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    allowed_fields = {"all", "subject", "object"}
    if search_field not in allowed_fields:
        raise ValueError("Недопустимое поле поиска.")

    search_term = search_term.strip()
    with _get_connection() as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        params: List[Any] = []
        where_clause = ""
        if search_term:
            like_value = f"%{search_term}%"
            if search_field == "subject":
                where_clause = "WHERE EXISTS (SELECT 1 FROM triplets t WHERE t.extraction_id = e.id AND t.subject LIKE ? COLLATE NOCASE)"
                params.append(like_value)
            elif search_field == "object":
                where_clause = "WHERE EXISTS (SELECT 1 FROM triplets t WHERE t.extraction_id = e.id AND t.object LIKE ? COLLATE NOCASE)"
                params.append(like_value)
            else:
                where_clause = (
                    "WHERE EXISTS (SELECT 1 FROM triplets t "
                    "WHERE t.extraction_id = e.id AND (t.subject LIKE ? COLLATE NOCASE OR t.object LIKE ? COLLATE NOCASE))"
                )
                params.extend([like_value, like_value])

        params.append(limit)
        rows = cursor.execute(
            f"""
            SELECT
                e.id AS extraction_id,
                e.method,
                e.created_at,
                d.id AS document_id,
                d.source_name,
                d.source_type,
                COUNT(t.id) AS triplets_count
            FROM extractions e
            JOIN documents d ON d.id = e.document_id
            LEFT JOIN triplets t ON t.extraction_id = e.id
            {where_clause}
            GROUP BY e.id, e.method, e.created_at, d.id, d.source_name, d.source_type
            ORDER BY e.id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        extractions: List[Dict[str, Any]] = []
        for row in rows:
            triplets = cursor.execute(
                """
                SELECT id, subject, predicate, object
                FROM triplets
                WHERE extraction_id = ?
                ORDER BY id ASC
                """,
                (row["extraction_id"],),
            ).fetchall()

            extractions.append(
                {
                    "extraction_id": row["extraction_id"],
                    "document_id": row["document_id"],
                    "method": row["method"],
                    "created_at": row["created_at"],
                    "source_name": row["source_name"],
                    "source_type": row["source_type"],
                    "triplets_count": row["triplets_count"],
                    "triplets": [dict(triplet) for triplet in triplets],
                }
            )

    return extractions


def delete_extraction(extraction_id: int) -> bool:
    with _get_connection() as connection:
        cursor = connection.cursor()
        row = cursor.execute(
            "SELECT document_id FROM extractions WHERE id = ?",
            (extraction_id,),
        ).fetchone()
        if row is None:
            return False

        document_id = row[0]
        cursor.execute("DELETE FROM triplets WHERE extraction_id = ?", (extraction_id,))
        cursor.execute("DELETE FROM extractions WHERE id = ?", (extraction_id,))

        remaining = cursor.execute(
            "SELECT COUNT(*) FROM extractions WHERE document_id = ?",
            (document_id,),
        ).fetchone()[0]
        if remaining == 0:
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

        connection.commit()
    return True
