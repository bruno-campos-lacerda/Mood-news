import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent / "data" / "chat_history.sqlite3"


@dataclass(frozen=True)
class ChatResult:
    id: int
    news: str
    positive_probability: float
    negative_probability: float
    created_at: str


class ChatHistoryRepository:
    def __init__(self, database_path: Path = DEFAULT_HISTORY_PATH):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news TEXT NOT NULL,
                    positive_probability REAL NOT NULL,
                    negative_probability REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self):
        return sqlite3.connect(self.database_path, timeout=10)

    def add(
        self,
        news: str,
        positive_probability: float,
        negative_probability: float,
    ) -> ChatResult:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO chat_results (
                    news, positive_probability, negative_probability, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (news, positive_probability, negative_probability, created_at),
            )
            result_id = cursor.lastrowid
        return ChatResult(
            id=result_id,
            news=news,
            positive_probability=positive_probability,
            negative_probability=negative_probability,
            created_at=created_at,
        )

    def list_all(self) -> list[ChatResult]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, news, positive_probability, negative_probability, created_at
                FROM chat_results
                ORDER BY id
                """
            ).fetchall()
        return [ChatResult(*row) for row in rows]
