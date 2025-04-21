import sqlite3
from datetime import datetime
from typing import Dict, Any
import tiktoken


class LLMLogger:
    def __init__(self, db_path: str = "llm_logs.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_provider TEXT,
                model_name TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                response TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                duration_ms INTEGER
            )
        """
        )
        conn.commit()
        conn.close()

    def _count_tokens(self, text: str, model: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # Fallback to approximate counting if model not found
            return len(text.split()) // 2

    def log_call(
        self,
        model_provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        duration_ms: int,
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        input_tokens = self._count_tokens(
            system_prompt or "", model_name
        ) + self._count_tokens(user_prompt, model_name)
        output_tokens = self._count_tokens(response, model_name)

        cursor.execute(
            """
            INSERT INTO llm_calls (
                timestamp, model_provider, model_name, system_prompt,
                user_prompt, response, input_tokens, output_tokens, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                model_provider,
                model_name,
                system_prompt,
                user_prompt,
                response,
                input_tokens,
                output_tokens,
                duration_ms,
            ),
        )

        conn.commit()
        conn.close()

    def get_all_logs(self) -> list[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM llm_calls ORDER BY timestamp DESC")
        columns = [description[0] for description in cursor.description]
        logs = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return logs
