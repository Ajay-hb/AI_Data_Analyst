"""SQLite-backed SQL tool for natural language analytics."""
from __future__ import annotations

import re
import sqlite3
from typing import Any

import pandas as pd

TABLE_NAME = "dataset"


class SQLTool:
    def __init__(self, df: pd.DataFrame, table_name: str = TABLE_NAME):
        self.table_name = table_name
        self.conn = sqlite3.connect(":memory:")
        df.to_sql(table_name, self.conn, index=False, if_exists="replace")

    def schema(self) -> str:
        cur = self.conn.execute(f"PRAGMA table_info({self.table_name})")
        rows = cur.fetchall()
        return "\n".join(f"- {r[1]} ({r[2]})" for r in rows)

    def execute(self, sql: str) -> pd.DataFrame:
        sql = sql.strip().rstrip(";")
        if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed.")
        forbidden = re.compile(r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|ATTACH|DETACH)\b", re.I)
        if forbidden.search(sql):
            raise ValueError("Only read-only SELECT queries are allowed.")
        return pd.read_sql_query(sql, self.conn)

    def close(self) -> None:
        self.conn.close()

    def sample_queries(self) -> list[str]:
        return [
            f"SELECT * FROM {self.table_name} LIMIT 10",
            f"SELECT COUNT(*) AS row_count FROM {self.table_name}",
        ]
