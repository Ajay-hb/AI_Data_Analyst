"""Natural language → Pandas / SQL agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from agents.llm_client import extract_code_block, fallback_pandas_code, fallback_sql, llm_invoke
from tools.pandas_tool import execute_pandas_code, result_to_display
from tools.sql_tool import SQLTool


@dataclass
class QueryResult:
    question: str
    mode: Literal["pandas", "sql"]
    code: str
    data: pd.DataFrame | Any
    error: str | None = None


class CSVAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.sql_tool = SQLTool(df)

    def _pandas_prompt(self, question: str) -> str:
        cols = ", ".join(self.df.columns.tolist())
        sample = self.df.head(3).to_string()
        return f"""You are a data analyst. Given a pandas DataFrame `df`, write Python code that sets `result`.
Use only pandas/numpy. No imports. Assign final answer to variable `result`.

Columns: {cols}
Sample:
{sample}

Question: {question}

Return ONLY a python code block."""

    def _sql_prompt(self, question: str) -> str:
        return f"""You are a SQL analyst. Table name is `dataset`.
Schema:
{self.sql_tool.schema()}

Question: {question}

Write a single read-only SELECT query. Return ONLY a sql code block."""

    def ask(self, question: str, mode: str = "auto") -> QueryResult:
        use_sql = mode == "sql" or (mode == "auto" and any(w in question.lower() for w in ("select", "sql", "query")))

        if use_sql:
            return self._run_sql(question)
        return self._run_pandas(question)

    def _run_pandas(self, question: str) -> QueryResult:
        code = None
        llm_text = llm_invoke(self._pandas_prompt(question))
        if llm_text:
            code = extract_code_block(llm_text, "python")
        if not code:
            code = fallback_pandas_code(self.df, question)

        try:
            raw, code = execute_pandas_code(self.df, code)
            data = result_to_display(raw)
            return QueryResult(question=question, mode="pandas", code=code, data=data)
        except Exception as e:
            return QueryResult(question=question, mode="pandas", code=code, data=pd.DataFrame(), error=str(e))

    def _run_sql(self, question: str) -> QueryResult:
        sql = None
        llm_text = llm_invoke(self._sql_prompt(question))
        if llm_text:
            sql = extract_code_block(llm_text, "sql")
        if not sql:
            sql = fallback_sql(self.df, question)

        try:
            data = self.sql_tool.execute(sql)
            return QueryResult(question=question, mode="sql", code=sql, data=data)
        except Exception as e:
            return QueryResult(question=question, mode="sql", code=sql, data=pd.DataFrame(), error=str(e))

    def close(self) -> None:
        if hasattr(self, "sql_tool") and self.sql_tool:
            try:
                self.sql_tool.close()
            except Exception:
                pass
