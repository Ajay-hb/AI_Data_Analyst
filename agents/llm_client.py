"""LLM client with Gemini/OpenAI support and offline fallback."""
from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd


def get_llm():
    """Return a LangChain chat model if API keys are configured."""
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
    except ImportError:
        return None

    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        except Exception:
            pass

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        except Exception:
            pass

    return None


def llm_invoke(prompt: str) -> str | None:
    llm = get_llm()
    if llm is None:
        return None
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        return None


def _guess_dimension_metric(df: pd.DataFrame, question: str) -> tuple[str | None, str | None]:
    q = question.lower()
    cols = df.columns.tolist()
    numeric = df.select_dtypes(include="number").columns.tolist()
    cats = [c for c in cols if c not in numeric]

    metric = None
    for c in numeric:
        if c.lower() in q or any(w in c.lower() for w in q.split() if len(w) > 3):
            metric = c
            break
    if metric is None:
        for word in ("sales", "revenue", "profit", "demand", "price", "amount", "total"):
            if word in q:
                matches = [c for c in numeric if word in c.lower()]
                if matches:
                    metric = matches[0]
                    break
    if metric is None and numeric:
        metric = numeric[0]

    dim = None
    for c in cats:
        if c.lower() in q or any(w in c.lower() for w in ("region", "state", "product", "category", "customer", "month")):
            dim = c
            break
    if dim is None:
        for word in ("region", "state", "product", "category", "customer", "month", "store"):
            if word in q:
                matches = [c for c in cats if word in c.lower()]
                if matches:
                    dim = matches[0]
                    break

    return dim, metric


def fallback_pandas_code(df: pd.DataFrame, question: str) -> str:
    q = question.lower()
    dim, metric = _guess_dimension_metric(df, question)

    if "top" in q and metric:
        n = 10
        m = re.search(r"top\s+(\d+)", q)
        if m:
            n = int(m.group(1))
        if dim:
            return f'result = df.groupby("{dim}")["{metric}"].sum().sort_values(ascending=False).head({n}).reset_index()'
        return f'result = df.nlargest({n}, "{metric}")'

    if dim and metric and any(w in q for w in ("highest", "most", "by", "which", "group")):
        return f'result = df.groupby("{dim}")["{metric}"].sum().sort_values(ascending=False).reset_index()'

    if "average" in q or "mean" in q:
        if dim and metric:
            return f'result = df.groupby("{dim}")["{metric}"].mean().sort_values(ascending=False).reset_index()'
        if metric:
            return f'result = pd.DataFrame({{"average_{metric}": [df["{metric}"].mean()]}})'

    if any(w in q for w in ("trend", "monthly", "over time")) and metric:
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            return (
                f'_tmp = df.copy()\n'
                f'_tmp["{date_col}"] = pd.to_datetime(_tmp["{date_col}"], errors="coerce")\n'
                f'_tmp = _tmp.dropna(subset=["{date_col}"])\n'
                f'result = _tmp.groupby(_tmp["{date_col}"].dt.to_period("M"))["{metric}"].sum().reset_index()\n'
                f'result["{date_col}"] = result["{date_col}"].astype(str)'
            )

    if metric:
        return f'result = df["{metric}"].describe().to_frame().T'
    return "result = df.head(20)"


def fallback_sql(df: pd.DataFrame, question: str) -> str:
    dim, metric = _guess_dimension_metric(df, question)
    if dim and metric:
        return (
            f"SELECT {dim}, SUM({metric}) AS total_{metric.lower()} "
            f"FROM dataset GROUP BY {dim} ORDER BY total_{metric.lower()} DESC LIMIT 15"
        )
    return "SELECT * FROM dataset LIMIT 20"


def extract_code_block(text: str, lang: str = "python") -> str:
    pattern = rf"```{lang}\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    pattern = r"```\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
