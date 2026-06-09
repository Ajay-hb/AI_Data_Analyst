"""Automatic dataset profiling for uploaded CSVs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _outlier_columns(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict[str, Any]]:
    flagged = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count > 0:
            flagged.append({"column": col, "outlier_count": count, "pct": round(100 * count / len(series), 2)})
    return flagged


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = [
        {"column": c, "missing": int(missing[c]), "pct": float(missing_pct[c])}
        for c in df.columns
        if missing[c] > 0
    ]
    missing_report.sort(key=lambda x: x["pct"], reverse=True)

    corr = None
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(3)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "numeric_columns": numeric_cols,
        "categorical_columns": object_cols,
        "datetime_columns": datetime_cols,
        "missing_values": missing_report,
        "duplicate_rows": int(df.duplicated().sum()),
        "describe_numeric": df[numeric_cols].describe().to_dict() if numeric_cols else {},
        "describe_object": df[object_cols].describe().to_dict() if object_cols else {},
        "outliers": _outlier_columns(df, numeric_cols),
        "correlation": corr.to_dict() if corr is not None else None,
        "sample_head": df.head(5).to_dict(orient="records"),
    }


def profile_summary_text(profile: dict[str, Any]) -> str:
    lines = [
        f"Rows: {profile['rows']:,}",
        f"Columns: {profile['columns']}",
        f"Duplicates: {profile['duplicate_rows']}",
    ]
    if profile["missing_values"]:
        lines.append("Missing values:")
        for m in profile["missing_values"][:8]:
            lines.append(f"  - {m['column']}: {m['pct']}%")
    if profile["outliers"]:
        lines.append("Potential outliers:")
        for o in profile["outliers"][:5]:
            lines.append(f"  - {o['column']}: {o['outlier_count']} rows ({o['pct']}%)")
    return "\n".join(lines)
