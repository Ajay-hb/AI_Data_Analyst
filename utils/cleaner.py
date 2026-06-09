"""Auto data cleaning helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_numeric: str = "median",
    fill_categorical: str = "mode",
    strip_strings: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    report: dict[str, Any] = {"actions": []}

    if strip_strings:
        for col in out.select_dtypes(include=["object", "string"]).columns:
            out[col] = out[col].astype(str).str.strip().replace({"nan": pd.NA, "None": pd.NA})
        report["actions"].append("Stripped whitespace from text columns")

    before = len(out)
    if drop_duplicates:
        out = out.drop_duplicates()
        removed = before - len(out)
        if removed:
            report["actions"].append(f"Removed {removed} duplicate rows")

    for col in out.columns:
        if out[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            if fill_numeric == "median":
                out[col] = out[col].fillna(out[col].median())
            elif fill_numeric == "mean":
                out[col] = out[col].fillna(out[col].mean())
            else:
                out[col] = out[col].fillna(0)
            report["actions"].append(f"Filled missing numeric '{col}' with {fill_numeric}")
        else:
            mode = out[col].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) else "Unknown"
            out[col] = out[col].fillna(fill_val)
            report["actions"].append(f"Filled missing categorical '{col}' with mode")

    for col in out.columns:
        lowered = col.lower()
        if "date" in lowered and not pd.api.types.is_datetime64_any_dtype(out[col]):
            try:
                out[col] = pd.to_datetime(out[col], errors="coerce")
                report["actions"].append(f"Converted '{col}' to datetime")
            except (ValueError, TypeError):
                pass

    report["rows_after"] = len(out)
    return out, report
