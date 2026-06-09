"""Business insight and recommendation generation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from agents.llm_client import llm_invoke


def _rule_insights(profile: dict[str, Any], query_df: pd.DataFrame | None, question: str) -> list[str]:
    insights: list[str] = []

    if profile.get("duplicate_rows", 0) > 0:
        insights.append(
            f"Data quality: {profile['duplicate_rows']} duplicate rows detected — consider deduplication before KPI reporting."
        )

    missing = profile.get("missing_values", [])
    if missing:
        top = missing[0]
        insights.append(
            f"Data quality: '{top['column']}' has {top['pct']}% missing values — imputation or exclusion may affect accuracy."
        )

    if query_df is not None and not query_df.empty and query_df.select_dtypes(include="number").shape[1] >= 1:
        num_col = query_df.select_dtypes(include="number").columns[-1]
        series = query_df[num_col].dropna()
        if len(series) >= 2:
            top_row = query_df.iloc[0]
            dim_col = [c for c in query_df.columns if c != num_col]
            label = top_row[dim_col[0]] if dim_col else "Top segment"
            insights.append(
                f"Analysis: '{label}' leads on {num_col} with value {top_row[num_col]:,.2f} for the question: \"{question}\"."
            )
            if len(series) >= 3:
                spread = series.max() - series.min()
                insights.append(
                    f"Spread across segments on {num_col} is {spread:,.2f}, indicating meaningful performance differences."
                )

    outliers = profile.get("outliers", [])
    if outliers:
        insights.append(
            f"Outliers detected in {outliers[0]['column']} — validate extreme values before strategic decisions."
        )

    if not insights:
        insights.append("Upload complete. Ask questions about trends, top performers, and averages to generate deeper insights.")
    return insights


def _rule_recommendations(insights: list[str], profile: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    if any("duplicate" in i.lower() for i in insights):
        recs.append("Run automated data cleaning to remove duplicates before executive reporting.")
    if profile.get("outliers"):
        recs.append(f"Investigate outliers in {profile['outliers'][0]['column']} with domain experts.")
    if any("leads on" in i.lower() for i in insights):
        recs.append("Double down on marketing and inventory for the top-performing segment identified.")
    if any("missing" in i.lower() for i in insights):
        recs.append("Improve data collection for columns with high missing rates to strengthen forecast accuracy.")
    if not recs:
        recs.append("Define KPIs (revenue, retention, margin) and track them weekly using this assistant.")
    return recs[:5]


def generate_insights(
    profile: dict[str, Any],
    query_df: pd.DataFrame | None,
    question: str,
    profile_text: str,
) -> tuple[list[str], list[str]]:
    context = profile_text
    if query_df is not None and not query_df.empty:
        context += f"\n\nQuery result sample:\n{query_df.head(10).to_string()}"

    prompt = f"""You are a Senior Data Analyst. Generate concise business insights and actionable recommendations.

Dataset profile:
{context}

User question: {question}

Format:
INSIGHTS:
- bullet 1
- bullet 2

RECOMMENDATIONS:
- action 1
- action 2
"""
    llm_text = llm_invoke(prompt)
    if llm_text:
        insights, recs = _parse_llm_sections(llm_text)
        if insights:
            return insights, recs

    insights = _rule_insights(profile, query_df, question)
    recs = _rule_recommendations(insights, profile)
    return insights, recs


def _parse_llm_sections(text: str) -> tuple[list[str], list[str]]:
    insights: list[str] = []
    recs: list[str] = []
    section = None
    for line in text.splitlines():
        upper = line.strip().upper()
        if upper.startswith("INSIGHTS"):
            section = "insights"
            continue
        if upper.startswith("RECOMMENDATIONS"):
            section = "recs"
            continue
        if line.strip().startswith(("-", "•", "*")):
            item = line.strip().lstrip("-•* ").strip()
            if section == "insights" and item:
                insights.append(item)
            elif section == "recs" and item:
                recs.append(item)
    return insights, recs
