"""Multi-step agent workflow: profile → analyze → visualize → forecast → recommend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from agents.csv_agent import CSVAgent, QueryResult
from agents.forecast_agent import ForecastResult, forecast_series
from agents.insight_agent import generate_insights
from tools.chart_tool import create_chart, suggest_chart_type
from utils.profiler import profile_dataframe, profile_summary_text


@dataclass
class AnalysisRun:
    question: str
    profile: dict[str, Any]
    profile_text: str
    query: QueryResult | None = None
    chart: go.Figure | None = None
    insights: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    forecast: ForecastResult | None = None


class AnalystWorkflow:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile = profile_dataframe(df)
        self.profile_text = profile_summary_text(self.profile)
        self.agent = CSVAgent(df)

    def run_question(self, question: str, include_forecast: bool = False, forecast_periods: int = 6) -> AnalysisRun:
        run = AnalysisRun(
            question=question,
            profile=self.profile,
            profile_text=self.profile_text,
        )

        run.query = self.agent.ask(question)
        if run.query.error:
            run.insights = [f"Query error: {run.query.error}"]
            return run

        qdf = run.query.data if isinstance(run.query.data, pd.DataFrame) else pd.DataFrame()
        chart_type = suggest_chart_type(question, qdf if not qdf.empty else self.df)
        run.chart = create_chart(qdf if not qdf.empty else self.df.head(50), chart_type=chart_type, title=question)

        run.insights, run.recommendations = generate_insights(
            self.profile, qdf if not qdf.empty else None, question, self.profile_text
        )

        if include_forecast or any(w in question.lower() for w in ("forecast", "predict", "next month", "next 6")):
            run.forecast = forecast_series(self.df, periods=forecast_periods)

        return run

    def close(self) -> None:
        if hasattr(self, "agent") and self.agent:
            try:
                self.agent.close()
            except Exception:
                pass
