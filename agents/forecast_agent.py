"""Time-series forecasting with Prophet (optional) or simple trend fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    success: bool
    message: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    growth_pct: float | None
    next_value: float | None
    method: str


def _detect_date_metric(df: pd.DataFrame, metric_hint: str | None = None) -> tuple[str | None, str | None]:
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
    numeric = df.select_dtypes(include="number").columns.tolist()
    metric = None
    if metric_hint:
        matches = [c for c in numeric if metric_hint.lower() in c.lower()]
        metric = matches[0] if matches else None
    if metric is None:
        for word in ("revenue", "sales", "demand", "amount", "profit"):
            matches = [c for c in numeric if word in c.lower()]
            if matches:
                metric = matches[0]
                break
    if metric is None and numeric:
        metric = numeric[0]
    return date_col, metric


def _prepare_series(df: pd.DataFrame, date_col: str, metric: str) -> pd.DataFrame:
    work = df[[date_col, metric]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna()
    work = work.groupby(date_col, as_index=False)[metric].sum()
    work = work.sort_values(date_col)
    return work


def forecast_series(
    df: pd.DataFrame,
    periods: int = 6,
    metric_hint: str | None = None,
) -> ForecastResult:
    date_col, metric = _detect_date_metric(df, metric_hint)
    if not date_col or not metric:
        return ForecastResult(
            success=False,
            message="Could not find a date column and numeric metric for forecasting.",
            history=pd.DataFrame(),
            forecast=pd.DataFrame(),
            growth_pct=None,
            next_value=None,
            method="none",
        )

    series = _prepare_series(df, date_col, metric)
    if len(series) < 5:
        return ForecastResult(
            success=False,
            message="Not enough historical points for forecasting (need at least 5).",
            history=series,
            forecast=pd.DataFrame(),
            growth_pct=None,
            next_value=None,
            method="none",
        )

    prophet_df = series.rename(columns={date_col: "ds", metric: "y"})

    try:
        from prophet import Prophet

        model = Prophet(daily_seasonality=False, weekly_seasonality=False)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, freq="MS")
        fc = model.predict(future)
        forecast_tail = fc.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        next_value = float(forecast_tail.iloc[0]["yhat"])
        recent = prophet_df["y"].tail(periods).mean()
        future_mean = forecast_tail["yhat"].mean()
        growth = ((future_mean - recent) / recent * 100) if recent else None
        return ForecastResult(
            success=True,
            message=f"Prophet forecast for next {periods} periods on '{metric}'.",
            history=series.rename(columns={date_col: "ds", metric: "y"}),
            forecast=forecast_tail,
            growth_pct=round(growth, 2) if growth is not None else None,
            next_value=next_value,
            method="prophet",
        )
    except Exception:
        pass

    # Linear trend fallback
    y = prophet_df["y"].values.astype(float)
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + periods)
    future_y = np.polyval(coef, future_x)
    last_date = prophet_df["ds"].max()
    future_dates = pd.date_range(last_date, periods=periods + 1, freq="MS")[1:]
    forecast_tail = pd.DataFrame({"ds": future_dates, "yhat": future_y})
    recent = y[-periods:].mean() if len(y) >= periods else y.mean()
    future_mean = future_y.mean()
    growth = ((future_mean - recent) / recent * 100) if recent else None
    return ForecastResult(
        success=True,
        message=f"Trend-based forecast for next {periods} periods on '{metric}' (Prophet unavailable).",
        history=prophet_df,
        forecast=forecast_tail,
        growth_pct=round(growth, 2) if growth is not None else None,
        next_value=float(future_y[0]),
        method="trend",
    )
