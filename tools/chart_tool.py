"""
Plotly chart generation from data and query results.
Supports:
- Line
- Bar
- Scatter
- Histogram
- Pie
- Box
- Area
- Violin
- Heatmap
- Treemap
- Sunburst
- Funnel
- Waterfall
- Bubble
- Density Heatmap
- Radar
- Geo Map
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


def _pick_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    cols = df.columns.tolist()

    if len(cols) < 2:
        return cols[0] if cols else None, None, None

    numeric = df.select_dtypes(include="number").columns.tolist()
    non_numeric = [c for c in cols if c not in numeric]

    x = non_numeric[0] if non_numeric else cols[0]
    y = numeric[0] if numeric else cols[1]
    color = non_numeric[1] if len(non_numeric) > 1 else None

    return x, y, color


def suggest_chart_type(question: str, df: pd.DataFrame) -> str:
    q = question.lower()

    if any(
        w in q
        for w in (
            "trend",
            "over time",
            "monthly",
            "daily",
            "time series",
        )
    ):
        return "line"

    if any(
        w in q
        for w in (
            "top",
            "bottom",
            "rank",
            "highest",
            "lowest",
            "sales by",
            "revenue by",
        )
    ):
        return "bar"

    if any(
        w in q
        for w in (
            "distribution",
            "histogram",
            "spread",
        )
    ):
        return "histogram"

    if any(
        w in q
        for w in (
            "correlation",
            "relationship",
            "scatter",
        )
    ):
        return "scatter"

    if any(
        w in q
        for w in (
            "share",
            "proportion",
            "market share",
        )
    ):
        return "pie"

    if any(w in q for w in ("outlier", "boxplot", "box")):
        return "box"

    if any(w in q for w in ("heatmap", "correlation matrix")):
        return "heatmap"

    if any(w in q for w in ("treemap", "category breakdown")):
        return "treemap"

    if any(w in q for w in ("sunburst", "hierarchy")):
        return "sunburst"

    if any(w in q for w in ("funnel", "conversion")):
        return "funnel"

    if any(w in q for w in ("waterfall", "profit breakdown")):
        return "waterfall"

    if "bubble" in q:
        return "bubble"

    if "density" in q:
        return "density_heatmap"

    if any(w in q for w in ("radar", "kpi comparison")):
        return "radar"

    if any(w in q for w in ("map", "location", "region map")):
        return "map"

    if len(df) <= 12:
        return "bar"

    return "line" if len(df) > 20 else "bar"


def create_chart(
    df: pd.DataFrame,
    chart_type: str | None = None,
    x: str | None = None,
    y: str | None = None,
    title: str = "Chart",
) -> go.Figure:

    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to chart")
        return fig

    work = df.copy()

    if chart_type is None:
        chart_type = "bar"

    if x is None or y is None:
        auto_x, auto_y, _ = _pick_columns(work)
        x = x or auto_x
        y = y or auto_y

    # Line
    if chart_type == "line" and x and y:
        return px.line(work, x=x, y=y, title=title)

    # Bar
    if chart_type == "bar" and x and y:
        if len(work) > 25:
            work = (
                work.nlargest(25, y)
                if y in work.columns and pd.api.types.is_numeric_dtype(work[y])
                else work.head(25)
            )
        return px.bar(work, x=x, y=y, title=title)

    # Scatter
    if chart_type == "scatter" and x and y:
        return px.scatter(work, x=x, y=y, title=title)

    # Histogram
    if chart_type == "histogram":
        col = (
            y
            or (
                work.select_dtypes(include="number").columns[0]
                if len(work.select_dtypes(include="number").columns)
                else work.columns[0]
            )
        )
        return px.histogram(work, x=col, title=title)

    # Pie
    if chart_type == "pie" and x and y:
        return px.pie(work, names=x, values=y, title=title)

    # Box
    if chart_type == "box" and x and y:
        return px.box(work, x=x, y=y, title=title)

    # Area
    if chart_type == "area" and x and y:
        return px.area(work, x=x, y=y, title=title)

    # Violin
    if chart_type == "violin" and x and y:
        return px.violin(work, x=x, y=y, box=True, title=title)

    # Heatmap
    if chart_type == "heatmap":
        numeric_df = work.select_dtypes(include="number")

        if numeric_df.empty:
            return go.Figure()

        corr = numeric_df.corr()

        return ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            showscale=True,
        )

    # Treemap
    if chart_type == "treemap" and x and y:
        return px.treemap(
            work,
            path=[x],
            values=y,
            title=title,
        )

    # Sunburst
    if chart_type == "sunburst" and x and y:
        return px.sunburst(
            work,
            path=[x],
            values=y,
            title=title,
        )

    # Funnel
    if chart_type == "funnel" and x and y:
        return px.funnel(
            work,
            x=y,
            y=x,
            title=title,
        )

    # Waterfall
    if chart_type == "waterfall" and x and y:
        fig = go.Figure(
            go.Waterfall(
                x=work[x],
                y=work[y],
            )
        )
        fig.update_layout(title=title)
        return fig

    # Bubble
    if chart_type == "bubble" and x and y:
        numeric_cols = work.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) >= 2:
            size_col = numeric_cols[-1]

            return px.scatter(
                work,
                x=x,
                y=y,
                size=size_col,
                title=title,
            )

    # Density Heatmap
    if chart_type == "density_heatmap" and x and y:
        return px.density_heatmap(
            work,
            x=x,
            y=y,
            title=title,
        )

    # Radar
    if chart_type == "radar" and x and y:
        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=work[y],
                theta=work[x],
                fill="toself",
            )
        )

        fig.update_layout(title=title)

        return fig

    # Geo Map
    if (
        chart_type == "map"
        and "Latitude" in work.columns
        and "Longitude" in work.columns
    ):
        return px.scatter_geo(
            work,
            lat="Latitude",
            lon="Longitude",
            title=title,
        )

    # Fallback
    numeric = work.select_dtypes(include="number").columns.tolist()

    if numeric:
        return px.bar(
            work.head(20),
            x=work.columns[0],
            y=numeric[0],
            title=title,
        )

    return px.bar(
        work.head(20),
        x=work.columns[0],
        y=work.columns[-1],
        title=title,
    )