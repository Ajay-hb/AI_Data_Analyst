"""
Agentic AI Data Analyst — upload CSV, chat with data, charts, forecasts, PDF reports.
Run: streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.forecast_agent import forecast_series
from agents.llm_client import get_llm
from agents.workflow import AnalystWorkflow
from tools.chart_tool import create_chart, suggest_chart_type
from utils.cleaner import clean_dataframe
from utils.profiler import profile_dataframe, profile_summary_text
from utils.report_pdf import build_pdf_report

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "Which region has the highest sales?",
    "Show monthly revenue trend",
    "Top 10 products by demand",
    "What is average sales by category?",
    "Predict next 6 months revenue",
]


def init_state():
    defaults = {
        "df": None,
        "profile": None,
        "workflow": None,
        "chat_history": [],
        "last_run": None,
        "clean_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_demo_csv() -> pd.DataFrame | None:
    demo = ROOT.parent / "demand_forecasting.csv"
    if demo.is_file():
        return pd.read_csv(demo, parse_dates=["Date"], infer_datetime_format=True)
    return None


def render_profile(profile: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", profile["columns"])
    c3.metric("Duplicates", profile["duplicate_rows"])
    c4.metric("Missing cols", len(profile["missing_values"]))

    st.subheader("Data quality")
    st.text(profile_summary_text(profile))

    col_l, col_r = st.columns(2)
    with col_l:
        if profile["missing_values"]:
            st.dataframe(
                pd.DataFrame(profile["missing_values"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No missing values detected.")
    with col_r:
        if profile["outliers"]:
            st.dataframe(
                pd.DataFrame(profile["outliers"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No strong outlier signals on numeric columns.")

    numeric = profile.get("numeric_columns", [])
    if len(numeric) >= 2:
        df_num = st.session_state.df[numeric]
        st.subheader("Correlation heatmap")
        st.dataframe(df_num.corr().round(2), use_container_width=True)


def render_forecast(fc):
    if not fc.success:
        st.warning(fc.message)
        return

    st.success(fc.message)
    m1, m2, m3 = st.columns(3)
    m1.metric("Method", fc.method)
    if fc.growth_pct is not None:
        m2.metric("Expected growth", f"{fc.growth_pct}%")
    if fc.next_value is not None:
        m3.metric("Next period estimate", f"{fc.next_value:,.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc.history["ds"], y=fc.history["y"], name="History", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fc.forecast["ds"], y=fc.forecast["yhat"], name="Forecast", mode="lines+markers"))
    if "yhat_lower" in fc.forecast.columns:
        fig.add_trace(go.Scatter(
            x=fc.forecast["ds"], y=fc.forecast["yhat_upper"],
            fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=fc.forecast["ds"], y=fc.forecast["yhat_lower"],
            fill="tonexty", mode="lines", line_color="rgba(0,0,0,0)",
            name="Confidence", fillcolor="rgba(31,78,121,0.15)",
        ))
    fig.update_layout(title="Forecast", height=420)
    st.plotly_chart(fig, use_container_width=True)


def main():
    init_state()

    st.title("📊 Agentic AI Data Analyst")
    st.caption("Upload a CSV · ask questions in plain English · get charts, insights, forecasts, and PDF reports.")

    with st.sidebar:
        st.header("Settings")
        llm = get_llm()
        if llm:
            st.success("LLM connected (Gemini or OpenAI)")
        else:
            st.warning(
                "No LLM API key. Set `GOOGLE_API_KEY` or `OPENAI_API_KEY` for smarter answers. "
                "Rule-based fallback is active."
            )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if st.button("Load demo dataset"):
            demo = load_demo_csv()
            if demo is not None:
                st.session_state.df = demo
                st.session_state.profile = profile_dataframe(demo)
                if st.session_state.workflow:
                    st.session_state.workflow.close()
                st.session_state.workflow = AnalystWorkflow(demo)
                st.session_state.chat_history = []
                st.session_state.last_run = None
                st.success("Loaded demand_forecasting.csv")
            else:
                st.error("Demo file not found in parent folder.")

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.session_state.profile = profile_dataframe(df)
            if st.session_state.workflow:
                st.session_state.workflow.close()
            st.session_state.workflow = AnalystWorkflow(df)
            st.session_state.chat_history = []
            st.session_state.last_run = None

        if st.session_state.df is not None:
            if st.button("Auto-clean data"):
                cleaned, report = clean_dataframe(st.session_state.df)
                st.session_state.df = cleaned
                st.session_state.profile = profile_dataframe(cleaned)
                if st.session_state.workflow:
                    st.session_state.workflow.close()
                st.session_state.workflow = AnalystWorkflow(cleaned)
                st.session_state.clean_report = report
                st.success("Cleaning complete.")

            st.download_button(
                "Export clean CSV (Power BI)",
                data=st.session_state.df.to_csv(index=False).encode(),
                file_name="clean_dataset.csv",
                mime="text/csv",
            )

        query_mode = st.radio("Query engine", ["auto", "pandas", "sql"], horizontal=True)
        forecast_periods = st.slider("Forecast periods", 3, 12, 6)

    if st.session_state.df is None:
        st.info("Upload a CSV or click **Load demo dataset** in the sidebar to begin.")
        st.markdown("### Example questions")
        for q in EXAMPLE_QUESTIONS:
            st.markdown(f"- {q}")
        return

    df = st.session_state.df
    profile = st.session_state.profile

    tab_profile, tab_chat, tab_forecast, tab_report = st.tabs(
        ["Profile", "Chat analyst", "Forecast", "PDF report"]
    )

    with tab_profile:
        render_profile(profile)
        st.subheader("Preview")
        st.dataframe(df.head(100), use_container_width=True)
        if st.session_state.clean_report:
            st.subheader("Last cleaning actions")
            for action in st.session_state.clean_report.get("actions", []):
                st.markdown(f"- {action}")

    with tab_chat:
        st.subheader("Ask your data anything")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{q}"):
                st.session_state["_pending_q"] = q

        question = st.chat_input("e.g. Which region has highest demand?")
        pending = st.session_state.pop("_pending_q", None)
        if pending:
            question = pending

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("code"):
                    st.code(msg["code"], language=msg.get("lang", "python"))
                if msg.get("dataframe") is not None:
                    st.dataframe(msg["dataframe"], use_container_width=True)
                if msg.get("chart"):
                    st.plotly_chart(msg["chart"], use_container_width=True)

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            workflow: AnalystWorkflow = st.session_state.workflow
            include_fc = any(w in question.lower() for w in ("forecast", "predict", "next"))
            with st.spinner("Agents: query → chart → insights…"):
                if query_mode != "auto":
                    run = workflow.agent.ask(question, mode=query_mode)
                    from agents.insight_agent import generate_insights
                    qdf = run.data if hasattr(run, "data") else pd.DataFrame()
                    chart = create_chart(
                        qdf if not qdf.empty else df.head(30),
                        chart_type=suggest_chart_type(question, qdf),
                        title=question,
                    )
                    insights, recs = generate_insights(profile, qdf, question, profile_summary_text(profile))
                    from agents.workflow import AnalysisRun
                    run_full = AnalysisRun(
                        question=question, profile=profile,
                        profile_text=profile_summary_text(profile),
                        query=run, chart=chart,
                        insights=insights, recommendations=recs,
                    )
                    if include_fc:
                        run_full.forecast = forecast_series(df, periods=forecast_periods)
                    st.session_state.last_run = run_full
                else:
                    run_full = workflow.run_question(
                        question, include_forecast=include_fc, forecast_periods=forecast_periods
                    )
                    st.session_state.last_run = run_full

            assistant_parts = []
            if run_full.query and run_full.query.error:
                assistant_parts.append(f"**Error:** {run_full.query.error}")
            else:
                assistant_parts.append("**Analysis complete.** See table and chart below.")
            if run_full.insights:
                assistant_parts.append("**Insights:**\n" + "\n".join(f"- {i}" for i in run_full.insights))
            if run_full.recommendations:
                assistant_parts.append("**Recommendations:**\n" + "\n".join(f"- {r}" for r in run_full.recommendations))

            entry = {
                "role": "assistant",
                "content": "\n\n".join(assistant_parts),
                "code": run_full.query.code if run_full.query else None,
                "lang": "sql" if run_full.query and run_full.query.mode == "sql" else "python",
                "dataframe": run_full.query.data if run_full.query and not run_full.query.error else None,
                "chart": run_full.chart,
            }
            st.session_state.chat_history.append(entry)
            st.rerun()

    with tab_forecast:
        st.subheader("Forecast module")
        metric_hint = st.text_input("Metric hint (optional)", placeholder="e.g. Demand, Revenue, Sales")
        if st.button("Run forecast", type="primary"):
            with st.spinner("Forecasting…"):
                fc = forecast_series(df, periods=forecast_periods, metric_hint=metric_hint or None)
            st.session_state["_last_forecast"] = fc
        fc = st.session_state.get("_last_forecast")
        if fc:
            if not fc.success:
                st.warning(fc.message)
            else:
                render_forecast(fc)

    with tab_report:
        st.subheader("Executive PDF report")
        last = st.session_state.last_run
        insights = last.insights if last else [
            "Upload data and run at least one analysis in the Chat tab to enrich this report."
        ]
        recs = last.recommendations if last else [
            "Ask questions about top performers, trends, and averages."
        ]
        fc_text = None
        if last and last.forecast and last.forecast.success:
            fc = last.forecast
            fc_text = (
                f"Method: {fc.method}. Expected growth: {fc.growth_pct}%. "
                f"Next estimate: {fc.next_value:,.2f}."
            )
        pdf = build_pdf_report(
            title="AI Data Analyst — Executive Report",
            profile=profile,
            profile_text=profile_summary_text(profile),
            insights=insights,
            recommendations=recs,
            forecast_summary=fc_text,
        )
        st.download_button(
            "Download analysis_report.pdf",
            data=pdf,
            file_name="analysis_report.pdf",
            mime="application/pdf",
        )
        st.markdown("Report includes: executive summary, dataset overview, insights, forecasts, recommendations.")


if __name__ == "__main__":
    main()
