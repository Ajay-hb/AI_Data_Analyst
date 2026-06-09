# Agentic AI Data Analyst

Upload a CSV and chat with your data in plain English. Get **profiling**, **Pandas/SQL queries**, **Plotly charts**, **business insights**, **forecasts**, and **PDF reports**.

## Architecture

```text
User → Streamlit → Profiling → LLM Agent → Pandas / SQL / Charts → Insights → Forecast → PDF
```

## Quick start

```bash
cd AI_Data_Analyst
pip install -r requirements.txt
streamlit run app.py
```

Optional LLM (smarter NL → code):

```bash
set GOOGLE_API_KEY=your_key
# or
set OPENAI_API_KEY=your_key
```

Without an API key, **rule-based fallback** still answers common questions.

## Features

| Phase | Feature |
|-------|---------|
| 1 | CSV upload + demo dataset |
| 2 | Auto profiling (missing, duplicates, outliers, correlations) |
| 3 | Natural language → Pandas or SQL |
| 4 | Auto Plotly charts |
| 5 | Insight + recommendation engine |
| 6 | Prophet forecast (trend fallback if unavailable) |
| 7 | PDF executive report (ReportLab) |
| 8 | Auto data cleaning + Power BI CSV export |

## Project layout

```text
AI_Data_Analyst/
├── app.py
├── agents/
│   ├── csv_agent.py
│   ├── insight_agent.py
│   ├── forecast_agent.py
│   ├── workflow.py
│   └── llm_client.py
├── tools/
│   ├── pandas_tool.py
│   ├── sql_tool.py
│   └── chart_tool.py
├── utils/
│   ├── profiler.py
│   ├── cleaner.py
│   └── report_pdf.py
└── requirements.txt
```

## Example questions

- Which region has the highest sales?
- Show monthly revenue trend
- Top 10 customers by revenue
- Predict next 6 months demand

## Deployment

- **Render / Hugging Face Spaces:** set `streamlit run app.py` as start command
- Add `GOOGLE_API_KEY` or `OPENAI_API_KEY` as secrets

## Resume line

Built an Agentic AI Data Analyst platform enabling natural-language interaction with CSV datasets via multi-agent workflows for profiling, querying, visualization, forecasting, and PDF reporting using Python, Streamlit, LangChain, SQLite, Plotly, and Prophet.
