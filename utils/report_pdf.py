"""PDF executive report generation."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf_report(
    title: str,
    profile: dict[str, Any],
    profile_text: str,
    insights: list[str],
    recommendations: list[str],
    forecast_summary: str | None = None,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], spaceAfter=10, textColor=colors.HexColor("#1f4e79"))
    body = styles["BodyText"]

    story = []
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Executive Summary", heading))
    story.append(Paragraph(
        "This report summarizes dataset quality, analytical findings, forecasts, and recommended actions.",
        body,
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Dataset Overview", heading))
    for line in profile_text.splitlines():
        story.append(Paragraph(line.replace("&", "&amp;"), body))
    story.append(Spacer(1, 0.15 * inch))

    if profile.get("missing_values"):
        rows = [["Column", "Missing %"]] + [
            [m["column"], f"{m['pct']}%"] for m in profile["missing_values"][:10]
        ]
        table = Table(rows, colWidths=[3 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Key Insights", heading))
    for item in insights:
        story.append(Paragraph(f"• {item}", body))
    story.append(Spacer(1, 0.15 * inch))

    if forecast_summary:
        story.append(Paragraph("Forecasts", heading))
        story.append(Paragraph(forecast_summary, body))
        story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Recommendations", heading))
    for item in recommendations:
        story.append(Paragraph(f"• {item}", body))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
