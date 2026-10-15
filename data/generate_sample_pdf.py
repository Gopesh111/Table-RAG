"""Generates a realistic 3-page financial and technical PDF with multi-column tables using ReportLab."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_sample_report(output_path: Path) -> Path:
    """Builds a realistic 3-page corporate report containing multiple complex tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#475569"),
        leading=16,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10,
    )

    story = []

    # =========================================================================
    # PAGE 1: Financial Performance & Executive Summary
    # =========================================================================
    story.append(Paragraph("TechCorp Global Solutions — Q3 Performance Report", title_style))
    story.append(Paragraph("Comprehensive Financial, Operational, and Infrastructure Review | FY2024", subtitle_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. Executive Summary & Financial Highlights", h1_style))
    exec_summary = (
        "During the third quarter of fiscal year 2024, TechCorp achieved record revenue expansion driven "
        "by accelerated enterprise adoption of our cloud platform. Gross margins expanded across all core software segments, "
        "while operational discipline maintained operating expenditure growth below topline expansion rates. "
        "The consolidated financial indicators below reflect unaudited quarterly results reconciled against internal controls."
    )
    story.append(Paragraph(exec_summary, body_style))
    story.append(Spacer(1, 10))

    financial_data = [
        ["Financial Metric", "Q1 2024", "Q2 2024", "Q3 2024", "YoY Growth (%)"],
        ["Total Revenue ($M)", "$124.5M", "$138.2M", "$156.8M", "+26.1%"],
        ["Gross Profit ($M)", "$89.6M", "$101.4M", "$116.3M", "+29.8%"],
        ["Operating Income ($M)", "$31.2M", "$36.8M", "$44.5M", "+42.6%"],
        ["Net Income ($M)", "$24.1M", "$28.5M", "$35.2M", "+46.0%"],
        ["Operating Margin (%)", "25.1%", "26.6%", "28.4%", "+330 bps"],
        ["R&D Expenditure ($M)", "$18.4M", "$19.8M", "$22.1M", "+20.1%"],
        ["Free Cash Flow ($M)", "$21.5M", "$26.3M", "$32.8M", "+52.5%"],
    ]

    t1 = Table(financial_data, colWidths=[150, 85, 85, 85, 100])
    t1.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(t1)
    story.append(Spacer(1, 14))

    financial_analysis = (
        "Free cash flow conversion reached 73.7% of EBITDA, providing substantial capital allocation flexibility for "
        "strategic research and expansion in generative AI inference tooling. Total liquidity stood at $284.1 million "
        "in cash equivalents and short-term US Treasury securities."
    )
    story.append(Paragraph(financial_analysis, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Infrastructure Reliability & Service Level Objectives
    # =========================================================================
    story.append(Paragraph("2. Global Infrastructure Reliability & Fleet Metrics", h1_style))
    infra_summary = (
        "TechCorp operates across four primary cloud regions maintaining high-availability microservice clusters. "
        "Our reliability engineering teams monitor availability targets aligned with client Service Level Agreements (SLAs). "
        "During Q3, edge network upgrades reduced mean time to recovery (MTTR) by 34% across critical path gateways."
    )
    story.append(Paragraph(infra_summary, body_style))
    story.append(Spacer(1, 10))

    infra_data = [
        ["Region Identifier", "Target SLA", "Actual Q3 Uptime", "p99 Latency (ms)", "P1 Incidents"],
        ["US-East (N. Virginia)", "99.95%", "99.985%", "18.4 ms", "2"],
        ["US-West (Oregon)", "99.95%", "99.992%", "22.1 ms", "1"],
        ["EU-Central (Frankfurt)", "99.90%", "99.940%", "29.8 ms", "3"],
        ["AP-South (Mumbai)", "99.90%", "99.925%", "34.2 ms", "4"],
        ["AP-Northeast (Tokyo)", "99.95%", "99.970%", "26.5 ms", "1"],
    ]

    t2 = Table(infra_data, colWidths=[150, 85, 95, 95, 80])
    t2.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0fdfa"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#99f6e4")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(t2)
    story.append(Spacer(1, 14))

    infra_analysis = (
        "Zero data loss events occurred during the reporting period. The four P1 incidents in Mumbai were attributed "
        "to subsea fiber cuts impacting upstream transit tier-1 telecommunications carriers, which were safely mitigated "
        "via automated BGP route failover to Singapore within 240 seconds."
    )
    story.append(Paragraph(infra_analysis, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Storage Tiering & Cloud Cost Allocation
    # =========================================================================
    story.append(Paragraph("3. Storage Tiering Strategy & Infrastructure Cost Allocation", h1_style))
    storage_summary = (
        "To balance low-latency retrieval against long-term storage expenditures, TechCorp implements automated "
        "lifecycle policies tiering telemetry and operational logs across four distinct storage classes based on access frequency. "
        "The inventory and expenditure breakdown as of September 30, 2024 is summarized below."
    )
    story.append(Paragraph(storage_summary, body_style))
    story.append(Spacer(1, 10))

    storage_data = [
        ["Storage Tier", "Capacity (PB)", "Unit Cost ($/GB)", "Monthly Cost ($k)", "Access Frequency"],
        ["Hot NVMe Block Store", "1.2 PB", "$0.080", "$96.0k", "Continuous (Daily IOPS)"],
        ["Standard Object Storage", "8.5 PB", "$0.021", "$178.5k", "Moderate (Weekly Query)"],
        ["Infrequent Access (IA)", "14.2 PB", "$0.012", "$170.4k", "Occasional (Monthly)"],
        ["Glacier Deep Archive", "42.0 PB", "$0.001", "$42.0k", "Rare (Annual Regulatory)"],
        ["Total Fleet Footprint", "65.9 PB", "Weighted Avg", "$486.9k", "Blended Distribution"],
    ]

    t3 = Table(storage_data, colWidths=[140, 85, 90, 95, 95])
    t3.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338ca")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.HexColor("#eef2ff"), colors.white]),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e0e7ff")),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(t3)
    story.append(Spacer(1, 14))

    conclusion_text = (
        "Automated tiering migrations reduced aggregate cloud storage unit costs by 18.2% relative to Q3 FY2023. "
        "Future cost-optimization initiatives for Q4 will focus on predictive deletion of unindexed debug payloads "
        "and aggressive deduplication of staging backups."
    )
    story.append(Paragraph(conclusion_text, body_style))

    # Build the document
    doc.build(story)
    print(f"Successfully generated 3-page financial report at: {output_path}")
    return output_path


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "sample_docs" / "techcorp_q3_report.pdf"
    generate_sample_report(out_file)
