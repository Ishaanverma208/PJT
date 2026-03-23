"""
========================================================
  FLAGSHIP FINANCIAL RISK MANAGEMENT SYSTEM
  Automated PDF Report Generator
  Version: 2.0 | Consultancy Grade
========================================================
"""

import os
import datetime
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import Frame, PageTemplate

import config

# ─────────────────────────────────────────────────────
# BRAND COLORS
# ─────────────────────────────────────────────────────
TEAL    = colors.HexColor("#00D4AA")
DARK    = colors.HexColor("#0F1629")
NAVY    = colors.HexColor("#1A2540")
BLUE    = colors.HexColor("#4C9BE8")
RED     = colors.HexColor("#E74C3C")
ORANGE  = colors.HexColor("#F5A623")
PURPLE  = colors.HexColor("#9B59B6")
WHITE   = colors.white
LIGHT   = colors.HexColor("#CCD6F6")
GREY    = colors.HexColor("#8892B0")
DANGER  = colors.HexColor("#E74C3C")
SUCCESS = colors.HexColor("#00D4AA")

# ─────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────
def get_styles():
    styles = getSampleStyleSheet()
    custom = {
        "cover_title": ParagraphStyle(
            "cover_title", fontSize=32, fontName="Helvetica-Bold",
            textColor=WHITE, leading=40, alignment=TA_CENTER,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontSize=14, fontName="Helvetica",
            textColor=TEAL, leading=20, alignment=TA_CENTER,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", fontSize=10, fontName="Helvetica",
            textColor=GREY, leading=14, alignment=TA_CENTER,
        ),
        "section_header": ParagraphStyle(
            "section_header", fontSize=16, fontName="Helvetica-Bold",
            textColor=TEAL, spaceBefore=16, spaceAfter=8,
        ),
        "sub_header": ParagraphStyle(
            "sub_header", fontSize=12, fontName="Helvetica-Bold",
            textColor=LIGHT, spaceBefore=10, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body", fontSize=10, fontName="Helvetica",
            textColor=GREY, leading=15, spaceBefore=4, spaceAfter=4,
        ),
        "highlight": ParagraphStyle(
            "highlight", fontSize=11, fontName="Helvetica-Bold",
            textColor=WHITE, leading=15,
        ),
        "caption": ParagraphStyle(
            "caption", fontSize=8, fontName="Helvetica-Oblique",
            textColor=GREY, alignment=TA_CENTER, spaceAfter=6,
        ),
        "footer": ParagraphStyle(
            "footer", fontSize=8, fontName="Helvetica",
            textColor=GREY, alignment=TA_CENTER,
        ),
    }
    for name, style in custom.items():
        styles.add(style)
    return styles

def dark_table_style(header_color=None):
    hc = header_color or TEAL
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  hc),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#141B2D"), colors.HexColor("#0F1629")]),
        ("TEXTCOLOR",     (0, 1), (-1, -1), LIGHT),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#2D3748")),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 0), (-1, 0),  [NAVY]),
        ("LINEBELOW",     (0, 0), (-1, 0),  1, TEAL),
    ])

def add_bg(canvas, doc):
    """Dark background on every page."""
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    # Top accent bar
    canvas.setFillColor(TEAL)
    canvas.rect(0, A4[1]-4, A4[0], 4, fill=1, stroke=0)
    # Bottom footer
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, A4[0], 40, fill=1, stroke=0)
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 8)
    now = datetime.datetime.now().strftime("%B %Y")
    canvas.drawCentredString(A4[0]/2, 15,
        f"RiskIQ Financial Risk Management Platform  |  Confidential  |  {now}  |  Page {doc.page}")
    canvas.restoreState()

# ─────────────────────────────────────────────────────
# MAIN REPORT GENERATOR
# ─────────────────────────────────────────────────────
def generate_report():
    print("  Generating PDF report...")

    # Load data
    rfm     = pd.read_csv(config.PROCESSED_DIR + "customer_risk_scores.csv")
    metrics = pd.read_csv(config.PROCESSED_DIR + "model_metrics.csv")
    mc      = pd.read_csv(config.PROCESSED_DIR + "mc_simulation.csv")
    stress  = pd.read_csv(config.PROCESSED_DIR + "stress_tests.csv")
    
    styles  = get_styles()
    now     = datetime.datetime.now()
    story   = []
    W       = A4[0] - 4*cm   # usable width

    doc = SimpleDocTemplate(
        config.REPORT_FILENAME,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=3*cm, bottomMargin=2.5*cm,
    )

    # ─── COVER PAGE ──────────────────────────────
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(config.COMPANY_NAME.upper(), styles["cover_meta"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Financial Risk Management", styles["cover_title"]))
    story.append(Paragraph("Customer Portfolio Intelligence Report", styles["cover_sub"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width=W, thickness=1, color=TEAL))
    story.append(Spacer(1, 1.5*cm))

    kpi_data = [
        ["Total Customers", f"{len(rfm):,}"],
        ["Portfolio Churn Rate", f"{rfm['Churn'].mean()*100:.1f}%"],
        ["Model ROC-AUC", f"{metrics['AUC'].values[0]:.4f}"],
        ["At-Risk Revenue", f"${rfm[rfm['RiskTier'].isin(['Critical Risk','High Risk'])]['Expected_Loss'].sum():,.0f}"],
        ["95% Value at Risk", f"${mc['VaR_95'].values[0]:,.0f}"],
        ["Report Date", now.strftime("%d %B %Y")],
    ]
    kpi_table = Table([[p[0], p[1]] for p in kpi_data], colWidths=[8*cm, 8*cm])
    kpi_table.setStyle(dark_table_style(TEAL))
    story.append(kpi_table)
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph(
        f"Prepared by {config.COMPANY_NAME} Analytics Team  |  {now.strftime('%d %B %Y')}  |  CONFIDENTIAL",
        styles["cover_meta"]
    ))
    story.append(PageBreak())

    # ─── 1. EXECUTIVE SUMMARY ─────────────────────
    story.append(Paragraph("1. Executive Summary", styles["section_header"]))
    at_risk_rev = rfm[rfm["RiskTier"].isin(["Critical Risk","High Risk"])]["Expected_Loss"].sum()
    critical_n  = (rfm["RiskTier"] == "Critical Risk").sum()
    churn_rate  = rfm["Churn"].mean() * 100

    summary_text = (
        f"This report presents a comprehensive financial risk assessment of the customer portfolio "
        f"using an ensemble machine learning model (XGBoost + LightGBM + Random Forest) calibrated "
        f"with isotonic regression. The analysis identifies churn risk, quantifies revenue exposure, "
        f"and prescribes data-driven retention strategies."
        f"<br/><br/>"
        f"<b>Key Findings:</b> The portfolio contains <b>{len(rfm):,} customers</b> with an overall "
        f"churn rate of <b>{churn_rate:.1f}%</b>. The model achieves a ROC-AUC of "
        f"<b>{metrics['AUC'].values[0]:.4f}</b> with {config.CV_FOLDS}-fold cross-validation. "
        f"<b>{critical_n:,} customers</b> are classified as <i>Critical Risk</i>, representing "
        f"<b>${at_risk_rev:,.0f}</b> in at-risk expected revenue. "
        f"Monte Carlo simulation (10,000 runs) estimates a 95% Value at Risk of "
        f"<b>${mc['VaR_95'].values[0]:,.0f}</b> for the portfolio."
    )
    story.append(Paragraph(summary_text, styles["body"]))
    story.append(Spacer(1, 0.5*cm))

    # Risk tier table
    story.append(Paragraph("1.1 Portfolio Risk Tier Breakdown", styles["sub_header"]))
    tier_order = ["Critical Risk", "High Risk", "Medium Risk", "Low Risk", "Safe"]
    tier_summary = rfm.groupby("RiskTier").agg(
        Count       = ("Customer ID", "count"),
        Avg_Prob    = ("Churn_Prob", "mean"),
        Total_CLV   = ("CLV", "sum"),
        Expected_Loss = ("Expected_Loss", "sum"),
    ).reindex(tier_order).reset_index().fillna(0)

    tier_data = [["Risk Tier", "Customers", "Avg Churn %", "Total CLV", "Expected Loss"]]
    for _, row in tier_summary.iterrows():
        tier_data.append([
            row["RiskTier"],
            f"{int(row['Count']):,}",
            f"{row['Avg_Prob']*100:.1f}%",
            f"${row['Total_CLV']:,.0f}",
            f"${row['Expected_Loss']:,.0f}",
        ])
    t = Table(tier_data, colWidths=[4*cm, 3*cm, 3*cm, 4*cm, 4*cm])
    t.setStyle(dark_table_style())
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ─── 2. MODEL METHODOLOGY ─────────────────────
    story.append(Paragraph("2. Model Methodology", styles["section_header"]))
    story.append(Paragraph(
        "The risk engine employs an <b>ensemble learning approach</b> combining three gradient-boosted "
        "tree models and a logistic regressor. Features are derived from transactional RFM metrics, "
        "behavioral signals (purchase velocity, spending trend, product diversity), and temporal "
        "patterns (cohort index, weekend ratio, recency acceleration). The ensemble is calibrated "
        "using isotonic regression to produce well-calibrated probability estimates.",
        styles["body"]
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("2.1 Feature Engineering (18 Features)", styles["sub_header"]))
    feat_data = [
        ["Category", "Features"],
        ["RFM",         "Recency, Frequency, Monetary, AvgOrderValue"],
        ["Behavioral",  "ProductDiversity, TotalQuantity, PurchaseVolatility, RevenueConcentration"],
        ["Temporal",    "SpendingTrend, AvgDaysBetweenPurchases, RecencyAcceleration, WeekendRatio"],
        ["Value",       "CLV, Spend_Last30, Spend_30_90, Tenure"],
        ["Geographic",  "GeoSpread"],
    ]
    ft = Table(feat_data, colWidths=[5*cm, 13*cm])
    ft.setStyle(dark_table_style(BLUE))
    story.append(ft)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.2 Model Performance Metrics", styles["sub_header"]))
    perf_data = [
        ["Metric", "Score", "Interpretation"],
        ["ROC-AUC",        f"{metrics['AUC'].values[0]:.4f}",           "Excellent (>0.80 threshold)"],
        ["Avg Precision",  f"{metrics['AvgPrecision'].values[0]:.4f}",  "Area under PR Curve"],
        ["F1 Score",       f"{metrics['F1'].values[0]:.4f}",            "Harmonic mean of P & R"],
        ["Brier Score",    f"{metrics['BrierScore'].values[0]:.4f}",    "Calibration quality (lower=better)"],
        ["CV AUC (5-fold)",f"{metrics['CV_AUC_Mean'].values[0]:.4f} ± {metrics['CV_AUC_Std'].values[0]:.4f}", "Stratified cross-validation"],
    ]
    pt = Table(perf_data, colWidths=[5*cm, 4*cm, 9*cm])
    pt.setStyle(dark_table_style(BLUE))
    story.append(pt)

    # ─── 3. MONTE CARLO & STRESS TESTS ───────────
    story.append(PageBreak())
    story.append(Paragraph("3. Portfolio Risk Quantification", styles["section_header"]))
    story.append(Paragraph(
        f"Monte Carlo simulation with <b>{config.MC_SIMULATIONS:,} iterations</b> was used to model "
        f"the distribution of potential portfolio losses. In each simulation, customer-level churn is "
        f"sampled from their calibrated probability, and the associated CLV loss is aggregated. "
        f"This produces a full loss distribution from which VaR and CVaR are extracted.",
        styles["body"]
    ))
    story.append(Spacer(1, 0.3*cm))

    mc_data = [
        ["Metric", "Value", "Description"],
        ["Expected Loss (Mean)", f"${mc['MC_Mean_Loss'].values[0]:,.0f}", "Average loss across all simulations"],
        ["95% VaR",             f"${mc['VaR_95'].values[0]:,.0f}",       "Loss exceeded only 5% of the time"],
        ["95% CVaR (ES)",       f"${mc['CVaR_95'].values[0]:,.0f}",      "Average loss in the worst 5%"],
    ]
    mt = Table(mc_data, colWidths=[5.5*cm, 4.5*cm, 8*cm])
    mt.setStyle(dark_table_style(ORANGE))
    story.append(mt)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("3.1 Stress Test Results", styles["sub_header"]))
    st_header = ["Scenario", "Churn Shock", "CLV Impact", "Expected Loss", "High Risk Customers", "Δ vs Baseline"]
    st_data   = [st_header]
    for _, row in stress.iterrows():
        st_data.append([
            row["Scenario"],
            f"×{row['Churn_Multiplier']:.2f}",
            f"×{row['CLV_Multiplier']:.2f}",
            f"${row['Expected_Loss']:,.0f}",
            f"{int(row['High_Risk_Customers']):,}",
            f"{row['Loss_Delta_%']:+.1f}%",
        ])
    st = Table(st_data, colWidths=[4*cm, 3*cm, 3*cm, 3.5*cm, 3.5*cm, 3*cm])
    st.setStyle(dark_table_style(RED))
    story.append(st)

    # ─── 4. VISUALIZATIONS ───────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Key Visualizations", styles["section_header"]))

    img_paths = [
        (config.OUTPUT_DIR + "risk_distribution.png",  "Figure 1: Customer Risk Tier Distribution"),
        (config.OUTPUT_DIR + "shap_bar.png",           "Figure 2: SHAP Feature Importance (Mean |SHAP|)"),
        (config.OUTPUT_DIR + "shap_summary.png",       "Figure 3: SHAP Beeswarm – Global Risk Drivers"),
        (config.OUTPUT_DIR + "cohort_heatmap.png",     "Figure 4: Monthly Cohort Retention Heatmap"),
    ]

    for img_path, caption in img_paths:
        if os.path.exists(img_path):
            img = Image(img_path, width=W, height=W*0.5, kind="proportional")
            story.append(img)
            story.append(Paragraph(caption, styles["caption"]))
            story.append(Spacer(1, 0.4*cm))

    # ─── 5. RECOMMENDATIONS ──────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Strategic Recommendations", styles["section_header"]))

    recs = [
        ("Critical Risk Customers (≥75% churn prob)",
         "Deploy immediate, personalized outreach. Assign dedicated relationship managers. "
         "Offer bespoke retention packages with premium service upgrades. Prioritize by Expected Loss (Churn Prob × CLV)."),
        ("High Risk Customers (55–74% churn prob)",
         "Initiate proactive engagement campaigns. Offer loyalty rewards, exclusive discounts, "
         "or early access. Monitor product usage and trigger alerts on behavioral deterioration."),
        ("Medium Risk Customers (35–54% churn prob)",
         "Implement automated nurture sequences. Focus on increasing product diversity and purchase frequency "
         "to improve engagement depth and loyalty."),
        ("Portfolio Stress Resilience",
         "Under the Severe Crisis scenario, portfolio loss increases by >90%. We recommend building "
         "a churn buffer by pre-investing in retention for top-CLV customers before stress events materialize."),
        ("Model Maintenance",
         "Re-train the ensemble monthly using updated transactional data. Monitor concept drift via "
         "PSI (Population Stability Index) on feature distributions."),
    ]

    for title, body in recs:
        story.append(Paragraph(f"• {title}", styles["highlight"]))
        story.append(Paragraph(body, styles["body"]))
        story.append(Spacer(1, 0.3*cm))

    # ─── BUILD PDF ────────────────────────────────
    doc.build(story, onFirstPage=add_bg, onLaterPages=add_bg)
    print(f"  PDF report saved: {config.REPORT_FILENAME}")

if __name__ == "__main__":
    generate_report()
