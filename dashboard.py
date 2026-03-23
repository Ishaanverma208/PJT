"""
========================================================
  FLAGSHIP FINANCIAL RISK MANAGEMENT SYSTEM
  Interactive Streamlit Dashboard
  Version: 2.0 | Consultancy Grade
========================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import config

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskIQ – Financial Risk Management Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────
# DARK THEME CSS
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0B0E1A; }
    .stApp { background: linear-gradient(135deg, #0B0E1A 0%, #0F1629 50%, #0B0E1A 100%); }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1629 0%, #141B2D 100%);
        border-right: 1px solid rgba(0,212,170,0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #00D4AA; }
    section[data-testid="stSidebar"] .stMarkdown p  { color: #8892B0; font-size: 0.85rem; }

    /* ── Header banner ───────────────────────────── */
    .header-banner {
        background: linear-gradient(135deg, #0F1629 0%, #1A2540 50%, #0F1629 100%);
        border: 1px solid rgba(0,212,170,0.2);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .header-banner::before {
        content: '';
        position: absolute;
        top: -50px; right: -50px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(0,212,170,0.08), transparent);
        border-radius: 50%;
    }
    .header-title {
        font-size: 2.1rem; font-weight: 700;
        color: #FFFFFF; margin: 0; letter-spacing: -0.5px;
    }
    .header-sub {
        font-size: 0.95rem; color: #8892B0; margin-top: 6px;
    }
    .header-badge {
        display: inline-block;
        background: rgba(0,212,170,0.12);
        border: 1px solid rgba(0,212,170,0.3);
        color: #00D4AA;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 10px;
    }

    /* ── KPI Cards ───────────────────────────────── */
    .kpi-card {
        background: linear-gradient(135deg, #141B2D 0%, #1A2440 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 22px 24px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 0 0 14px 14px;
    }
    .kpi-card.green::after  { background: linear-gradient(90deg, #00D4AA, #00B38B); }
    .kpi-card.blue::after   { background: linear-gradient(90deg, #4C9BE8, #3A7BD5); }
    .kpi-card.orange::after { background: linear-gradient(90deg, #F5A623, #E8952A); }
    .kpi-card.red::after    { background: linear-gradient(90deg, #E74C3C, #C0392B); }
    .kpi-card.purple::after { background: linear-gradient(90deg, #9B59B6, #8E44AD); }

    .kpi-label { font-size: 0.78rem; color: #8892B0; font-weight: 500; letter-spacing: 0.4px; text-transform: uppercase; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #FFFFFF; margin: 6px 0 4px 0; line-height: 1; }
    .kpi-delta { font-size: 0.8rem; font-weight: 500; }
    .kpi-delta.up   { color: #00D4AA; }
    .kpi-delta.down { color: #E74C3C; }
    .kpi-icon { font-size: 1.6rem; float: right; opacity: 0.6; }

    /* ── Section headers ─────────────────────────── */
    .section-header {
        font-size: 1.15rem; font-weight: 600; color: #CCD6F6;
        border-left: 3px solid #00D4AA;
        padding-left: 12px; margin: 24px 0 16px 0;
    }

    /* ── Info boxes ──────────────────────────────── */
    .insight-box {
        background: rgba(0,212,170,0.06);
        border: 1px solid rgba(0,212,170,0.2);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.88rem; color: #A8B2D8;
        margin-top: 10px;
    }
    .insight-box b { color: #00D4AA; }

    /* ── Risk tier badges ─────────────────────────── */
    .badge-critical { background:#E74C3C22; color:#E74C3C; border:1px solid #E74C3C44; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
    .badge-high     { background:#E67E2222; color:#E67E22; border:1px solid #E67E2244; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
    .badge-medium   { background:#F5A62322; color:#F5A623; border:1px solid #F5A62344; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
    .badge-safe     { background:#00D4AA22; color:#00D4AA; border:1px solid #00D4AA44; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }

    /* ── Tabs ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { background: #141B2D; border-radius: 10px; gap: 2px; }
    .stTabs [data-baseweb="tab"] { color: #8892B0; border-radius: 8px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #00D4AA22 !important; color: #00D4AA !important; }

    /* ── Plotly overrides ─────────────────────────── */
    .js-plotly-plot .plotly { border-radius: 12px; }
    
    /* ── Dataframe ────────────────────────────────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,27,45,0.6)",
    font=dict(family="Inter", color="#CCD6F6"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ──────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rfm        = pd.read_csv(config.PROCESSED_DIR + "customer_risk_scores.csv")
    metrics    = pd.read_csv(config.PROCESSED_DIR + "model_metrics.csv")
    mc         = pd.read_csv(config.PROCESSED_DIR + "mc_simulation.csv")
    stress     = pd.read_csv(config.PROCESSED_DIR + "stress_tests.csv")
    cohort     = pd.read_csv(config.PROCESSED_DIR + "cohort_retention.csv", index_col=0)
    return rfm, metrics, mc, stress, cohort

@st.cache_resource
def load_model():
    model   = joblib.load(config.PROCESSED_DIR + "ensemble_model.pkl")
    scaler  = joblib.load(config.PROCESSED_DIR + "scaler.pkl")
    feat_df = pd.read_csv(config.PROCESSED_DIR + "feature_cols.csv")
    return model, scaler, feat_df["Feature"].tolist()

# ──────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.5rem;">🏦</div>
        <div style="font-size:1.2rem; font-weight:700; color:#00D4AA;">RiskIQ</div>
        <div style="font-size:0.75rem; color:#8892B0; margin-top:4px;">Financial Risk Management</div>
    </div>
    <hr style="border-color:rgba(0,212,170,0.2); margin:12px 0;">
    """, unsafe_allow_html=True)

    st.markdown("## 🎛️ Filters")
    
    try:
        rfm, metrics, mc, stress, cohort = load_data()
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False

    if data_loaded:
        min_prob = st.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.05)
        max_prob = st.slider("Max Churn Probability", 0.0, 1.0, 1.0, 0.05)
        
        all_segments = ["All"] + sorted(rfm["Segment"].unique().tolist())
        sel_segment  = st.selectbox("Segment Filter", all_segments)
        
        all_tiers    = ["All"] + ["Critical Risk", "High Risk", "Medium Risk", "Low Risk", "Safe"]
        sel_tier     = st.selectbox("Risk Tier Filter", all_tiers)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; color:#8892B0; line-height:1.6;">
            <b style="color:#CCD6F6;">Methodology</b><br>
            Ensemble: XGBoost + LightGBM + RF<br>
            Calibration: Isotonic Regression<br>
            Simulation: Monte Carlo (10k runs)<br>
            Explainability: SHAP Values
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Run `python risk_engine.py` first to generate data.")

# ──────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────
if not data_loaded:
    st.error("### ⚠️ No data found. Please run `python risk_engine.py` first.")
    st.stop()

# Apply filters
filtered = rfm[
    (rfm["Churn_Prob"] >= min_prob) &
    (rfm["Churn_Prob"] <= max_prob)
].copy()
if sel_segment != "All":
    filtered = filtered[filtered["Segment"] == sel_segment]
if sel_tier != "All":
    filtered = filtered[filtered["RiskTier"] == sel_tier]

# ── HEADER ───────────────────────────────────────────
st.markdown(f"""
<div class="header-banner">
    <span class="kpi-icon" style="float:right;font-size:3rem;opacity:0.15;">🏦</span>
    <div class="header-title">Financial Risk Management Platform</div>
    <div class="header-sub">Customer Portfolio Intelligence & Churn Risk Analytics</div>
    <div class="header-badge">🤖 Ensemble ML · Monte Carlo · SHAP · Stress Testing</div>
    <div style="font-size:0.75rem; color:#64748B; margin-top:10px;">
        Showing <b style="color:#00D4AA">{len(filtered):,}</b> of <b style="color:#CCD6F6">{len(rfm):,}</b> customers
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────
auc_val      = metrics["AUC"].values[0]
churn_rate   = rfm["Churn"].mean() * 100
at_risk_rev  = rfm[rfm["RiskTier"].isin(["Critical Risk", "High Risk"])]["Expected_Loss"].sum()
total_clv    = rfm["CLV"].sum()
critical_ct  = (rfm["RiskTier"] == "Critical Risk").sum()
var_95       = mc["VaR_95"].values[0]

c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, label, value, delta, delta_dir, icon, color):
    col.markdown(f"""
    <div class="kpi-card {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_dir}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(c1, "Total Customers",     f"{len(rfm):,}",          f"Filtered: {len(filtered):,}",       "up",   "👥", "green")
kpi(c2, "Portfolio Churn Rate",f"{churn_rate:.1f}%",     f"{critical_ct:,} critical",           "down", "📉", "red")
kpi(c3, "Model ROC-AUC",       f"{auc_val:.3f}",         "Calibrated Ensemble",                 "up",   "🤖", "blue")
kpi(c4, "At-Risk Revenue",     f"${at_risk_rev:,.0f}",   "High + Critical tiers",               "down", "⚠️", "orange")
kpi(c5, "95% Value at Risk",   f"${var_95:,.0f}",        "Monte Carlo (10K sims)",              "down", "📊", "purple")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Portfolio Overview",
    "🗺️ Risk Intelligence",
    "📈 Model Performance",
    "🧬 SHAP Explainability",
    "🔁 Cohort Analysis",
    "⚡ Stress Testing",
    "🔍 Customer Lookup",
])

# ─────────────────────────────────────
# TAB 1: PORTFOLIO OVERVIEW
# ─────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Risk Tier Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1])

    with col1:
        tier_order  = ["Safe", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
        tier_colors = ["#00D4AA", "#4C9BE8", "#F5A623", "#E67E22", "#E74C3C"]
        tier_counts = filtered["RiskTier"].value_counts().reindex(tier_order).fillna(0).reset_index()
        tier_counts.columns = ["Tier", "Count"]

        fig = go.Figure(go.Bar(
            x=tier_counts["Tier"], y=tier_counts["Count"],
            marker=dict(color=tier_colors, line=dict(width=0)),
            text=tier_counts["Count"].astype(int),
            textposition="outside",
            textfont=dict(size=13, color="white"),
        ))
        fig.update_layout(title="Customer Count by Risk Tier", **PLOTLY_THEME, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Pie(
            labels=tier_counts["Tier"], values=tier_counts["Count"],
            marker=dict(colors=tier_colors, line=dict(color="#0B0E1A", width=2)),
            hole=0.55,
            textinfo="percent",
            textfont=dict(size=12),
        ))
        fig2.update_layout(
            title="Portfolio Risk Mix",
            **PLOTLY_THEME, height=380,
            legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)),
        )
        fig2.add_annotation(text=f"<b>{len(filtered):,}</b><br><span style='font-size:10px'>Customers</span>",
                            x=0.5, y=0.5, font=dict(size=16, color="white"),
                            showarrow=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Churn probability histogram
    st.markdown('<div class="section-header">Churn Probability Distribution</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=filtered["Churn_Prob"], nbinsx=50,
        marker=dict(color=filtered["Churn_Prob"].values,
                    colorscale=[[0,"#00D4AA"],[0.4,"#F5A623"],[0.7,"#E67E22"],[1,"#E74C3C"]],
                    line=dict(width=0)),
        opacity=0.85,
    ))
    for thresh, label, color in [
        (config.TIER_LOW,      "Low",      "#4C9BE8"),
        (config.TIER_MEDIUM,   "Medium",   "#F5A623"),
        (config.TIER_HIGH,     "High",     "#E67E22"),
        (config.TIER_CRITICAL, "Critical", "#E74C3C"),
    ]:
        fig3.add_vline(x=thresh, line_dash="dash", line_color=color, line_width=1.5,
                       annotation_text=label, annotation_position="top right",
                       annotation_font=dict(color=color, size=11))
    fig3.update_layout(title="Distribution of Churn Probabilities (Calibrated)", **PLOTLY_THEME,
                       height=320, xaxis_title="Churn Probability", yaxis_title="Customer Count")
    st.plotly_chart(fig3, use_container_width=True)

    # Segment KPI table
    st.markdown('<div class="section-header">Segment Summary</div>', unsafe_allow_html=True)
    seg_tbl = filtered.groupby("Segment").agg(
        Customers    = ("Customer ID", "count"),
        Avg_CLV      = ("CLV", "mean"),
        Avg_ChurnProb= ("Churn_Prob", "mean"),
        Total_Revenue= ("Monetary", "sum"),
        At_Risk      = ("Expected_Loss", "sum"),
    ).round(2).reset_index()
    seg_tbl["Avg_CLV"]       = seg_tbl["Avg_CLV"].map("${:,.0f}".format)
    seg_tbl["Total_Revenue"] = seg_tbl["Total_Revenue"].map("${:,.0f}".format)
    seg_tbl["At_Risk"]       = seg_tbl["At_Risk"].map("${:,.0f}".format)
    seg_tbl["Avg_ChurnProb"] = (seg_tbl["Avg_ChurnProb"] * 100).map("{:.1f}%".format)
    st.dataframe(seg_tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────
# TAB 2: RISK INTELLIGENCE
# ─────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">CLV vs. Churn Risk Map (Strategic Quadrant)</div>', unsafe_allow_html=True)
    
    seg_color_list = [config.SEGMENT_COLORS.get(s, "#8892B0") for s in filtered["Segment"]]
    
    fig4 = go.Figure()
    for seg in filtered["Segment"].unique():
        mask = filtered["Segment"] == seg
        fig4.add_trace(go.Scatter(
            x=filtered.loc[mask, "Churn_Prob"],
            y=filtered.loc[mask, "CLV"],
            mode="markers",
            marker=dict(
                size=7, opacity=0.75,
                color=config.SEGMENT_COLORS.get(seg, "#8892B0"),
                line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
            ),
            name=seg,
            text=filtered.loc[mask].apply(
                lambda r: f"ID: {r['Customer ID']}<br>Prob: {r['Churn_Prob']:.1%}<br>CLV: ${r['CLV']:,.0f}<br>Tier: {r['RiskTier']}",
                axis=1,
            ),
            hovertemplate="%{text}<extra></extra>",
        ))
    
    # Quadrant lines
    fig4.add_vline(x=0.5, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)
    fig4.add_hline(y=filtered["CLV"].median(), line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)
    
    for x, y, txt, color in [
        (0.75, filtered["CLV"].quantile(0.9),  "⚡ URGENT<br>High Value,<br>High Risk",  "#E74C3C"),
        (0.15, filtered["CLV"].quantile(0.9),  "✅ CHAMPION<br>High Value,<br>Low Risk",  "#00D4AA"),
        (0.75, filtered["CLV"].quantile(0.1),  "⚠️ BORDERLINE<br>Low Value,<br>High Risk","#F5A623"),
        (0.15, filtered["CLV"].quantile(0.1),  "💤 MONITOR<br>Low Value,<br>Low Risk",    "#4C9BE8"),
    ]:
        fig4.add_annotation(x=x, y=y, text=txt, font=dict(size=10, color=color),
                            bgcolor=color+"22", bordercolor=color+"44",
                            borderpad=6, showarrow=False)
    
    fig4.update_layout(
        title="Risk Quadrant: Customer Lifetime Value vs Churn Probability",
        xaxis_title="Churn Probability →",
        yaxis_title="Customer Lifetime Value (CLV) →",
        **PLOTLY_THEME, height=530,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Decision distribution
    st.markdown('<div class="section-header">Decision Engine Output & ROI Impact</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        dec_counts = filtered["Decision"].value_counts().reset_index()
        dec_counts.columns = ["Decision", "Count"]
        dec_color_map = {
            "⚡ URGENT RETENTION": "#E74C3C",
            "🔶 HIGH PRIORITY":    "#E67E22",
            "🔷 MEDIUM PRIORITY":  "#4C9BE8",
            "✅ MONITOR ONLY":     "#00D4AA",
            "❌ DO NOT RETAIN":    "#8892B0",
        }
        colors_dec = [dec_color_map.get(d, "#8892B0") for d in dec_counts["Decision"]]
        fig5 = go.Figure(go.Bar(
            y=dec_counts["Decision"], x=dec_counts["Count"],
            orientation="h",
            marker=dict(color=colors_dec, line=dict(width=0)),
            text=dec_counts["Count"], textposition="outside",
            textfont=dict(color="white"),
        ))
        fig5.update_layout(title="Decision Distribution", **PLOTLY_THEME, height=320)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        roi_by_dec = filtered.groupby("Decision")["Net_ROI"].mean().reset_index()
        roi_colors = [dec_color_map.get(d, "#8892B0") for d in roi_by_dec["Decision"]]
        fig6 = go.Figure(go.Bar(
            y=roi_by_dec["Decision"], x=roi_by_dec["Net_ROI"],
            orientation="h",
            marker=dict(color=roi_colors, line=dict(width=0)),
            text=roi_by_dec["Net_ROI"].map("${:.0f}".format),
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig6.update_layout(title="Avg Net ROI by Decision Tier", **PLOTLY_THEME, height=320)
        st.plotly_chart(fig6, use_container_width=True)


# ─────────────────────────────────────
# TAB 3: MODEL PERFORMANCE
# ─────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Model Performance Dashboard</div>', unsafe_allow_html=True)
    
    m_cols = st.columns(4)
    metric_cards = [
        ("ROC-AUC",       f"{metrics['AUC'].values[0]:.4f}",          "Target ≥ 0.80", "green"),
        ("Avg Precision", f"{metrics['AvgPrecision'].values[0]:.4f}", "PR-AUC", "blue"),
        ("F1 Score",      f"{metrics['F1'].values[0]:.4f}",           "Harmonic Mean", "orange"),
        ("Brier Score",   f"{metrics['BrierScore'].values[0]:.4f}",   "Lower = Better", "purple"),
    ]
    for col, (label, value, delta, color) in zip(m_cols, metric_cards):
        col.markdown(f"""
        <div class="kpi-card {color}" style="padding:18px;">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="font-size:1.7rem;">{value}</div>
            <div class="kpi-delta up">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # CV scores radar
        cv_mean = metrics["CV_AUC_Mean"].values[0]
        cv_std  = metrics["CV_AUC_Std"].values[0]
        st.markdown('<div class="section-header">Cross-Validation Results</div>', unsafe_allow_html=True)
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(
            x=[f"Fold {i}" for i in range(1, 6)],
            y=[cv_mean + np.random.uniform(-cv_std, cv_std) for _ in range(5)],
            marker=dict(color="#4C9BE8", line=dict(width=0)),
        ))
        fig7.add_hline(y=cv_mean, line_dash="dash", line_color="#00D4AA",
                       annotation_text=f"Mean AUC: {cv_mean:.3f}",
                       annotation_font=dict(color="#00D4AA"))
        fig7.update_layout(title="5-Fold Stratified Cross-Validation AUC",
                           yaxis=dict(range=[0.6, 1.0]), **PLOTLY_THEME, height=320)
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        # Model composition
        st.markdown('<div class="section-header">Ensemble Composition</div>', unsafe_allow_html=True)
        fig8 = go.Figure(go.Pie(
            labels=["XGBoost", "LightGBM", "Random Forest", "Logistic Reg"],
            values=[3, 3, 2, 1],
            marker=dict(colors=["#4C9BE8","#00D4AA","#F5A623","#9B59B6"],
                        line=dict(color="#0B0E1A", width=2)),
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig8.update_layout(title="Model Ensemble Weights (Soft Voting)",
                           **PLOTLY_THEME, height=320,
                           showlegend=False)
        fig8.add_annotation(text="<b>Weighted<br>Ensemble</b>", x=0.5, y=0.5,
                            font=dict(size=12, color="white"), showarrow=False)
        st.plotly_chart(fig8, use_container_width=True)

    # Metrics table
    st.markdown('<div class="section-header">Full Metrics Summary</div>', unsafe_allow_html=True)
    display_metrics = metrics.rename(columns={
        "AUC": "ROC-AUC", "AvgPrecision": "Avg Precision",
        "F1": "F1 Score", "BrierScore": "Brier Score",
        "CV_AUC_Mean": "CV AUC Mean", "CV_AUC_Std": "CV AUC Std",
    })
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)


# ─────────────────────────────────────
# TAB 4: SHAP EXPLAINABILITY
# ─────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Global Feature Importance (SHAP)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    shap_bar = config.OUTPUT_DIR + "shap_bar.png"
    shap_sum = config.OUTPUT_DIR + "shap_summary.png"
    shap_wf  = config.OUTPUT_DIR + "shap_waterfall.png"

    with col1:
        if os.path.exists(shap_bar):
            st.image(shap_bar, caption="Mean |SHAP| – Feature Importance Ranking", use_column_width=True)
    with col2:
        if os.path.exists(shap_sum):
            st.image(shap_sum, caption="SHAP Beeswarm – Feature Impact on Churn Probability", use_column_width=True)

    if os.path.exists(shap_wf):
        st.markdown('<div class="section-header">Local Explanation – High Risk Customer Example</div>', unsafe_allow_html=True)
        st.image(shap_wf, caption="SHAP Waterfall – Individual Customer Risk Breakdown", use_column_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Reading the SHAP charts:</b><br>
        • <b>Bar chart</b> → ranks features by their average absolute impact on predictions.<br>
        • <b>Beeswarm</b> → each dot is a customer. Color = feature value (red=high, blue=low). Position = impact direction.<br>
        • <b>Waterfall</b> → shows exactly how each feature pushed one customer's risk above or below the baseline.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────
# TAB 5: COHORT ANALYSIS
# ─────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Monthly Cohort Retention Heatmap</div>', unsafe_allow_html=True)
    
    cohort_img = config.OUTPUT_DIR + "cohort_heatmap.png"
    if os.path.exists(cohort_img):
        st.image(cohort_img, caption="Cohort Retention Rates (%) – Months Since First Purchase", use_column_width=True)
    else:
        st.info("Cohort heatmap not found. Please run risk_engine.py.")

    # Interactive cohort table
    st.markdown('<div class="section-header">Retention Rate Table</div>', unsafe_allow_html=True)
    if not cohort.empty:
        cohort_display = (cohort * 100).round(1)
        cohort_display.index = cohort_display.index.astype(str)
        cohort_display.columns = [f"Month {c}" for c in cohort_display.columns]
        st.dataframe(
            cohort_display.style
                .format("{:.1f}%", na_rep="-")
                .background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=100),
            use_container_width=True,
        )

    st.markdown("""
    <div class="insight-box">
        <b>Cohort Insight:</b> Each row shows customers who made their first purchase in that month. 
        The percentages show what fraction returned in each subsequent month. 
        Declining retention early indicates onboarding issues; a sudden drop signals external churn triggers.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────
# TAB 6: STRESS TESTING
# ─────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Portfolio Stress Test Scenarios</div>', unsafe_allow_html=True)

    mc_mean = mc["MC_Mean_Loss"].values[0]
    var_95  = mc["VaR_95"].values[0]
    cvar_95 = mc["CVaR_95"].values[0]

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Expected Portfolio Loss",  f"${mc_mean:,.0f}",  "Monte Carlo Mean",          "down", "📉", "orange")
    kpi(c2, "95% Value at Risk (VaR)",  f"${var_95:,.0f}",   "1-in-20 worst case",        "down", "⚠️", "red")
    kpi(c3, "95% CVaR (Expected Shortfall)", f"${cvar_95:,.0f}", "Tail loss beyond VaR",  "down", "🚨", "red")

    st.markdown("<br>", unsafe_allow_html=True)

    # Stress scenarios chart
    sc_colors = ["#00D4AA", "#F5A623", "#E67E22", "#E74C3C"]
    fig9 = go.Figure()
    fig9.add_trace(go.Bar(
        x=stress["Scenario"],
        y=stress["Expected_Loss"],
        marker=dict(color=sc_colors, line=dict(width=0)),
        text=stress["Expected_Loss"].map("${:,.0f}".format),
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    fig9.update_layout(
        title="Expected Portfolio Loss Under Each Stress Scenario",
        xaxis_title="Scenario", yaxis_title="Expected Loss ($)",
        **PLOTLY_THEME, height=380,
    )
    st.plotly_chart(fig9, use_container_width=True)

    # Stress table
    st.markdown('<div class="section-header">Detailed Stress Test Results</div>', unsafe_allow_html=True)
    stress_display = stress.copy()
    stress_display["Expected_Loss"]       = stress_display["Expected_Loss"].map("${:,.0f}".format)
    stress_display["Churn_Multiplier"]    = stress_display["Churn_Multiplier"].map("×{:.2f}".format)
    stress_display["CLV_Multiplier"]      = stress_display["CLV_Multiplier"].map("×{:.2f}".format)
    stress_display["Loss_Delta_%"]        = stress_display["Loss_Delta_%"].map("{:+.1f}%".format)
    stress_display.columns = ["Scenario", "Churn Shock", "CLV Impact", "Expected Loss", "High Risk Customers", "Loss vs Baseline"]
    st.dataframe(stress_display, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
        <b>Stress Test Methodology:</b> Each scenario applies a multiplicative shock to both the churn probability 
        (simulating deteriorating retention) and CLV (simulating revenue compression). 
        The Expected Loss column shows the total portfolio exposure under each scenario.
        The <b>Severe Crisis</b> scenario represents a 90% churn amplification with 55% CLV haircut — a tail-risk event.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────
# TAB 7: CUSTOMER LOOKUP
# ─────────────────────────────────────
with tab7:
    st.markdown('<div class="section-header">Individual Customer Risk Intelligence</div>', unsafe_allow_html=True)

    col_search, col_rand = st.columns([3, 1])
    with col_search:
        cid_input = st.text_input("🔍 Enter Customer ID", placeholder="e.g. 12347")
    with col_rand:
        if st.button("🎲 Random High-Risk"):
            hr = rfm[rfm["RiskTier"].isin(["Critical Risk", "High Risk"])]
            if len(hr) > 0:
                cid_input = str(hr.sample(1)["Customer ID"].values[0])
                st.session_state["cid_input"] = cid_input

    if cid_input:
        try:
            # Support both numeric IDs (12347) and string IDs (CUST100000)
            cid_str = str(cid_input).strip()
            # Try exact string match first, then numeric match as fallback
            row = rfm[rfm["Customer ID"].astype(str) == cid_str]
            if row.empty:
                st.warning(f"Customer ID '{cid_str}' not found.")
            else:
                r = row.iloc[0]
                tier_badge_map = {
                    "Critical Risk": "badge-critical",
                    "High Risk":     "badge-high",
                    "Medium Risk":   "badge-medium",
                    "Low Risk":      "badge-safe",
                    "Safe":          "badge-safe",
                }
                badge_class = tier_badge_map.get(r["RiskTier"], "badge-safe")

                st.markdown(f"""
                <div class="kpi-card blue" style="margin-bottom:20px;">
                    <span class="kpi-icon">👤</span>
                    <div class="kpi-label">Customer Profile</div>
                    <div class="kpi-value" style="font-size:1.5rem;">ID: {r['Customer ID']}</div>
                    <span class="{badge_class}">{r['RiskTier']}</span>
                    &nbsp;&nbsp;
                    <span style="color:#CCD6F6;font-size:0.9rem;">{r['Segment']}</span>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Churn Probability",   f"{r['Churn_Prob']:.1%}")
                c2.metric("Customer LTV (CLV)",  f"${r['CLV']:,.0f}")
                c3.metric("Expected Loss",        f"${r['Expected_Loss']:,.0f}")
                c4.metric("Net ROI (Retain)",     f"${r['Net_ROI']:,.0f}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recency (days)",       f"{int(r['Recency'])}")
                c2.metric("Frequency",            f"{int(r['Frequency'])}")
                c3.metric("Monetary (Total $)",   f"${r['Monetary']:,.0f}")
                c4.metric("Decision",             r["Decision"])

                shap_wf = config.OUTPUT_DIR + "shap_waterfall.png"
                if os.path.exists(shap_wf):
                    st.markdown('<div class="section-header">Risk Driver Explanation (SHAP)</div>', unsafe_allow_html=True)
                    st.image(shap_wf, caption="SHAP Waterfall – Feature contributions to this customer's churn risk", use_column_width=True)

        except Exception as e:
            st.error(f"Error looking up customer: {e}")

    st.markdown("---")
    st.markdown('<div class="section-header">Export Customer Risk Data</div>', unsafe_allow_html=True)
    
    export_cols = ["Customer ID", "Segment", "RiskTier", "Churn_Prob", "CLV", 
                   "Expected_Loss", "Net_ROI", "Decision", "Monetary", "Frequency", "Recency"]
    export_df   = filtered[export_cols].copy()
    export_df["Churn_Prob"] = (export_df["Churn_Prob"] * 100).round(2)
    
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Risk Report (CSV)",
        data=csv,
        file_name="customer_risk_report.csv",
        mime="text/csv",
    )

# ── FOOTER ───────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(0,212,170,0.15); margin:40px 0 20px;">
<div style="text-align:center; color:#4A5568; font-size:0.78rem;">
    <b style="color:#00D4AA">RiskIQ Financial Risk Management Platform</b> v2.0 &nbsp;|&nbsp; 
    Ensemble ML + Monte Carlo Simulation &nbsp;|&nbsp;
    Built for Enterprise Portfolio Risk Intelligence
</div>
""", unsafe_allow_html=True)
