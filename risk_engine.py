"""
========================================================
  FLAGSHIP FINANCIAL RISK MANAGEMENT SYSTEM
  Advanced ML Risk Engine
  Version: 2.0 | Consultancy Grade
  
  Features:
  - 18+ engineered features (RFM + behavioral + temporal)
  - Ensemble model: XGBoost + LightGBM + Random Forest
  - Calibrated probabilities (Platt scaling)
  - Stratified K-Fold cross-validation
  - K-Means customer segmentation (5 tiers)
  - Cohort retention analysis
  - Monte Carlo portfolio simulation (95% VaR)
  - 4-tier stress testing
  - SHAP explainability (global + local)
  - Decision engine with ROI optimization
========================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.cluster import KMeans

from xgboost import XGBClassifier
import lightgbm as lgb

import config

warnings.filterwarnings("ignore")
np.random.seed(config.RANDOM_STATE)

# ──────────────────────────────────────────────────────
# SETUP OUTPUT DIRECTORIES
# ──────────────────────────────────────────────────────
os.makedirs(config.PROCESSED_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  FINANCIAL RISK MANAGEMENT ENGINE v2.0")
print("  Powered by Ensemble ML + Monte Carlo Simulation")
print("=" * 65)

# ──────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ──────────────────────────────────────────────────────
print("\n[1/9] Loading and cleaning data...")

df = pd.read_csv(config.RAW_DATA_PATH, encoding="utf-8", low_memory=False)
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Remove cancellations and invalid rows
df = df.dropna(subset=["Customer ID"])
# Keep Customer ID as-is (supports both numeric and string IDs like CUST12345)
df["Customer ID"] = df["Customer ID"].astype(str).str.strip()
df = df[~df["Invoice"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])
df["TotalAmount"] = df["Quantity"] * df["Price"]

print(f"  After cleaning: {df.shape[0]:,} rows | {df['Customer ID'].nunique():,} unique customers")

# ──────────────────────────────────────────────────────
# 2. ADVANCED FEATURE ENGINEERING (18+ Features)
# ──────────────────────────────────────────────────────
print("\n[2/9] Engineering advanced features...")

ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Base RFM
rfm = df.groupby("Customer ID").agg(
    Recency    = ("InvoiceDate", lambda x: (ref_date - x.max()).days),
    Frequency  = ("Invoice",     "nunique"),
    Monetary   = ("TotalAmount", "sum"),
).reset_index()

# Average Order Value
rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

# Unique products purchased
product_div = df.groupby("Customer ID")["StockCode"].nunique().reset_index()
product_div.columns = ["Customer ID", "ProductDiversity"]
rfm = rfm.merge(product_div, on="Customer ID", how="left")

# Total items purchased
total_qty = df.groupby("Customer ID")["Quantity"].sum().reset_index()
total_qty.columns = ["Customer ID", "TotalQuantity"]
rfm = rfm.merge(total_qty, on="Customer ID", how="left")

# Unique countries (geographic spread) — optional column
if "Country" in df.columns:
    geo = df.groupby("Customer ID")["Country"].nunique().reset_index()
    geo.columns = ["Customer ID", "GeoSpread"]
    rfm = rfm.merge(geo, on="Customer ID", how="left")
else:
    rfm["GeoSpread"] = 1

# Purchase volatility (std of order values)
vol = df.groupby(["Customer ID", "Invoice"])["TotalAmount"].sum().reset_index()
vol_std = vol.groupby("Customer ID")["TotalAmount"].std().reset_index()
vol_std.columns = ["Customer ID", "PurchaseVolatility"]
rfm = rfm.merge(vol_std, on="Customer ID", how="left")

# Avg days between purchases
def avg_days_between(dates):
    if len(dates) < 2:
        return np.nan
    s = sorted(dates)
    diffs = [(s[i+1] - s[i]).days for i in range(len(s)-1)]
    return np.mean(diffs)

purchase_cadence = (
    df.groupby("Customer ID")["InvoiceDate"]
    .apply(avg_days_between)
    .reset_index()
)
purchase_cadence.columns = ["Customer ID", "AvgDaysBetweenPurchases"]
rfm = rfm.merge(purchase_cadence, on="Customer ID", how="left")

# Spending trend (slope of monthly spend)
monthly = (
    df.set_index("InvoiceDate")
    .groupby("Customer ID")["TotalAmount"]
    .resample("ME").sum()
    .reset_index()
)
monthly.columns = ["Customer ID", "Month", "MonthlySpend"]

def spending_trend(grp):
    if len(grp) < 3:
        return 0.0
    x = np.arange(len(grp))
    slope, _, _, _, _ = stats.linregress(x, grp["MonthlySpend"].values)
    return slope

trend = monthly.groupby("Customer ID").apply(spending_trend).reset_index()
trend.columns = ["Customer ID", "SpendingTrend"]
rfm = rfm.merge(trend, on="Customer ID", how="left")

# Revenue concentration (Herfindahl index – how concentrated spend is in few orders)
def herfindahl(grp):
    total = grp["TotalAmount"].sum()
    if total == 0:
        return 1.0
    shares = (grp["TotalAmount"] / total) ** 2
    return shares.sum()

hh = vol.groupby("Customer ID").apply(herfindahl).reset_index()
hh.columns = ["Customer ID", "RevenueConcentration"]
rfm = rfm.merge(hh, on="Customer ID", how="left")

# Recency acceleration (last 30 days vs prior 30-90 days spend ratio)
cutoff_30  = ref_date - pd.Timedelta(days=30)
cutoff_90  = ref_date - pd.Timedelta(days=90)

spend_30   = df[df["InvoiceDate"] >= cutoff_30].groupby("Customer ID")["TotalAmount"].sum().reset_index()
spend_30.columns = ["Customer ID", "Spend_Last30"]
spend_3090 = df[(df["InvoiceDate"] >= cutoff_90) & (df["InvoiceDate"] < cutoff_30)].groupby("Customer ID")["TotalAmount"].sum().reset_index()
spend_3090.columns = ["Customer ID", "Spend_30_90"]

rfm = rfm.merge(spend_30, on="Customer ID", how="left")
rfm = rfm.merge(spend_3090, on="Customer ID", how="left")
rfm["RecencyAcceleration"] = (rfm["Spend_Last30"] + 1) / (rfm["Spend_30_90"] + 1)

# Weekend purchase ratio
if "InvoiceDate" in df.columns:
    df["IsWeekend"] = df["InvoiceDate"].dt.dayofweek >= 5
    weekend_ratio = df.groupby("Customer ID")["IsWeekend"].mean().reset_index()
    weekend_ratio.columns = ["Customer ID", "WeekendRatio"]
    rfm = rfm.merge(weekend_ratio, on="Customer ID", how="left")
else:
    rfm["WeekendRatio"] = 0.5

# Customer tenure (days since first purchase)
tenure = df.groupby("Customer ID")["InvoiceDate"].min().reset_index()
tenure.columns = ["Customer ID", "FirstPurchase"]
tenure["Tenure"] = (ref_date - tenure["FirstPurchase"]).dt.days
rfm = rfm.merge(tenure[["Customer ID", "Tenure"]], on="Customer ID", how="left")

# ──────────────────────────────────────────────────────
# 3. TARGET VARIABLE & CLV
# ──────────────────────────────────────────────────────
# Auto-calibrate churn window if the fixed threshold yields <5% or >95% churn
# This handles both historical datasets and recent/synthetic datasets gracefully
_fixed_churn_rate = (rfm["Recency"] > config.CHURN_WINDOW_DAYS).mean()
if _fixed_churn_rate < 0.05 or _fixed_churn_rate > 0.95:
    # Use 70th percentile of recency as threshold → ~30% churn rate  
    CHURN_THRESHOLD = rfm["Recency"].quantile(0.70)
    print(f"  ⚠️  Auto-calibrating churn window: fixed {config.CHURN_WINDOW_DAYS}d gave {_fixed_churn_rate:.1%} churn.")
    print(f"     Using 70th-percentile recency threshold: {CHURN_THRESHOLD:.0f} days")
else:
    CHURN_THRESHOLD = config.CHURN_WINDOW_DAYS

rfm["Churn"] = (rfm["Recency"] > CHURN_THRESHOLD).astype(int)
rfm["CLV"]   = rfm["Monetary"] * config.AVG_PROFIT_MARGIN

# Fill NaN
rfm = rfm.fillna(0)
rfm.replace([np.inf, -np.inf], 0, inplace=True)

print(f"  Features engineered: {rfm.shape[1]} columns")
print(f"  Churn rate: {rfm['Churn'].mean()*100:.1f}%  |  Customers: {len(rfm):,}")

# ──────────────────────────────────────────────────────
# 4. CUSTOMER SEGMENTATION (K-Means)
# ──────────────────────────────────────────────────────
print("\n[3/9] Running customer segmentation...")

seg_features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "Tenure"]
X_seg = rfm[seg_features].copy()
scaler_seg = StandardScaler()
X_seg_scaled = scaler_seg.fit_transform(X_seg)

kmeans = KMeans(n_clusters=config.N_SEGMENTS, random_state=config.RANDOM_STATE, n_init=20)
rfm["SegmentID"] = kmeans.fit_predict(X_seg_scaled)

# Auto-label segments by CLV rank
seg_clv = rfm.groupby("SegmentID")["CLV"].mean().sort_values(ascending=False)
rank_map = {seg_id: list(config.SEGMENT_LABELS.values())[i] for i, seg_id in enumerate(seg_clv.index)}
rfm["Segment"] = rfm["SegmentID"].map(rank_map)

seg_summary = rfm.groupby("Segment").agg(
    Count      = ("Customer ID", "count"),
    AvgCLV     = ("CLV", "mean"),
    AvgRecency = ("Recency", "mean"),
    ChurnRate  = ("Churn", "mean"),
).round(2)
print(seg_summary.to_string())

# ──────────────────────────────────────────────────────
# 5. COHORT ANALYSIS
# ──────────────────────────────────────────────────────
print("\n[4/9] Building cohort analysis...")

df_cohort = df.merge(rfm[["Customer ID"]], on="Customer ID")
df_cohort["CohortMonth"] = df_cohort.groupby("Customer ID")["InvoiceDate"].transform("min").dt.to_period("M")
df_cohort["PurchaseMonth"] = df_cohort["InvoiceDate"].dt.to_period("M")
df_cohort["CohortIndex"] = (
    (df_cohort["PurchaseMonth"] - df_cohort["CohortMonth"])
    .apply(lambda x: x.n if hasattr(x, 'n') else 0)
)

cohort_data = (
    df_cohort.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
    .nunique()
    .reset_index()
)
cohort_pivot = cohort_data.pivot_table(index="CohortMonth", columns="CohortIndex", values="Customer ID")
cohort_size  = cohort_pivot.iloc[:, 0]
cohort_pct   = cohort_pivot.divide(cohort_size, axis=0).round(3)
cohort_pct   = cohort_pct.iloc[:12, :12]   # Keep top 12 months

# Save cohort
cohort_pct.to_csv(config.PROCESSED_DIR + "cohort_retention.csv")

# ──────────────────────────────────────────────────────
# 6. ENSEMBLE ML MODEL TRAINING
# ──────────────────────────────────────────────────────
print("\n[5/9] Training ensemble model (XGBoost + LightGBM + Random Forest)...")

FEATURE_COLS = [
    "Frequency", "Monetary", "AvgOrderValue", "ProductDiversity",
    "TotalQuantity", "GeoSpread", "PurchaseVolatility", "AvgDaysBetweenPurchases",
    "SpendingTrend", "RevenueConcentration", "RecencyAcceleration",
    "WeekendRatio", "Tenure", "Spend_Last30", "Spend_30_90", "CLV",
]

X = rfm[FEATURE_COLS].copy()
y = rfm["Churn"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)

scaler_model = StandardScaler()
X_train_s = scaler_model.fit_transform(X_train)
X_test_s  = scaler_model.transform(X_test)
X_all_s   = scaler_model.transform(X)

# Base models
xgb_base = XGBClassifier(**config.XGBOOST_PARAMS)
lgb_base = lgb.LGBMClassifier(**config.LIGHTGBM_PARAMS)
rf_base  = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
lr_base  = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, C=0.5)

# Ensemble (soft voting)
ensemble = VotingClassifier(
    estimators=[
        ("xgb", xgb_base),
        ("lgb", lgb_base),
        ("rf",  rf_base),
        ("lr",  lr_base),
    ],
    voting="soft",
    weights=[3, 3, 2, 1],   # XGB & LGB get more weight
)

# Calibrate for reliable probabilities
calibrated_model = CalibratedClassifierCV(ensemble, method="isotonic", cv=3)
calibrated_model.fit(X_train_s, y_train)

y_pred  = calibrated_model.predict(X_test_s)
y_prob  = calibrated_model.predict_proba(X_test_s)[:, 1]

# ──────────────────────────────────────────────────────
# 7. MODEL EVALUATION
# ──────────────────────────────────────────────────────
print("\n[6/9] Evaluating model performance...")

auc    = roc_auc_score(y_test, y_prob)
ap     = average_precision_score(y_test, y_prob)
f1     = f1_score(y_test, y_pred)
prec   = precision_score(y_test, y_pred)
rec    = recall_score(y_test, y_pred)
brier  = brier_score_loss(y_test, y_prob)

print(f"\n  ┌─────────────────────────────┐")
print(f"  │  MODEL PERFORMANCE METRICS  │")
print(f"  ├─────────────────────────────┤")
print(f"  │  ROC-AUC Score  : {auc:.4f}    │")
print(f"  │  Avg Precision  : {ap:.4f}    │")
print(f"  │  F1 Score       : {f1:.4f}    │")
print(f"  │  Precision      : {prec:.4f}    │")
print(f"  │  Recall         : {rec:.4f}    │")
print(f"  │  Brier Score    : {brier:.4f}    │")
print(f"  └─────────────────────────────┘\n")
print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))

# Cross-validation
cv    = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_all_s, y), 1):
    m = CalibratedClassifierCV(
        VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(**config.XGBOOST_PARAMS)),
                ("lgb", lgb.LGBMClassifier(**config.LIGHTGBM_PARAMS)),
                ("rf",  RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)),
            ],
            voting="soft", weights=[3, 3, 2],
        ),
        method="isotonic", cv=2
    )
    m.fit(X_all_s[tr_idx], y.iloc[tr_idx])
    p = m.predict_proba(X_all_s[val_idx])[:, 1]
    cv_scores.append(roc_auc_score(y.iloc[val_idx], p))

print(f"  Cross-Validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Save metrics
metrics = {
    "AUC": round(auc, 4), "AvgPrecision": round(ap, 4),
    "F1": round(f1, 4), "Precision": round(prec, 4),
    "Recall": round(rec, 4), "BrierScore": round(brier, 4),
    "CV_AUC_Mean": round(np.mean(cv_scores), 4),
    "CV_AUC_Std": round(np.std(cv_scores), 4),
}
pd.DataFrame([metrics]).to_csv(config.PROCESSED_DIR + "model_metrics.csv", index=False)

# ──────────────────────────────────────────────────────
# 8. RISK SCORING & DECISION ENGINE
# ──────────────────────────────────────────────────────
print("\n[7/9] Running risk scoring & decision engine...")

churn_prob = calibrated_model.predict_proba(X_all_s)[:, 1]
rfm["Churn_Prob"] = churn_prob

# Risk tier assignment
def assign_tier(p):
    if p >= config.TIER_CRITICAL : return "Critical Risk"
    elif p >= config.TIER_HIGH   : return "High Risk"
    elif p >= config.TIER_MEDIUM : return "Medium Risk"
    elif p >= config.TIER_LOW    : return "Low Risk"
    else                         : return "Safe"

rfm["RiskTier"] = rfm["Churn_Prob"].apply(assign_tier)

# Financial impact metrics
rfm["Expected_Loss"]    = rfm["Churn_Prob"] * rfm["CLV"]
rfm["Retention_Gain"]   = rfm["CLV"] * (1 - config.DISCOUNT_RATE)
rfm["Net_ROI"]          = rfm["Retention_Gain"] - config.RETENTION_COST_PER_CUSTOMER
rfm["Intervention_ROI"] = rfm["Net_ROI"] * rfm["Churn_Prob"]

# Decision policy (5-tier)
def decision_policy(row):
    p   = row["Churn_Prob"]
    roi = row["Net_ROI"]
    if p >= config.TIER_CRITICAL and roi > 0:
        return "⚡ URGENT RETENTION"
    elif p >= config.TIER_HIGH and roi > 0:
        return "🔶 HIGH PRIORITY"
    elif p >= config.TIER_MEDIUM and roi > 0:
        return "🔷 MEDIUM PRIORITY"
    elif p < config.TIER_MEDIUM:
        return "✅ MONITOR ONLY"
    else:
        return "❌ DO NOT RETAIN"

rfm["Decision"] = rfm.apply(decision_policy, axis=1)

print("\n  Decision Distribution:")
print(rfm["Decision"].value_counts().to_string())
print(f"\n  Total At-Risk Revenue: ${rfm[rfm['RiskTier'].isin(['Critical Risk','High Risk'])]['Expected_Loss'].sum():,.0f}")

# ──────────────────────────────────────────────────────
# 9. MONTE CARLO PORTFOLIO SIMULATION
# ──────────────────────────────────────────────────────
print("\n[8/9] Running Monte Carlo simulation (VaR)...")

mc_results = []
n_customers = len(rfm)
probs       = rfm["Churn_Prob"].values
clvs        = rfm["CLV"].values

for _ in range(config.MC_SIMULATIONS):
    churns       = np.random.binomial(1, probs)
    portfolio_loss = np.sum(churns * clvs)
    mc_results.append(portfolio_loss)

mc_results = np.array(mc_results)
var_95     = np.percentile(mc_results, 95)
cvar_95    = mc_results[mc_results >= var_95].mean()
mc_mean    = mc_results.mean()

print(f"  Monte Carlo ({config.MC_SIMULATIONS:,} simulations):")
print(f"  Expected Portfolio Loss: ${mc_mean:,.0f}")
print(f"  95% VaR (Value at Risk): ${var_95:,.0f}")
print(f"  95% CVaR (Tail Risk)   : ${cvar_95:,.0f}")

mc_stats = {
    "MC_Mean_Loss": round(mc_mean, 2),
    "VaR_95":       round(var_95, 2),
    "CVaR_95":      round(cvar_95, 2),
    "MC_Simulations": config.MC_SIMULATIONS,
}
pd.DataFrame([mc_stats]).to_csv(config.PROCESSED_DIR + "mc_simulation.csv", index=False)

# ──────────────────────────────────────────────────────
# STRESS TESTING
# ──────────────────────────────────────────────────────
stress_results = []
for scenario, params in config.STRESS_SCENARIOS.items():
    adj_probs  = np.clip(probs * params["churn_multiplier"], 0, 1)
    adj_clvs   = clvs * params["clv_multiplier"]
    loss       = np.sum(adj_probs * adj_clvs)
    high_risk  = np.sum(adj_probs >= config.TIER_HIGH)
    stress_results.append({
        "Scenario"          : scenario,
        "Churn_Multiplier"  : params["churn_multiplier"],
        "CLV_Multiplier"    : params["clv_multiplier"],
        "Expected_Loss"     : round(loss, 0),
        "High_Risk_Customers": high_risk,
        "Loss_Delta_%"      : round((loss / mc_mean - 1) * 100, 1),
    })

stress_df = pd.DataFrame(stress_results)
stress_df.to_csv(config.PROCESSED_DIR + "stress_tests.csv", index=False)
print(f"\n  Stress Test Results:")
print(stress_df[["Scenario", "Expected_Loss", "High_Risk_Customers", "Loss_Delta_%"]].to_string(index=False))

# ──────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ──────────────────────────────────────────────────────
print("\n[9/9] Generating SHAP explainability plots...")

# Use XGB base model for SHAP (most interpretable)
xgb_shap = XGBClassifier(**config.XGBOOST_PARAMS)
xgb_shap.fit(X_train_s, y_train)

explainer   = shap.Explainer(xgb_shap, X_train_s)
shap_values = explainer(X_test_s)

shap_df = pd.DataFrame(shap_values.values, columns=FEATURE_COLS)
shap_df.to_csv(config.PROCESSED_DIR + "shap_values.csv", index=False)

# Global SHAP summary
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, show=False, plot_size=(10, 7))
plt.title("SHAP Feature Importance – Global Risk Drivers", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(config.OUTPUT_DIR + "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# SHAP bar plot (mean |SHAP|)
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, plot_type="bar", show=False)
plt.title("Mean |SHAP| Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(config.OUTPUT_DIR + "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# SHAP waterfall for single high-risk customer
hr_idx = np.where(y_prob > 0.75)[0]
if len(hr_idx) > 0:
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[hr_idx[0]], show=False)
    plt.title("SHAP Waterfall – High Risk Customer Example", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR + "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── Risk Distribution Chart ──────────────────────────
plt.figure(figsize=(10, 5))
colors = ["#00D4AA", "#4C9BE8", "#F5A623", "#E67E22", "#E74C3C"]
tier_order = ["Safe", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
tier_counts = rfm["RiskTier"].value_counts().reindex(tier_order).fillna(0)
bars = plt.bar(tier_order, tier_counts.values, color=colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, tier_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{int(val):,}",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.xlabel("Risk Tier", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.title("Customer Portfolio Risk Distribution", fontsize=14, fontweight="bold")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(config.OUTPUT_DIR + "risk_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Cohort Heatmap ────────────────────────────────────
plt.figure(figsize=(14, 7))
mask = cohort_pct.isnull()
sns.heatmap(
    cohort_pct * 100,
    annot=True, fmt=".0f", linewidths=0.5,
    cmap="RdYlGn", mask=mask,
    cbar_kws={"label": "Retention Rate (%)"},
    annot_kws={"size": 8},
)
plt.title("Cohort Retention Analysis – Monthly Retention Rates (%)", fontsize=13, fontweight="bold")
plt.xlabel("Months Since First Purchase", fontsize=11)
plt.ylabel("Cohort (Month of First Purchase)", fontsize=11)
plt.tight_layout()
plt.savefig(config.OUTPUT_DIR + "cohort_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ──────────────────────────────────────────────────────
# SAVE MASTER DATASET & MODEL
# ──────────────────────────────────────────────────────
rfm.to_csv(config.PROCESSED_DIR + "customer_risk_scores.csv", index=False)
joblib.dump(calibrated_model, config.PROCESSED_DIR + "ensemble_model.pkl")
joblib.dump(scaler_model,     config.PROCESSED_DIR + "scaler.pkl")
pd.DataFrame(FEATURE_COLS, columns=["Feature"]).to_csv(config.PROCESSED_DIR + "feature_cols.csv", index=False)

print("\n" + "=" * 65)
print("  ENGINE COMPLETE — All outputs saved!")
print(f"  Master dataset : {config.PROCESSED_DIR}customer_risk_scores.csv")
print(f"  Model          : {config.PROCESSED_DIR}ensemble_model.pkl")
print(f"  Visuals        : {config.OUTPUT_DIR}")
print("=" * 65)
