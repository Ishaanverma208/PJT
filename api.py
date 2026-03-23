"""
========================================================
  FINANCIAL RISK MANAGEMENT PLATFORM
  FastAPI Backend  — v2.0
  Serves: REST API + Static SPA Frontend
========================================================
"""

import os
import sys
import math
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── Paths ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR    = BASE_DIR / "outputs"
STATIC_DIR    = BASE_DIR / "static"
REPORT_PATH   = OUTPUT_DIR / "risk_report.pdf"

# ── Load Artifacts ────────────────────────────────────
print("Loading model artifacts...", flush=True)

customers_df = pd.read_csv(PROCESSED_DIR / "customer_risk_scores.csv", low_memory=False)
customers_df["Customer ID"] = customers_df["Customer ID"].astype(str).str.strip()

mc_df        = pd.read_csv(PROCESSED_DIR / "mc_simulation.csv")
stress_df    = pd.read_csv(PROCESSED_DIR / "stress_tests.csv")
metrics_df   = pd.read_csv(PROCESSED_DIR / "model_metrics.csv")
shap_df      = pd.read_csv(PROCESSED_DIR / "shap_values.csv")
feature_cols = pd.read_csv(PROCESSED_DIR / "feature_cols.csv")["Feature"].tolist()

model  = joblib.load(PROCESSED_DIR / "ensemble_model.pkl")
scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")

print(f"  OK - Loaded {len(customers_df):,} customers", flush=True)

# ── FastAPI App ───────────────────────────────────────
app = FastAPI(
    title="RiskIQ Analytics API",
    description="Financial Risk Management Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helpers ───────────────────────────────────────────
def safe_float(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return 0.0
    return float(val)

def clean_row(row: dict) -> dict:
    return {k: (safe_float(v) if isinstance(v, float) else v) for k, v in row.items()}

TIER_ORDER = ["Safe", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
TIER_COLORS = {
    "Safe"          : "#22c55e",
    "Low Risk"      : "#3b82f6",
    "Medium Risk"   : "#f59e0b",
    "High Risk"     : "#f97316",
    "Critical Risk" : "#ef4444",
}


# ── Root → SPA ────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_spa():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


# ── Portfolio ─────────────────────────────────────────
@app.get("/api/portfolio")
async def get_portfolio():
    total      = len(customers_df)
    at_risk    = int(customers_df["RiskTier"].isin(["Critical Risk", "High Risk"]).sum())
    exp_loss   = safe_float(customers_df["Expected_Loss"].sum())
    churn_rate = safe_float(customers_df["Churn_Prob"].mean())
    total_clv  = safe_float(customers_df["CLV"].sum())

    mc_row  = mc_df.iloc[0].to_dict()
    met_row = metrics_df.iloc[0].to_dict()

    tier_dist = {}
    for tier in TIER_ORDER:
        cnt = int((customers_df["RiskTier"] == tier).sum())
        tier_dist[tier] = {
            "count" : cnt,
            "pct"   : round(cnt / total * 100, 1),
            "color" : TIER_COLORS[tier],
        }

    segment_dist = {}
    if "Segment" in customers_df.columns:
        for seg, grp in customers_df.groupby("Segment"):
            segment_dist[str(seg)] = {
                "count"   : int(len(grp)),
                "avg_clv" : safe_float(grp["CLV"].mean()),
            }

    return {
        "total_customers"     : total,
        "at_risk_customers"   : at_risk,
        "total_clv"           : round(total_clv, 2),
        "expected_loss"       : round(exp_loss, 2),
        "avg_churn_rate"      : round(churn_rate * 100, 1),
        "var_95"              : round(safe_float(mc_row.get("VaR_95", 0)), 2),
        "cvar_95"             : round(safe_float(mc_row.get("CVaR_95", 0)), 2),
        "model_auc"           : round(safe_float(met_row.get("AUC", 0)), 4),
        "model_f1"            : round(safe_float(met_row.get("F1", 0)), 4),
        "tier_distribution"   : tier_dist,
        "segment_distribution": segment_dist,
    }


# ── Customers ─────────────────────────────────────────
@app.get("/api/customers")
async def get_customers(
    page  : int = Query(1, ge=1),
    limit : int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    tier  : Optional[str] = None,
):
    df = customers_df.copy()
    if search:
        df = df[df["Customer ID"].str.contains(search, case=False, na=False)]
    if tier and tier != "All":
        df = df[df["RiskTier"] == tier]

    total   = len(df)
    start   = (page - 1) * limit
    page_df = df.sort_values("Churn_Prob", ascending=False).iloc[start : start + limit]

    cols    = [c for c in ["Customer ID", "RiskTier", "Churn_Prob", "CLV",
                            "Decision", "Recency", "Frequency", "Monetary", "Segment"]
               if c in page_df.columns]
    records = [clean_row(r) for r in page_df[cols].to_dict("records")]

    return {"total": total, "page": page, "limit": limit, "customers": records}


# ── Customer Detail ───────────────────────────────────
@app.get("/api/customer/{customer_id}")
async def get_customer(customer_id: str):
    row = customers_df[customers_df["Customer ID"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")

    r = clean_row(row.iloc[0].to_dict())

    shap_contrib = {}
    if not shap_df.empty:
        idx      = row.index[0] % len(shap_df)
        shap_row = shap_df.iloc[idx]
        shap_contrib = {col: round(float(shap_row[col]), 4)
                        for col in shap_df.columns if col in feature_cols}
        shap_contrib = dict(sorted(shap_contrib.items(),
                                    key=lambda x: abs(x[1]), reverse=True)[:8])
    return {"customer": r, "shap_contributions": shap_contrib}


# ── Metrics ───────────────────────────────────────────
@app.get("/api/metrics")
async def get_metrics():
    return clean_row(metrics_df.iloc[0].to_dict())


# ── Stress Tests ──────────────────────────────────────
@app.get("/api/stress-test")
async def get_stress_tests():
    return {"scenarios": [clean_row(r) for r in stress_df.to_dict("records")]}


# ── Monte Carlo ───────────────────────────────────────
@app.get("/api/monte-carlo")
async def get_monte_carlo():
    return clean_row(mc_df.iloc[0].to_dict())


# ── Live Prediction ───────────────────────────────────
class PredictRequest(BaseModel):
    recency           : float
    frequency         : float
    monetary          : float
    tenure            : float
    product_diversity : float
    avg_order_value   : float

@app.post("/api/predict")
async def predict_risk(req: PredictRequest):
    medians = customers_df[feature_cols].median().to_dict()
    fv = {col: medians.get(col, 0) for col in feature_cols}
    fv["Recency"]              = req.recency
    fv["Frequency"]            = req.frequency
    fv["Monetary"]             = req.monetary
    fv["Tenure"]               = req.tenure
    fv["ProductDiversity"]     = req.product_diversity
    fv["AvgOrderValue"]        = req.avg_order_value
    fv["CLV"]                  = req.monetary * 0.25
    fv["Spend_Last30"]         = req.monetary * 0.05
    fv["Spend_30_90"]          = req.monetary * 0.15
    fv["RecencyAcceleration"]  = (fv["Spend_Last30"] + 1) / (fv["Spend_30_90"] + 1)

    X_scaled = scaler.transform([[fv[col] for col in feature_cols]])
    prob     = float(model.predict_proba(X_scaled)[0][1])

    def tier(p):
        if p >= 0.75: return "Critical Risk"
        if p >= 0.55: return "High Risk"
        if p >= 0.35: return "Medium Risk"
        if p >= 0.20: return "Low Risk"
        return "Safe"

    clv = req.monetary * 0.25
    roi = clv * 0.90 - 500
    t   = tier(prob)

    def decision(p, r):
        if p >= 0.75 and r > 0: return "URGENT RETENTION"
        if p >= 0.55 and r > 0: return "HIGH PRIORITY"
        if p >= 0.35 and r > 0: return "MEDIUM PRIORITY"
        if p < 0.35:             return "MONITOR ONLY"
        return "DO NOT RETAIN"

    return {
        "churn_probability": round(prob * 100, 1),
        "risk_tier"        : t,
        "tier_color"       : TIER_COLORS[t],
        "expected_loss"    : round(prob * clv, 2),
        "net_roi"          : round(roi, 2),
        "decision"         : decision(prob, roi),
        "clv"              : round(clv, 2),
    }


# ── PDF Download ──────────────────────────────────────
@app.get("/api/report/download")
async def download_report():
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not found. Run `python run.py` first."
        )
    return FileResponse(
        path=str(REPORT_PATH),
        media_type="application/pdf",
        filename="RiskIQ_Risk_Report.pdf",
        headers={"Content-Disposition": "attachment; filename=RiskIQ_Risk_Report.pdf"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
