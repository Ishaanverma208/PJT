"""
========================================================
  FLAGSHIP FINANCIAL RISK MANAGEMENT SYSTEM
  Configuration & Business Parameters
  Version: 2.0 | Consultancy Grade
========================================================
"""

# ─────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────
RAW_DATA_PATH = "data/raw/online_retail_II.csv"
PROCESSED_DIR = "data/processed/"
OUTPUT_DIR    = "outputs/"

# ─────────────────────────────────────────
# BUSINESS PARAMETERS
# ─────────────────────────────────────────
RETENTION_COST_PER_CUSTOMER = 500    # ₹ / $ cost of retention intervention
AVG_PROFIT_MARGIN           = 0.25   # 25% average margin on revenue
DISCOUNT_RATE               = 0.10   # 10% discount offered in retention campaign
CHURN_WINDOW_DAYS           = 90     # Customers inactive >90 days → churned
CLV_PROJECTION_MONTHS       = 12     # CLV horizon in months

# ─────────────────────────────────────────
# RISK TIER THRESHOLDS (Churn Probability)
# ─────────────────────────────────────────
TIER_CRITICAL    = 0.75   # >= 75%  → Critical Risk
TIER_HIGH        = 0.55   # >= 55%  → High Risk
TIER_MEDIUM      = 0.35   # >= 35%  → Medium Risk
TIER_LOW         = 0.20   # >= 20%  → Low Risk
                           # <  20%  → Safe

# ─────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CV_FOLDS        = 5

XGBOOST_PARAMS = {
    "n_estimators"   : 300,
    "max_depth"      : 5,
    "learning_rate"  : 0.05,
    "subsample"      : 0.8,
    "colsample_bytree": 0.8,
    "random_state"   : RANDOM_STATE,
}

LIGHTGBM_PARAMS = {
    "n_estimators"   : 300,
    "max_depth"      : 5,
    "learning_rate"  : 0.05,
    "subsample"      : 0.8,
    "colsample_bytree": 0.8,
    "random_state"   : RANDOM_STATE,
    "verbose"        : -1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators"   : 200,
    "max_depth"      : 6,
    "min_samples_leaf": 5,
    "random_state"   : RANDOM_STATE,
    "n_jobs"         : -1,
}

# ─────────────────────────────────────────
# CUSTOMER SEGMENTATION (K-Means)
# ─────────────────────────────────────────
N_SEGMENTS = 5
SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At-Risk",
    3: "Hibernating",
    4: "Lost",
}
SEGMENT_COLORS = {
    "Champions"       : "#00D4AA",
    "Loyal Customers" : "#4C9BE8",
    "At-Risk"         : "#F5A623",
    "Hibernating"     : "#9B59B6",
    "Lost"            : "#E74C3C",
}

# ─────────────────────────────────────────
# MONTE CARLO SIMULATION
# ─────────────────────────────────────────
MC_SIMULATIONS  = 10000
MC_CONFIDENCE   = 0.95   # 95% VaR

# ─────────────────────────────────────────
# STRESS TEST SCENARIOS
# ─────────────────────────────────────────
STRESS_SCENARIOS = {
    "Baseline"      : {"churn_multiplier": 1.0,  "clv_multiplier": 1.0},
    "Mild Recession": {"churn_multiplier": 1.25, "clv_multiplier": 0.85},
    "Market Shock"  : {"churn_multiplier": 1.55, "clv_multiplier": 0.65},
    "Severe Crisis" : {"churn_multiplier": 1.90, "clv_multiplier": 0.45},
}

# ─────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────
DASHBOARD_TITLE     = "Financial Risk Management Platform"
DASHBOARD_SUBTITLE  = "Customer Portfolio Intelligence & Churn Risk Analytics"
COMPANY_NAME        = "RiskIQ Analytics"
REPORT_FILENAME     = "outputs/risk_report.pdf"
