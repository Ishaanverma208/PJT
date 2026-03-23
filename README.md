# 🏦 Financial Risk Management Platform
### Customer Portfolio Intelligence & Churn Risk Analytics
> **Enterprise-grade ML pipeline** | Built for top-tier consultancy presentations

---

## 🎯 Overview

A flagship financial risk management system that combines **ensemble machine learning**, **Monte Carlo simulation**, **SHAP explainability**, and an **interactive Streamlit dashboard** to deliver actionable customer churn risk intelligence.

**Key Highlights:**
- 🤖 **Ensemble Model** — XGBoost + LightGBM + Random Forest (calibrated with isotonic regression)
- 📊 **18+ Engineered Features** — RFM + behavioral velocity + temporal patterns
- 📉 **Monte Carlo VaR** — 10,000 simulations for 95% portfolio Value at Risk
- 🔬 **SHAP Explainability** — Global beeswarm + local waterfall for any customer
- 🌀 **Stress Testing** — 4 macro scenarios (Baseline → Severe Crisis)
- 📄 **Auto-generated PDF Report** — Consultancy-grade with executive summary
- 🖥️ **Interactive Dashboard** — 7-tab dark-themed Streamlit app

---

## 🗂️ Project Structure

```
PJT/
├── config.py            # Business parameters & model configuration
├── risk_engine.py       # Full ML pipeline (features → model → outputs)
├── dashboard.py         # Interactive Streamlit dashboard
├── report_generator.py  # Auto PDF report (ReportLab)
├── run.py               # Orchestrator (runs everything)
├── requirements.txt
├── data/
│   ├── raw/             # online_retail_II.csv
│   └── processed/       # Model outputs, risk scores, metrics
└── outputs/             # Charts, SHAP plots, PDF report
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Engine + Report)
```bash
python run.py
```

### 3. Launch Interactive Dashboard
```bash
streamlit run dashboard.py
```
Open `http://localhost:8501` in your browser.

---

## 🧠 Methodology

### Feature Engineering (18 Features)
| Category | Features |
|---|---|
| **RFM** | Recency, Frequency, Monetary, AvgOrderValue |
| **Behavioral** | ProductDiversity, PurchaseVolatility, RevenueConcentration |
| **Temporal** | SpendingTrend, RecencyAcceleration, AvgDaysBetweenPurchases |
| **Value** | CLV, Spend_Last30, Spend_30_90, Tenure |
| **Geographic** | GeoSpread, WeekendRatio |

### Ensemble Architecture
```
XGBoost (weight=3)
LightGBM (weight=3)    →  Soft Voting  →  Isotonic Calibration  →  Churn Probability
Random Forest (weight=2)
Logistic Regression (weight=1)
```

### Risk Tier Thresholds
| Tier | Churn Probability | Action |
|---|---|---|
| ⚡ Critical Risk | ≥ 75% | Urgent Retention |
| 🔶 High Risk | 55–74% | High Priority |
| 🔷 Medium Risk | 35–54% | Medium Priority |
| 🔵 Low Risk | 20–34% | Monitor |
| ✅ Safe | < 20% | No Action |

### Monte Carlo VaR
- **10,000 simulations** sampling customer churn from calibrated probabilities
- Reports: **Expected Loss**, **95% VaR**, **95% CVaR (Expected Shortfall)**

### Stress Testing
| Scenario | Churn Shock | CLV Impact |
|---|---|---|
| Baseline | ×1.00 | ×1.00 |
| Mild Recession | ×1.25 | ×0.85 |
| Market Shock | ×1.55 | ×0.65 |
| Severe Crisis | ×1.90 | ×0.45 |

---

## 📊 Dashboard Tabs
1. **Portfolio Overview** — Risk distribution, churn histogram, segment table
2. **Risk Intelligence** — CLV vs Churn scatter quadrant + Decision engine
3. **Model Performance** — AUC, F1, cross-validation, ensemble weights
4. **SHAP Explainability** — Global beeswarm, bar chart, local waterfall
5. **Cohort Analysis** — Monthly retention heatmap
6. **Stress Testing** — Monte Carlo VaR + 4-scenario stress table
7. **Customer Lookup** — Search any customer ID → risk card + SHAP breakdown

---

## 📄 Output Files
| File | Description |
|---|---|
| `data/processed/customer_risk_scores.csv` | All customers with churn prob, CLV, tier, decision |
| `data/processed/model_metrics.csv` | AUC, F1, Brier Score, CV results |
| `data/processed/mc_simulation.csv` | Monte Carlo VaR and CVaR |
| `data/processed/stress_tests.csv` | 4-scenario stress test results |
| `outputs/risk_report.pdf` | Full consultancy PDF report |
| `outputs/shap_summary.png` | SHAP beeswarm chart |
| `outputs/cohort_heatmap.png` | Cohort retention heatmap |

---

## 🛠️ Technologies
`Python 3.10+` · `XGBoost` · `LightGBM` · `scikit-learn` · `SHAP` · `Streamlit` · `Plotly` · `ReportLab` · `Pandas` · `NumPy` · `SciPy`

---

*© RiskIQ Analytics — Confidential & Proprietary*
