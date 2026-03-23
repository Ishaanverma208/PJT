"""
========================================================
  FLAGSHIP FINANCIAL RISK MANAGEMENT SYSTEM
  Orchestrator — Run Everything in One Command
  Usage: python run.py
========================================================
"""

import os
import sys
import subprocess

os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def banner(msg):
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)

def run_step(label, script):
    banner(f"STEP: {label}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n  ❌ ERROR in {script}. Check output above.")
        sys.exit(1)
    print(f"\n  ✅ {label} — COMPLETED")

if __name__ == "__main__":
    print("\n" + "★" * 65)
    print("  RISKIQ FINANCIAL RISK MANAGEMENT PLATFORM v2.0")
    print("  Full Pipeline Orchestrator")
    print("★" * 65)

    # Step 1: Run ML engine
    run_step("Advanced ML Risk Engine", "risk_engine.py")

    # Step 2: Generate PDF report
    run_step("PDF Report Generation", "report_generator.py")

    banner("ALL STEPS COMPLETE 🎉")
    print("""
  ┌─────────────────────────────────────────────────┐
  │  GENERATED ARTIFACTS                            │
  │                                                 │
  │  data/processed/                                │
  │    ├── customer_risk_scores.csv  (all scores)   │
  │    ├── model_metrics.csv         (AUC, F1 ...)  │
  │    ├── mc_simulation.csv         (VaR, CVaR)    │
  │    ├── stress_tests.csv          (4 scenarios)  │
  │    └── cohort_retention.csv      (monthly)      │
  │                                                 │
  │  outputs/                                       │
  │    ├── risk_report.pdf           (full report)  │
  │    ├── shap_summary.png                         │
  │    ├── shap_bar.png                             │
  │    ├── shap_waterfall.png                       │
  │    ├── risk_distribution.png                    │
  │    └── cohort_heatmap.png                       │
  │                                                 │
  │  TO LAUNCH DASHBOARD:                           │
  │    streamlit run dashboard.py                   │
  └─────────────────────────────────────────────────┘
    """)
