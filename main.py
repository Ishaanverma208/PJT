import pandas as pd

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/raw/online_retail_II.csv")
print("Loaded:", df.shape)

# -----------------------------
# 2. CLEAN DATA
# -----------------------------
df = df.dropna(subset=["Customer ID"])
df = df[~df["Invoice"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])

df["TotalAmount"] = df["Quantity"] * df["Price"]

print("After cleaning:", df.shape)

# -----------------------------
# 3. CREATE RFM FEATURES
# -----------------------------
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "Invoice": "nunique",
    "TotalAmount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("RFM created:", rfm.shape)

# -----------------------------
# 4. EXTRA FEATURES
# -----------------------------

# Average Order Value
rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

# Monthly Trend (Fixed Version)
monthly = (
    df.set_index("InvoiceDate")
      .groupby("Customer ID")["TotalAmount"]
      .resample("ME")   # Fixed warning
      .sum()
)

trend = monthly.groupby("Customer ID").apply(
    lambda x: x.pct_change().mean()
)

# Add Trend column
rfm["Trend"] = trend

# Fix infinite values
import numpy as np
rfm["Trend"].replace([np.inf, -np.inf], 0, inplace=True)
# -----------------------------
# 5. CREATE CHURN LABEL
# -----------------------------
rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

print("Churn distribution:")
print(rfm["Churn"].value_counts())

# -----------------------------
# 6. HANDLE MISSING
# -----------------------------
rfm = rfm.fillna(0)

# -----------------------------
# 7. SAVE DATASET
# -----------------------------
rfm.to_csv("data/processed/customer_features.csv")

print("Saved: data/processed/customer_features.csv")
print(rfm.head())
# -----------------------------
# 8. TRAIN MODEL
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

# Prepare data
X = rfm.drop(["Churn", "Recency"], axis=1)
y = rfm["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("AUC Score:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
# -----------------------------
# 10. DECISION ENGINE
# -----------------------------

# Get churn probability for ALL customers
churn_prob = model.predict_proba(X)[:, 1]
rfm["Churn_Prob"] = churn_prob


# -----------------------------
# Business Parameters
# -----------------------------
RETENTION_COST = 500      # ₹ cost per customer
AVG_MARGIN = 0.25         # 25% profit margin
DISCOUNT = 0.10           # 10% discount


# -----------------------------
# Customer Value & Risk
# -----------------------------

# Customer Lifetime Value
rfm["CLV"] = rfm["Monetary"] * AVG_MARGIN

# Expected loss if customer leaves
rfm["Expected_Loss"] = rfm["Churn_Prob"] * rfm["CLV"]

# Expected gain from retention
rfm["Retention_Gain"] = rfm["CLV"] * (1 - DISCOUNT)

# ROI of retention
rfm["ROI"] = rfm["Retention_Gain"] - RETENTION_COST


# -----------------------------
# Decision Policy
# -----------------------------
def decision_policy(row):

    if row["Churn_Prob"] > 0.7 and row["ROI"] > 0:
        return "HIGH PRIORITY RETENTION"

    elif row["Churn_Prob"] > 0.4 and row["ROI"] > 0:
        return "MEDIUM PRIORITY RETENTION"

    elif row["Churn_Prob"] <= 0.4:
        return "NO ACTION"

    else:
        return "DO NOT RETAIN"


# Apply decision policy
rfm["Decision"] = rfm.apply(decision_policy, axis=1)


# -----------------------------
# View Results
# -----------------------------
print("\nDecision Distribution:")
print(rfm["Decision"].value_counts())

print("\nSample Decisions:")
print(rfm[["Churn_Prob", "CLV", "ROI", "Decision"]].head())

# -----------------------------
# 9. EXPLAINABILITY (SHAP)
# -----------------------------

import shap

# Create explainer
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values
shap_values = explainer(X_test)

# Global importance plot
shap.summary_plot(shap_values, X_test, show=False)

# Save plot
import matplotlib.pyplot as plt
plt.savefig("shap_summary.png")
plt.close()

print("SHAP explanation saved as shap_summary.png")