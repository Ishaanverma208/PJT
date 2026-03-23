import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -------------------------
# CONFIGURATION
# -------------------------
NUM_ROWS = 2000000
NUM_CUSTOMERS = 8000
NUM_PRODUCTS = 1800

np.random.seed(42)
random.seed(42)

# -------------------------
# Generate Customers
# -------------------------
customer_ids = [f"CUST{100000+i}" for i in range(NUM_CUSTOMERS)]

# -------------------------
# Generate Product Codes
# -------------------------
stock_codes = [f"STK{10000+i}" for i in range(NUM_PRODUCTS)]

# -------------------------
# Generate Dates (last 2 years)
# -------------------------
start_date = datetime(2023, 1, 1)

def random_date():
    return start_date + timedelta(days=random.randint(0, 730),
                                  seconds=random.randint(0, 86400))

# -------------------------
# Generate Data
# -------------------------
data = []

for i in range(NUM_ROWS):
    
    customer = random.choice(customer_ids)
    invoice_no = f"INV{1000000+i}"
    date = random_date()
    quantity = np.random.randint(1, 50)
    
    # Simulate price volatility
    base_price = np.random.uniform(10, 500)
    
    # Introduce some high-risk abnormal spikes
    if np.random.rand() < 0.02:
        base_price *= np.random.uniform(5, 15)
    
    price = round(base_price, 2)
    
    stock = random.choice(stock_codes)
    
    data.append([
        customer,
        date,
        invoice_no,
        quantity,
        price,
        stock
    ])

df = pd.DataFrame(data, columns=[
    "Customer ID",
    "InvoiceDate",
    "Invoice",
    "Quantity",
    "Price",
    "StockCode"
])

df.to_csv("financial_risk_dataset_large.csv", index=False)

print("Dataset Generated Successfully!")
print(df.head())