
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

customers = pd.read_csv(RAW / "customer_info.csv")
usage = pd.read_csv(RAW / "usage_data.csv")
labels = pd.read_csv(RAW / "churn_labels.csv")

print("\n--- CUSTOMER INFO: NUMERIC RANGES ---")
print(customers.select_dtypes(include="number").describe().T)

print("\n--- NEGATIVE CUSTOMER VALUES ---")
for col in customers.select_dtypes(include="number").columns:
    count = (customers[col] < 0).sum()
    if count:
        print(f"{col}: {count} negative values")
        print(customers.loc[customers[col] < 0, ["CustomerID", col]].head())

print("\n--- USAGE DATA: NUMERIC RANGES ---")
print(usage.select_dtypes(include="number").describe().T)

print("\n--- NEGATIVE USAGE VALUES ---")
for col in usage.select_dtypes(include="number").columns:
    count = (usage[col] < 0).sum()
    if count:
        print(f"{col}: {count} negative values")
        print(usage.loc[usage[col] < 0, ["CustomerID", "Month", col]].head())

print("\n--- LABEL CHECK ---")
print("Label values:", sorted(labels["Churn"].dropna().unique()))
print("Missing labels:", labels["Churn"].isna().sum())
print("Duplicate label IDs:", labels["CustomerID"].duplicated().sum())

print("\nSuspicious-value inspection completed.")
