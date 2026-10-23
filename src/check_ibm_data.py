
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "ibm" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE CUSTOMERS ---")
print(df["customerID"].duplicated().sum())

print("\n--- TOTAL CHARGES CHECK ---")

# Convert TotalCharges to numeric.
# Invalid or blank values become NaN.
total_charges = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Invalid or missing TotalCharges:", total_charges.isna().sum())

print("\n--- NEGATIVE CHARGES ---")
print("Negative MonthlyCharges:", (df["MonthlyCharges"] < 0).sum())
print("Negative TotalCharges:", (total_charges < 0).sum())

print("\n--- CHURN DISTRIBUTION ---")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).mul(100).round(2))

print("\nIBM dataset quality check completed!")
