
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "raw" / "ibm" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)

print("Original shape:", df.shape)

# Convert TotalCharges from text to numeric.
# Blank or invalid entries become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Convert the target variable to binary.
# No = 0 (customer stayed)
# Yes = 1 (customer churned)
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Ensure the target was converted correctly.
assert df["Churn"].notna().all(), "Unexpected churn label found"

# Keep track of missing TotalCharges.
df["TotalChargesMissing"] = (
    df["TotalCharges"].isna().astype(int)
)

# Check for duplicate customer IDs.
assert df["customerID"].is_unique, "Duplicate customer IDs found"

# Save the cleaned dataset.
output_path = OUTPUT_DIR / "ibm_churn_clean.csv"
df.to_csv(output_path, index=False)

print("\n--- CLEANED DATA SUMMARY ---")
print("Cleaned shape:", df.shape)
print("Missing TotalCharges:", df["TotalCharges"].isna().sum())
print("Missing target labels:", df["Churn"].isna().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nData types:")
print(df.dtypes)

print("\nSaved cleaned data to:", output_path)
print("\nIBM data cleaning completed!")
