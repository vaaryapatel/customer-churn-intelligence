
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Load datasets
customers = pd.read_csv(
    PROCESSED_DIR / "customer_info_clean.csv"
)

usage_features = pd.read_csv(
    PROCESSED_DIR / "usage_features.csv"
)

churn = pd.read_csv(
    RAW_DIR / "churn_labels.csv"
)

# Merge customer information with usage features
merged = customers.merge(
    usage_features,
    on="CustomerID",
    how="inner",
    validate="one_to_one"
)

# Add churn labels
merged = merged.merge(
    churn,
    on="CustomerID",
    how="inner",
    validate="one_to_one"
)

# Validate that all customers are present
assert len(merged) == len(churn), (
    "Some labelled customers were lost during merging."
)

assert merged["CustomerID"].is_unique, (
    "Duplicate customer IDs found after merging."
)

# Save final dataset
output_path = PROCESSED_DIR / "churn_model_data.csv"
merged.to_csv(output_path, index=False)

print("\n--- MERGED DATASET SUMMARY ---")
print("Shape:", merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nMissing values:")
print(merged.isna().sum())

print("\nChurn distribution:")
print(merged["Churn"].value_counts())

print("\nFirst 5 rows:")
print(merged.head())

print("\nSaved to:", output_path)
