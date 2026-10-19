
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(RAW_DIR / "customer_info.csv")

print("Original rows:", len(customers))

# Remove rows that match across every column
customers = customers.drop_duplicates()
print("Rows after exact duplicate removal:", len(customers))

# Find customer IDs that still have conflicting records
conflict_mask = customers["CustomerID"].duplicated(keep=False)
conflicts = customers.loc[conflict_mask].sort_values("CustomerID")

conflicts.to_csv(CLEAN_DIR / "customer_conflicts.csv", index=False)

# Keep unambiguous customer records only
clean_customers = customers.loc[~conflict_mask].copy()

# Convert signup dates to pandas datetime
clean_customers["SignupDate"] = pd.to_datetime(
    clean_customers["SignupDate"],
    errors="coerce"
)

# Save cleaned customer data
clean_customers.to_csv(
    CLEAN_DIR / "customer_info_clean.csv",
    index=False
)

print("\n--- CLEANING SUMMARY ---")
print("Clean customer rows:", len(clean_customers))
print("Conflicting customer IDs:", conflicts["CustomerID"].nunique())
print("Rows saved for conflict review:", len(conflicts))
print("Invalid signup dates:", clean_customers["SignupDate"].isna().sum())
print("Missing ages remaining:", clean_customers["Age"].isna().sum())
print("\nSaved files:")
print(CLEAN_DIR / "customer_info_clean.csv")
print(CLEAN_DIR / "customer_conflicts.csv")
