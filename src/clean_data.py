
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(RAW_DIR / "customer_info.csv")

print("Original rows:", len(customers))

# Remove exact duplicates
customers = customers.drop_duplicates()

print("Rows after exact duplicate removal:", len(customers))

# Identify conflicting customer IDs
conflict_mask = customers["CustomerID"].duplicated(keep=False)

conflicts = customers.loc[conflict_mask].copy()

conflicts.to_csv(
    CLEAN_DIR / "customer_conflicts.csv",
    index=False
)

# Preserve one record per customer
clean_customers = customers.drop_duplicates(
    subset="CustomerID",
    keep="first"
).copy()

# Set uncertain signup dates to missing
conflicting_ids = set(conflicts["CustomerID"])

clean_customers.loc[
    clean_customers["CustomerID"].isin(conflicting_ids),
    "SignupDate"
] = pd.NA

# Convert valid dates
clean_customers["SignupDate"] = pd.to_datetime(
    clean_customers["SignupDate"],
    errors="coerce"
)

# Track missing ages without imputing them yet
clean_customers["AgeMissing"] = (
    clean_customers["Age"].isna().astype(int)
)

# Save cleaned data
clean_customers.to_csv(
    CLEAN_DIR / "customer_info_clean.csv",
    index=False
)

print("\n--- FINAL CUSTOMER CLEANING SUMMARY ---")
print("Clean customer rows:", len(clean_customers))
print("Unique customer IDs:", clean_customers["CustomerID"].nunique())
print("Conflicting customer IDs:", len(conflicting_ids))
print("Missing signup dates:", clean_customers["SignupDate"].isna().sum())
print("Missing ages:", clean_customers["Age"].isna().sum())
print("AgeMissing column created:", "AgeMissing" in clean_customers.columns)

print("\nCustomer cleaning completed!")
