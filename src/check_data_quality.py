
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

customers = pd.read_csv(DATA_DIR / "customer_info.csv")
usage = pd.read_csv(DATA_DIR / "usage_data.csv")
churn = pd.read_csv(DATA_DIR / "churn_labels.csv")

# 1. Check customer IDs
print("\n--- CUSTOMER ID CHECK ---")

for name, df in [
    ("Customers", customers),
    ("Usage", usage),
    ("Churn labels", churn),
]:
    print(f"\n{name}")
    print("Rows:", len(df))
    print("Unique customer IDs:", df["CustomerID"].nunique())
    print("Rows with repeated customer IDs:",
          df["CustomerID"].duplicated().sum())

# 2. Check duplicate customer records
print("\n--- CUSTOMER DUPLICATE CHECK ---")
print("Exact duplicate rows:", customers.duplicated().sum())
print(
    "Duplicate customer IDs:",
    customers.loc[
        customers["CustomerID"].duplicated(keep=False),
        "CustomerID"
    ].nunique()
)

# 3. Compare IDs between customer and churn datasets
customer_ids = set(customers["CustomerID"])
churn_ids = set(churn["CustomerID"])
usage_ids = set(usage["CustomerID"])

print("\n--- ID MATCHING CHECK ---")
print("Customers without churn labels:", len(customer_ids - churn_ids))
print("Churn labels without customer details:", len(churn_ids - customer_ids))
print("Usage customers without churn labels:", len(usage_ids - churn_ids))
print("Churn customers without usage records:", len(churn_ids - usage_ids))

# 4. Check usage records per customer
usage_counts = usage.groupby("CustomerID").size()

print("\n--- USAGE RECORD CHECK ---")
print("Minimum records per customer:", usage_counts.min())
print("Maximum records per customer:", usage_counts.max())
print("Average records per customer:", round(usage_counts.mean(), 2))

print("\nRecords per customer distribution:")
print(usage_counts.value_counts().sort_index())

# 5. Check usage months
print("\n--- MONTH CHECK ---")
print("Unique months:", usage["Month"].nunique())
print("First month:", usage["Month"].min())
print("Last month:", usage["Month"].max())

print("\nData quality checks completed.")


print("\n--- REPEATED CUSTOMER RECORD DETAILS ---")

repeated_ids = customers.loc[
    customers["CustomerID"].duplicated(keep=False),
    "CustomerID"
].unique()

repeated_records = customers[
    customers["CustomerID"].isin(repeated_ids)
].sort_values("CustomerID")

print("Number of repeated customer IDs:", len(repeated_ids))

print("\nFirst 20 repeated records:")
print(repeated_records.head(20).to_string(index=False))

print("\nDo repeated records disagree on any fields?")
for customer_id, group in repeated_records.groupby("CustomerID"):
    if len(group.drop_duplicates()) > 1:
        print(f"\n{customer_id}:")
        print(group.to_string(index=False))
