
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load monthly usage data
usage = pd.read_csv(RAW_DIR / "usage_data.csv")

# Convert month column to datetime
usage["Month"] = pd.to_datetime(usage["Month"])

# Sort customer records chronologically
usage = usage.sort_values(["CustomerID", "Month"])

# Calculate overall usage statistics
features = usage.groupby("CustomerID").agg(
    AvgCallMinutes=("CallMinutes", "mean"),
    AvgDataUsageGB=("DataUsageGB", "mean"),
    AvgSMSCount=("SMSCount", "mean"),
    TotalComplaints=("Complaints", "sum"),
    AvgComplaints=("Complaints", "mean")
).reset_index()

# Compare the first 3 months with the last 3 months
first_three = usage.groupby("CustomerID").head(3)
last_three = usage.groupby("CustomerID").tail(3)

early_usage = first_three.groupby("CustomerID").agg(
    EarlyDataUsage=("DataUsageGB", "mean"),
    EarlyCallMinutes=("CallMinutes", "mean")
)

recent_usage = last_three.groupby("CustomerID").agg(
    RecentDataUsage=("DataUsageGB", "mean"),
    RecentCallMinutes=("CallMinutes", "mean")
)

# Combine early and recent usage
usage_trends = early_usage.join(recent_usage)

# Calculate changes in usage
usage_trends["DataUsageChange"] = (
    usage_trends["RecentDataUsage"]
    - usage_trends["EarlyDataUsage"]
)

usage_trends["CallMinutesChange"] = (
    usage_trends["RecentCallMinutes"]
    - usage_trends["EarlyCallMinutes"]
)

# Keep useful trend features
usage_trends = usage_trends[
    ["DataUsageChange", "CallMinutesChange"]
].reset_index()

# Merge statistics and trends
features = features.merge(
    usage_trends,
    on="CustomerID",
    how="left",
    validate="one_to_one"
)

# Save engineered features
output_path = PROCESSED_DIR / "usage_features.csv"
features.to_csv(output_path, index=False)

print("\n--- FEATURE ENGINEERING SUMMARY ---")
print("Feature dataset shape:", features.shape)
print("\nColumns:")
print(features.columns.tolist())
print("\nFirst 5 rows:")
print(features.head())
print("\nMissing values:")
print(features.isna().sum())
print("\nSaved to:", output_path)
