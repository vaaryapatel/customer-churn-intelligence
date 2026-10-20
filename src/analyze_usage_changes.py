
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

usage = pd.read_csv(RAW_DIR / "usage_data.csv")
labels = pd.read_csv(RAW_DIR / "churn_labels.csv")

usage["Month"] = pd.to_datetime(usage["Month"])
usage = usage.sort_values(["CustomerID", "Month"])

metrics = ["CallMinutes", "DataUsageGB", "SMSCount"]

# Calculate month-to-month changes for each customer.
for metric in metrics:
    usage[f"{metric}Change"] = (
        usage.groupby("CustomerID")[metric].diff()
    )

change_columns = [f"{metric}Change" for metric in metrics]

# Summarize the changes for each customer.
features = usage.groupby("CustomerID")[change_columns].agg(
    ["mean", "std", "min"]
)

features.columns = [
    f"{metric}_{stat}"
    for metric, stat in features.columns
]

features = features.reset_index()

df = features.merge(
    labels,
    on="CustomerID",
    how="inner",
    validate="one_to_one"
)

print("\n--- CUSTOMER USAGE CHANGES BY CHURN ---")
print(
    df.groupby("Churn")[
        [c for c in df.columns if c not in ["CustomerID", "Churn"]]
    ].mean().T.round(4).to_string()
)

# Compare the last two months of data usage.
monthly_data = usage[
    ["CustomerID", "Month", "DataUsageGB"]
].copy()

monthly_data = monthly_data.sort_values(["CustomerID", "Month"])

last_two = monthly_data.groupby("CustomerID").tail(2)

last_two = last_two.groupby("CustomerID")["DataUsageGB"].agg(
    first="first",
    last="last"
)

last_two["DataUsageChange"] = last_two["last"] - last_two["first"]

last_two = last_two.reset_index().merge(
    labels,
    on="CustomerID",
    validate="one_to_one"
)

last_two["Declined"] = (
    last_two["DataUsageChange"] < 0
).astype(int)

print("\n--- CHURN RATE BY LAST-TWO-MONTH DATA USAGE CHANGE ---")

print(
    (last_two.groupby("Declined")["Churn"].mean() * 100)
    .round(2)
    .rename({
        0: "Usage did not decline",
        1: "Usage declined"
    })
)

print("\nUsage change analysis completed.")
