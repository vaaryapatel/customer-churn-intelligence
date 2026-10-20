
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

usage = pd.read_csv(RAW_DIR / "usage_data.csv")
labels = pd.read_csv(RAW_DIR / "churn_labels.csv")

df = usage.merge(
    labels,
    on="CustomerID",
    how="inner",
    validate="many_to_one"
)

df["Month"] = pd.to_datetime(df["Month"])

metrics = [
    "CallMinutes",
    "DataUsageGB",
    "SMSCount",
    "Complaints",
]

print("\n--- MONTHLY AVERAGES BY CHURN STATUS ---")

monthly = (
    df.groupby(["Month", "Churn"])[metrics]
    .mean()
    .round(3)
)

print(monthly.to_string())

print("\n--- OVERALL DIFFERENCE: CHURNED MINUS NON-CHURNED ---")

for metric in metrics:
    means = df.groupby("Churn")[metric].mean()

    difference = means.loc[1] - means.loc[0]

    print(
        f"{metric}: "
        f"Stayed={means.loc[0]:.3f}, "
        f"Churned={means.loc[1]:.3f}, "
        f"Difference={difference:.3f}"
    )

print("\nMonthly usage analysis completed.")
