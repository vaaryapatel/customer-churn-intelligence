
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model_data.csv"

df = pd.read_csv(DATA_PATH)

numeric_features = [
    "Age",
    "MonthlyCharges",
    "AvgCallMinutes",
    "AvgDataUsageGB",
    "AvgSMSCount",
    "TotalComplaints",
    "AvgComplaints",
    "DataUsageChange",
    "CallMinutesChange",
    "AgeMissing",
]

print("\n--- CHURN RATE ---")
print(df["Churn"].value_counts(normalize=True).mul(100).round(2))

print("\n--- NUMERIC FEATURES BY CHURN ---")
print(
    df.groupby("Churn")[numeric_features]
    .mean()
    .round(3)
    .T
)

print("\n--- CHURN RATE BY CATEGORY ---")
for column in ["Gender", "Region", "ContractType"]:
    print(f"\n{column}:")
    print(
        (df.groupby(column)["Churn"].mean() * 100)
        .round(2)
        .sort_values(ascending=False)
    )

print("\n--- MISSING AGE VS CHURN ---")
print(
    df.groupby("AgeMissing")["Churn"]
    .agg(["count", "mean"])
    .assign(ChurnPercent=lambda x: x["mean"] * 100)
    .round(3)
)

print("\n--- CORRELATION WITH CHURN ---")
correlations = (
    df[numeric_features + ["Churn"]]
    .corr(numeric_only=True)["Churn"]
    .drop("Churn")
    .sort_values(key=abs, ascending=False)
)

print(correlations.round(4))

print("\nFeature analysis completed.")
