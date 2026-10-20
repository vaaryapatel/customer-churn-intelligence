
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n--- NUMERIC FEATURE SIGNIFICANCE ---")

numeric_features = [
    "Age",
    "MonthlyCharges",
    "AvgCallMinutes",
    "AvgDataUsageGB",
    "AvgSMSCount",
    "TotalComplaints",
    "DataUsageChange",
    "CallMinutesChange",
]

for feature in numeric_features:
    group0 = df.loc[df["Churn"] == 0, feature].dropna()
    group1 = df.loc[df["Churn"] == 1, feature].dropna()

    if len(group0) and len(group1):
        _, p_value = mannwhitneyu(
            group0,
            group1,
            alternative="two-sided"
        )

        print(f"{feature:22s} p-value = {p_value:.4f}")

print("\n--- CATEGORICAL FEATURE SIGNIFICANCE ---")

for feature in ["Gender", "Region", "ContractType"]:
    table = pd.crosstab(df[feature], df["Churn"])
    _, p_value, _, _ = chi2_contingency(table)

    print(f"{feature:22s} p-value = {p_value:.4f}")

print("\n--- MONTHLY CHARGES BY CHURN ---")
print(
    df.groupby("Churn")["MonthlyCharges"]
    .describe()
    .round(2)
)

print("\nStatistical signal check completed.")
