
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "ibm_churn_clean.csv"
MODEL_PATH = ROOT / "models" / "ibm_xgboost_model.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load data and trained pipeline
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Select one customer for demonstration
customer = df.iloc[[0]].copy()

customer_id = customer["customerID"].iloc[0]
actual_churn = customer["Churn"].iloc[0]

X_customer = customer.drop(columns=["customerID", "Churn"])

# Predict churn probability
probability = model.predict_proba(X_customer)[0, 1]

# Extract preprocessing and classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Apply the same transformations used during training
X_processed = preprocessor.transform(X_customer)

if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

feature_names = preprocessor.get_feature_names_out()

X_processed = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# Calculate individual SHAP values
explainer = shap.TreeExplainer(classifier)
shap_values = explainer(X_processed)

# Rank features by absolute SHAP contribution
contributions = pd.DataFrame({
    "Feature": feature_names,
    "SHAPValue": shap_values.values[0]
})

contributions["AbsoluteSHAP"] = (
    contributions["SHAPValue"].abs()
)

contributions = contributions.sort_values(
    "AbsoluteSHAP",
    ascending=False
)

print("\n--- CUSTOMER CHURN EXPLANATION ---")
print("Customer ID:", customer_id)
print("Actual churn:", actual_churn)
print("Predicted churn probability:", round(probability * 100, 2), "%")

print("\n--- TOP 10 CONTRIBUTING FEATURES ---")
print(
    contributions[["Feature", "SHAPValue"]]
    .head(10)
    .to_string(index=False)
)

# Create SHAP waterfall plot
shap.plots.waterfall(
    shap_values[0],
    max_display=15,
    show=False
)

plt.savefig(
    REPORT_DIR / "ibm_customer_waterfall.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nWaterfall plot saved successfully!")
