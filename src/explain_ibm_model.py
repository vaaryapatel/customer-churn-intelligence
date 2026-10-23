
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

# Load dataset and trained model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df.drop(columns=["customerID", "Churn"])

# Separate preprocessing and XGBoost classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Transform input data using fitted preprocessing
X_processed = preprocessor.transform(X)

# Convert sparse matrix to dense array if necessary
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# Retrieve feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

X_processed = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# Select a sample to keep plots manageable
X_sample = X_processed.sample(
    n=min(500, len(X_processed)),
    random_state=42
)

# Explain the trained XGBoost model
explainer = shap.TreeExplainer(classifier)
shap_values = explainer(X_sample)

# SHAP summary plot
shap.summary_plot(
    shap_values.values,
    X_sample,
    show=False
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "ibm_shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# SHAP importance plot
shap.summary_plot(
    shap_values.values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "ibm_shap_importance.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Save importance values to CSV
importance = pd.DataFrame({
    "Feature": feature_names,
    "MeanAbsSHAP": np.abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    "MeanAbsSHAP",
    ascending=False
)

importance.to_csv(
    REPORT_DIR / "ibm_shap_importance.csv",
    index=False
)

print("\n--- TOP 15 IMPORTANT FEATURES ---")
print(importance.head(15).to_string(index=False))

print("\nSHAP plots saved to:", REPORT_DIR)
print("SHAP explainability completed!")
