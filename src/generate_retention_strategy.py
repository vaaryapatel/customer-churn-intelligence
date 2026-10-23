
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import time
from google.genai import errors
from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)

DATA_PATH = ROOT / "data" / "processed" / "ibm_churn_clean.csv"
MODEL_PATH = ROOT / "models" / "ibm_xgboost_model.joblib"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Select one customer for testing.
customer = df.iloc[[0]].copy()

customer_id = customer["customerID"].iloc[0]
X_customer = customer.drop(columns=["customerID", "Churn"])

# Predict churn probability.
churn_probability = float(
    model.predict_proba(X_customer)[0, 1]
)

# Get fitted preprocessing and XGBoost classifier.
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_processed = preprocessor.transform(X_customer)

if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

feature_names = preprocessor.get_feature_names_out()

X_processed = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# Calculate individual SHAP values.
explainer = shap.TreeExplainer(classifier)
shap_values = explainer(X_processed)

contributions = pd.DataFrame({
    "Feature": feature_names,
    "SHAPValue": shap_values.values[0]
})

contributions["AbsoluteSHAP"] = (
    contributions["SHAPValue"].abs()
)

top_factors = contributions.sort_values(
    "AbsoluteSHAP",
    ascending=False
).head(10)

# Remove demographic attributes from recommendation inputs.
sensitive_columns = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
}

profile = X_customer.drop(
    columns=list(sensitive_columns),
    errors="ignore"
).iloc[0].to_dict()

# Filter out demographic SHAP factors.
top_factors = top_factors[
    ~top_factors["Feature"].str.contains(
        "gender|SeniorCitizen|Partner|Dependents",
        case=False,
        regex=True
    )
]

# Convert NumPy values into ordinary Python values.
profile = {
    key: value.item() if isinstance(value, np.generic) else value
    for key, value in profile.items()
}

prompt = f"""
You are a telecom customer retention analyst.

Analyze the following customer's predicted churn risk
and recommend practical retention strategies.

Customer ID: {customer_id}
Predicted churn probability: {churn_probability:.2%}

Customer profile:
{profile}

Top model explanation factors:
{top_factors[["Feature", "SHAPValue"]].to_string(index=False)}

Instructions:
1. Summarize the customer's churn risk.
2. Identify the most relevant actionable risk factors.
3. Suggest 3 personalized retention strategies.
4. Explain why each strategy might help.
5. Assign a priority level (High, Medium, Low).
6. Do not invent discounts, company policies, or benefits.
7. Do not use demographic attributes in recommendations.
8. Do not claim SHAP contributions prove causation.
9. Present recommendations as suggestions requiring
   business validation, not guaranteed outcomes.
10. Use clear, professional language.

Format:
Customer Risk Summary
Key Risk Factors
Recommended Retention Strategies
Priority
"""

response = None
max_attempts = 4

for attempt in range(max_attempts):
    try:
        print(f"Calling Gemini (attempt {attempt + 1}/{max_attempts})...")

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        break

    except errors.ServerError as e:
        if e.code not in (500, 502, 503, 504):
            raise

        if attempt == max_attempts - 1:
            print("Gemini is temporarily unavailable. Please try later.")
            raise

        wait_seconds = 2 ** (attempt + 1)
        print(f"Server busy. Retrying in {wait_seconds} seconds...")
        time.sleep(wait_seconds)

print("\n--- CUSTOMER RETENTION ANALYSIS ---")
print("Customer ID:", customer_id)
print("Predicted churn probability:", f"{churn_probability:.2%}")

print("\n--- GEMINI RETENTION RECOMMENDATIONS ---")
print(response.text or "No text response returned.")
