
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model_data.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["CustomerID", "SignupDate", "Churn"])
y = df["Churn"]

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

# Churn is the minority class
negative_count = (y == 0).sum()
positive_count = (y == 1).sum()

scale_pos_weight = negative_count / positive_count

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n--- XGBOOST MODEL EVALUATION ---")
print(classification_report(y_test, predictions, zero_division=0))

print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print(
    "PR-AUC (Average Precision):",
    round(average_precision_score(y_test, probabilities), 4)
)
