
from pathlib import Path
import pandas as pd

# Find the project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Load the three CSV files
customers = pd.read_csv(DATA_DIR / "customer_info.csv")
usage = pd.read_csv(DATA_DIR / "usage_data.csv")
churn = pd.read_csv(DATA_DIR / "churn_labels.csv")


def inspect_data(name, dataframe):
    print(f"\n{'=' * 50}")
    print(f"DATASET: {name}")
    print(f"{'=' * 50}")

    print("\nShape (rows, columns):")
    print(dataframe.shape)

    print("\nColumn names and data types:")
    print(dataframe.dtypes)

    print("\nFirst 5 rows:")
    print(dataframe.head())

    print("\nMissing values per column:")
    print(dataframe.isnull().sum())

    print("\nDuplicate rows:")
    print(dataframe.duplicated().sum())


if __name__ == "__main__":
    inspect_data("Customer Information", customers)
    inspect_data("Usage Data", usage)
    inspect_data("Churn Labels", churn)

    print("\nChurn distribution:")
    print(churn["Churn"].value_counts(dropna=False))

    print("\nChurn percentages:")
    print(
        (churn["Churn"].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )
