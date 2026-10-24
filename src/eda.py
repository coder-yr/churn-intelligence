
from pathlib import Path

import pandas as pd

# Locate the project root and dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place the CSV file inside data/raw/")
        return

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    print("\n=== CUSTOMER CHURN DATASET ===")
    print(f"Number of customers: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n=== FIRST 5 CUSTOMERS ===")
    print(df.head())

    print("\n=== COLUMN DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== CHURN DISTRIBUTION ===")
    print(df["Churn"].value_counts())

    print("\n=== CHURN PERCENTAGE ===")
    print(
        (df["Churn"].value_counts(normalize=True) * 100)
        .round(2)
    )


if __name__ == "__main__":
    main()
