"""
Data Ingestion and Preprocessing for Uplift Modeling Benchmark.

Dataset: Kevin Hillstrom MineThatData E-Mail Analytics Dataset (Randomized Controlled Trial).
Features:
    - recency: Months since last purchase
    - history: Total dollar spend in previous year
    - mens: Bought men's merchandise (0/1)
    - womens: Bought women's merchandise (0/1)
    - newbie: Customer joined in past 12 months (0/1)
    - zip_code: Urban, Surburban, Rural (One-Hot Encoded)
    - channel: Web, Phone, Multichannel (One-Hot Encoded)

Treatment:
    - treatment = 1: Received marketing campaign (E-Mail)
    - treatment = 0: Control group (No E-Mail)

Outcomes:
    - visit: 1 if customer visited website within 2 weeks (binary engagement target)
    - conversion: 1 if customer purchased within 2 weeks (binary sales target)
    - spend: Total dollar purchase amount within 2 weeks (continuous monetary target)
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "hillstrom_raw.csv"
PROCESSED_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "uplift_train.csv"
PROCESSED_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "uplift_test.csv"


def load_and_preprocess_hillstrom(
    raw_csv_path: Path = RAW_PATH,
    target_treatment_col: str = "treatment",
) -> pd.DataFrame:
    """
    Loads raw Hillstrom dataset and engineers features for uplift modeling.
    
    Args:
        raw_csv_path: Path to the raw CSV file.
        target_treatment_col: Name for the binary treatment column.
        
    Returns:
        pd.DataFrame with clean encoded features, binary treatment, and targets.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw uplift file not found at: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)

    # 1. Binary treatment flag: 1 if received any email campaign, 0 if control (No E-Mail)
    df[target_treatment_col] = (df["segment"] != "No E-Mail").astype(int)

    # 2. Categorical encoding
    # Clean zip_code spelling ("Surburban" in original dataset -> "Suburban")
    df["zip_code_clean"] = df["zip_code"].replace({"Surburban": "Suburban"})
    zip_dummies = pd.get_dummies(df["zip_code_clean"], prefix="zip", drop_first=True, dtype=int)
    channel_dummies = pd.get_dummies(df["channel"], prefix="channel", drop_first=True, dtype=int)

    # 3. Numeric & binary feature list
    numeric_features = ["recency", "history"]
    binary_features = ["mens", "womens", "newbie"]
    base_features = numeric_features + binary_features

    # Combine into model dataframe
    processed_df = pd.concat([
        df[base_features],
        zip_dummies,
        channel_dummies,
        df[[target_treatment_col, "visit", "conversion", "spend", "segment"]]
    ], axis=1)

    # Remove duplicate columns if any from concat
    processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]

    return processed_df


def split_uplift_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify_cols: Tuple[str, str] = ("treatment", "visit"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits uplift dataset ensuring balanced treatment and outcome representation
    across both train and test partitions.
    
    Args:
        df: Processed uplift dataframe.
        test_size: Proportion of test split (default 0.20).
        random_state: Reproducibility seed.
        stratify_cols: Columns used for joint stratification.
        
    Returns:
        (train_df, test_df)
    """
    # Create joint stratification key
    joint_strat = df[stratify_cols[0]].astype(str) + "_" + df[stratify_cols[1]].astype(str)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=joint_strat,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    print("=" * 60)
    print("Pre-processing Hillstrom Uplift Benchmark Dataset...")
    print("=" * 60)

    df = load_and_preprocess_hillstrom(RAW_PATH)
    print(f"Loaded {len(df):,} total customer records.")
    print(f"Treatment breakdown:\n{df['treatment'].value_counts(normalize=True).round(4) * 100}%")
    print(f"Visit rate (Control): {df[df['treatment'] == 0]['visit'].mean():.4f}")
    print(f"Visit rate (Treated): {df[df['treatment'] == 1]['visit'].mean():.4f}")
    print(f"Overall ATE (Average Treatment Effect on Visit): "
          f"{(df[df['treatment'] == 1]['visit'].mean() - df[df['treatment'] == 0]['visit'].mean()):.4f}")

    train_df, test_df = split_uplift_data(df, test_size=0.20, random_state=42)
    print(f"\nTrain size: {len(train_df):,} | Test size: {len(test_df):,}")

    PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)
    print(f"Saved processed train partition to: {PROCESSED_TRAIN_PATH}")
    print(f"Saved processed test partition to: {PROCESSED_TEST_PATH}")


if __name__ == "__main__":
    main()
