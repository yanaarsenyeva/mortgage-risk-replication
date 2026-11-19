import numpy as np
import pandas as pd

from src.config import (
    MODELING_DATASET_PATH,
    PANEL_WITH_ORIGINATION_PATH,
    REPORTS_TABLES_DIR,
    TARGET_COLUMN,
    TIME_COLUMN,
)


NUMERIC_COLUMNS = [
    "credit_score",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
    "num_borrowers",
    "current_actual_upb",
    "current_interest_rate",
    "loan_age",
    "remaining_months_to_maturity",
]

CATEGORICAL_COLUMNS = [
    "first_time_homebuyer",
    "num_units",
    "occupancy_status",
    "property_type",
    "property_state",
    "loan_purpose",
    "channel",
    "product_type",
    "state",
]


def convert_numeric_columns(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    panel = panel.copy()

    for column in NUMERIC_COLUMNS:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

    return panel


def add_time_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    panel = panel.copy()

    panel["period_dt"] = pd.to_datetime(
        panel[TIME_COLUMN],
        format="%Y%m",
        errors="coerce",
    )

    panel["year"] = panel["period_dt"].dt.year
    panel["month"] = panel["period_dt"].dt.month

    panel["loan_age_months"] = panel["loan_age"]

    panel["months_to_maturity"] = panel[
        "remaining_months_to_maturity"
    ]

    panel["current_rate_spread"] = (
        panel["current_interest_rate"]
        - panel["original_interest_rate"]
    )

    panel["unpaid_principal_ratio"] = (
        panel["current_actual_upb"]
        / panel["original_upb"]
    )

    panel["prepayment_incentive"] = (
        panel["current_interest_rate"]
        - panel["original_interest_rate"]
    )

    return panel


def add_temporal_split(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    panel = panel.copy()

    periods = (
        panel[TIME_COLUMN]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    test_cutoff_index = int(
        len(periods) * 0.8
    )

    validation_cutoff_index = int(
        len(periods) * 0.7
    )

    test_start_period = periods[
        test_cutoff_index
    ]

    validation_start_period = periods[
        validation_cutoff_index
    ]

    panel["split"] = np.select(
        [
            panel[TIME_COLUMN] >= test_start_period,
            panel[TIME_COLUMN] >= validation_start_period,
        ],
        [
            "test",
            "validation",
        ],
        default="train",
    )

    return panel


def validate_dataset(
    panel: pd.DataFrame,
) -> None:
    print("Initial shape:", panel.shape)

    print("\nAfter split shape:", panel.shape)

    print("\nSplit distribution:")
    print(panel["split"].value_counts())

    print("\nTarget distribution:")
    print(panel[TARGET_COLUMN].value_counts())

    print("\nMissing values in numeric features:")

    missing_values = panel[
        NUMERIC_COLUMNS
        + [
            "loan_age_months",
            "months_to_maturity",
            "current_rate_spread",
            "unpaid_principal_ratio",
            "prepayment_incentive",
        ]
    ].isna().sum()

    print(missing_values)


def main() -> None:
    panel = pd.read_parquet(
        PANEL_WITH_ORIGINATION_PATH,
    )

    panel = convert_numeric_columns(panel)
    panel = add_time_features(panel)
    panel = add_temporal_split(panel)

    validate_dataset(panel)

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_summary = (
        panel
        .groupby("split")
        .size()
        .reset_index(name="n")
    )

    split_summary.to_csv(
        REPORTS_TABLES_DIR / "split_summary.csv",
        index=False,
    )

    panel.to_parquet(
        MODELING_DATASET_PATH,
        index=False,
    )

    print(f"\nSaved: {MODELING_DATASET_PATH}")
    print(
        "Saved: "
        f"{REPORTS_TABLES_DIR / 'split_summary.csv'}"
    )


if __name__ == "__main__":
    main()