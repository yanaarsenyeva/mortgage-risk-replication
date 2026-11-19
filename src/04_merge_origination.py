import pandas as pd

from src.config import (
    PANEL_PATH,
    PANEL_WITH_ORIGINATION_PATH,
)
from src.data import read_origination


ORIGINATION_FEATURES = [
    "loan_id",
    "credit_score",
    "first_time_homebuyer",
    "maturity_date",
    "mi_pct",
    "num_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty",
    "product_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_purpose",
    "original_loan_term",
    "num_borrowers",
    "seller_name",
    "property_valuation_method",
    "interest_only",
    "vantage_score",
]


def merge_origination(
    panel: pd.DataFrame,
    origination: pd.DataFrame,
) -> pd.DataFrame:
    origination_features = origination.loc[
        :,
        ORIGINATION_FEATURES,
    ].copy()

    origination_features = origination_features.drop_duplicates(
        subset=["loan_id"],
        keep="first",
    )

    panel_with_origination = panel.merge(
        origination_features,
        on="loan_id",
        how="left",
        validate="many_to_one",
    )

    return panel_with_origination


def validate_merge(
    panel: pd.DataFrame,
    panel_with_origination: pd.DataFrame,
) -> None:
    print("Panel shape:", panel.shape)
    print("Origination shape:", panel_with_origination.shape)

    missing_origination = panel_with_origination[
        "credit_score"
    ].isna()

    print("\nMissing origination records:")
    print(missing_origination.sum())

    print("\nNumeric summary:")
    numeric_columns = [
        "credit_score",
        "original_cltv",
        "original_dti",
        "original_upb",
        "original_ltv",
        "original_interest_rate",
        "original_loan_term",
        "num_borrowers",
    ]

    numeric_summary = panel_with_origination[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).describe()

    print(numeric_summary)

    print("\nCategorical distributions:")

    categorical_columns = [
        "first_time_homebuyer",
        "num_units",
        "occupancy_status",
        "property_type",
        "property_state",
        "loan_purpose",
        "channel",
        "product_type",
    ]

    for column in categorical_columns:
        print(f"\n{column}:")
        print(
            panel_with_origination[column]
            .value_counts(dropna=False)
            .head(10)
        )


def main() -> None:
    panel = pd.read_parquet(
        PANEL_PATH,
    )

    origination = read_origination()

    print("Panel shape:", panel.shape)
    print("Origination shape:", origination.shape)

    panel_with_origination = merge_origination(
        panel,
        origination,
    )

    validate_merge(
        panel,
        panel_with_origination,
    )

    panel_with_origination.to_parquet(
        PANEL_WITH_ORIGINATION_PATH,
        index=False,
    )

    print(f"\nSaved: {PANEL_WITH_ORIGINATION_PATH}")


if __name__ == "__main__":
    main()