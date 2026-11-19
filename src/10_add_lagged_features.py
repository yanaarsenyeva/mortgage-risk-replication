import numpy as np
import pandas as pd

from src.config import (
    MODELING_DATASET_PATH,
    MODELING_DATASET_WITH_LAGS_PATH,
    TARGET_COLUMN,
)


ACTIVE_STATES = [
    "CURRENT",
    "D30",
    "D60",
    "D90_PLUS",
]


def add_lagged_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    panel = panel.sort_values(
        ["loan_id", "period"],
        kind="mergesort",
    ).copy()

    grouped = panel.groupby("loan_id", sort=False)

    panel["state_lag1"] = grouped["state"].shift(1)

    panel["was_current_last_month"] = (
        panel["state_lag1"] == "CURRENT"
    ).astype(int)

    panel["was_d30_last_month"] = (
        panel["state_lag1"] == "D30"
    ).astype(int)

    panel["was_d60_last_month"] = (
        panel["state_lag1"] == "D60"
    ).astype(int)

    panel["was_d90_plus_last_month"] = (
        panel["state_lag1"] == "D90_PLUS"
    ).astype(int)

    is_delinquent = panel["state"].isin(
        [
            "D30",
            "D60",
            "D90_PLUS",
        ]
    )

    delinquent_group = is_delinquent.groupby(
        panel["loan_id"]
    )

    panel["consecutive_delinquent_months"] = (
        is_delinquent
        .astype(int)
        .groupby(
            [
                panel["loan_id"],
                (~is_delinquent).cumsum(),
            ]
        )
        .cumsum()
    )

    is_current = panel["state"] == "CURRENT"

    months_since_current = (
        is_current
        .astype(int)
        .groupby(
            [
                panel["loan_id"],
                (~is_current).cumsum(),
            ]
        )
        .cumsum()
    )

    panel["months_since_last_current"] = (
        months_since_current
        - is_current.astype(int)
    )

    for state in ACTIVE_STATES:
        state_flag = (
            panel["state"] == state
        ).astype(int)

        panel[
            f"num_{state.lower()}_last_12m"
        ] = (
            state_flag
            .groupby(panel["loan_id"])
            .rolling(
                window=12,
                min_periods=1,
            )
            .sum()
            .reset_index(level=0, drop=True)
        )

    panel["ever_delinquent"] = (
        panel["state"]
        .isin(
            [
                "D30",
                "D60",
                "D90_PLUS",
            ]
        )
        .groupby(panel["loan_id"])
        .cummax()
        .astype(int)
    )

    return panel


def validate_lags(
    panel: pd.DataFrame,
) -> None:
    print("Initial shape:", panel.shape)

    print("\nAfter lag features shape:", panel.shape)

    lag_columns = [
        "state_lag1",
        "was_current_last_month",
        "was_d30_last_month",
        "was_d60_last_month",
        "was_d90_plus_last_month",
        "consecutive_delinquent_months",
        "months_since_last_current",
        "num_current_last_12m",
        "num_d30_last_12m",
        "num_d60_last_12m",
        "num_d90_plus_last_12m",
        "ever_delinquent",
    ]

    print("\nMissing values in lag features:")
    print(panel[lag_columns].isna().sum())

    print("\nState lag distribution:")
    print(panel["state_lag1"].value_counts(dropna=False))

    print("\nConsecutive delinquent months:")
    print(
        panel[
            "consecutive_delinquent_months"
        ].value_counts().head(20)
    )


def main() -> None:
    panel = pd.read_parquet(
        MODELING_DATASET_PATH,
    )

    panel = add_lagged_features(panel)

    validate_lags(panel)

    panel.to_parquet(
        MODELING_DATASET_WITH_LAGS_PATH,
        index=False,
    )

    print(f"\nSaved: {MODELING_DATASET_WITH_LAGS_PATH}")


if __name__ == "__main__":
    main()