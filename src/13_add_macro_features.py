import pandas as pd

from src.config import (
    MODELING_DATASET_WITH_LAGS_AND_MACRO_PATH,
    MODELING_DATASET_WITH_LAGS_PATH,
    MORTGAGE30US_PATH,
)


def load_market_rates() -> pd.DataFrame:
    market_rates = pd.read_csv(
        MORTGAGE30US_PATH,
    )

    print("Market rate columns:")
    print(market_rates.columns.tolist())

    date_column = None

    for candidate in [
        "DATE",
        "date",
        "observation_date",
    ]:
        if candidate in market_rates.columns:
            date_column = candidate
            break

    if date_column is None:
        raise ValueError(
            "Cannot find date column in market rate CSV. "
            f"Available columns: {market_rates.columns.tolist()}"
        )

    rate_column = None

    for candidate in [
        "MORTGAGE30US",
        "PMMS30US",
        "national_mortgage_rate",
    ]:
        if candidate in market_rates.columns:
            rate_column = candidate
            break

    if rate_column is None:
        raise ValueError(
            "Cannot find mortgage rate column in market rate CSV. "
            f"Available columns: {market_rates.columns.tolist()}"
        )

    market_rates = market_rates.rename(
        columns={
            date_column: "date",
            rate_column: "national_mortgage_rate",
        }
    )

    market_rates["date"] = pd.to_datetime(
        market_rates["date"],
        errors="coerce",
    )

    market_rates["period"] = (
        market_rates["date"].dt.year * 100
        + market_rates["date"].dt.month
    )

    market_rates["national_mortgage_rate"] = pd.to_numeric(
        market_rates["national_mortgage_rate"],
        errors="coerce",
    )

    market_rates = market_rates[
        [
            "period",
            "national_mortgage_rate",
        ]
    ].dropna()

    market_rates = market_rates.groupby(
        "period",
        as_index=False,
    ).mean()

    return market_rates


def add_macro_features(
    panel: pd.DataFrame,
    market_rates: pd.DataFrame,
) -> pd.DataFrame:
    panel = panel.copy()

    panel["period"] = pd.to_numeric(
        panel["period"],
        errors="coerce",
    ).astype("Int64")

    market_rates = market_rates.copy()

    market_rates["period"] = pd.to_numeric(
        market_rates["period"],
        errors="coerce",
    ).astype("Int64")

    panel = panel.merge(
        market_rates,
        on="period",
        how="left",
        validate="many_to_one",
    )

    panel["original_interest_rate"] = pd.to_numeric(
        panel["original_interest_rate"],
        errors="coerce",
    )

    panel["current_interest_rate"] = pd.to_numeric(
        panel["current_interest_rate"],
        errors="coerce",
    )

    panel["national_mortgage_rate"] = pd.to_numeric(
        panel["national_mortgage_rate"],
        errors="coerce",
    )

    panel["original_rate_minus_market_rate"] = (
        panel["original_interest_rate"]
        - panel["national_mortgage_rate"]
    )

    panel["current_rate_minus_market_rate"] = (
        panel["current_interest_rate"]
        - panel["national_mortgage_rate"]
    )

    return panel


def validate_macro(
    panel: pd.DataFrame,
) -> None:
    print("Initial shape:", panel.shape)

    print("\nAfter macro features shape:", panel.shape)

    print("\nMissing national mortgage rate:")
    print(
        panel["national_mortgage_rate"].isna().sum()
    )

    print("\nMarket rate summary:")
    print(
        panel["national_mortgage_rate"].describe()
    )

    print("\nPrepayment incentive summary:")
    print(
        panel[
            "original_rate_minus_market_rate"
        ].describe()
    )

    print("\nCurrent rate incentive summary:")
    print(
        panel[
            "current_rate_minus_market_rate"
        ].describe()
    )


def main() -> None:
    panel = pd.read_parquet(
        MODELING_DATASET_WITH_LAGS_PATH,
    )

    market_rates = load_market_rates()

    print("Market rates shape:", market_rates.shape)
    print(
        "Market rate periods:",
        market_rates["period"].min(),
        "-",
        market_rates["period"].max(),
    )

    panel = add_macro_features(
        panel,
        market_rates,
    )

    validate_macro(panel)

    panel.to_parquet(
        MODELING_DATASET_WITH_LAGS_AND_MACRO_PATH,
        index=False,
    )

    print(
        f"\nSaved: {MODELING_DATASET_WITH_LAGS_AND_MACRO_PATH}"
    )


if __name__ == "__main__":
    main()