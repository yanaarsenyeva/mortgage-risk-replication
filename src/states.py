from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PANEL_PATH,
    PROCESSED_DATA_DIR,
)
from src.data import read_performance


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text == "":
        return ""

    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if np.isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))

    return text


def normalize_state(
    delinquency_status: object,
    zero_balance_code: object,
) -> str:
    delinquency_status = normalize_code(delinquency_status)
    zero_balance_code = normalize_code(zero_balance_code)

    if zero_balance_code == "1":
        return "PREPAID"

    if zero_balance_code == "9":
        return "REO_DISPOSITION"

    if zero_balance_code in {"2", "3", "15", "16", "96"}:
        return "TERMINATED_OTHER"

    if delinquency_status == "0":
        return "CURRENT"

    if delinquency_status == "1":
        return "D30"

    if delinquency_status == "2":
        return "D60"

    if delinquency_status.isdigit() and int(delinquency_status) >= 3:
        return "D90_PLUS"

    if delinquency_status == "RA":
        return "REO_ACQUISITION"

    return "UNKNOWN"


def build_states(performance: pd.DataFrame) -> pd.DataFrame:
    performance = performance.copy()

    performance["state"] = [
        normalize_state(
            delinquency_status,
            zero_balance_code,
        )
        for delinquency_status, zero_balance_code in zip(
            performance["current_loan_delinquency_status"],
            performance["zero_balance_code"],
        )
    ]

    return performance


def add_period_datetime(performance: pd.DataFrame) -> pd.DataFrame:
    performance = performance.copy()

    performance["period_dt"] = pd.to_datetime(
        performance["period"],
        format="%Y%m",
        errors="coerce",
    )

    return performance


def build_transitions(performance: pd.DataFrame) -> pd.DataFrame:
    performance = performance.sort_values(
        ["loan_id", "period"],
        kind="mergesort",
    ).copy()

    grouped = performance.groupby("loan_id", sort=False)

    performance["next_state"] = grouped["state"].shift(-1)
    performance["next_period"] = grouped["period"].shift(-1)
    performance["next_period_dt"] = grouped["period_dt"].shift(-1)

    performance["month_gap"] = (
        (
            performance["next_period_dt"].dt.year
            - performance["period_dt"].dt.year
        ) * 12
        + (
            performance["next_period_dt"].dt.month
            - performance["period_dt"].dt.month
        )
    )

    performance["valid_next_month"] = (
        performance["month_gap"] == 1
    )

    return performance


def validate_states(performance: pd.DataFrame) -> None:
    unknown_states = performance.loc[
        performance["state"] == "UNKNOWN",
        [
            "current_loan_delinquency_status",
            "zero_balance_code",
        ],
    ]

    if not unknown_states.empty:
        print("Unknown state combinations:")
        print(
            unknown_states.value_counts().head(20)
        )

    print("\nState distribution:")
    print(performance["state"].value_counts())

    print("\nValid one-month transition table:")
    valid_transitions = performance.loc[
        performance["valid_next_month"]
    ]

    transition_table = (
        valid_transitions
        .groupby(["state", "next_state"])
        .size()
        .reset_index(name="n")
        .sort_values(
            ["state", "next_state"],
            ignore_index=True,
        )
    )

    print(transition_table.to_string(index=False))


def main() -> None:
    performance = read_performance()

    print("Performance shape:", performance.shape)

    performance = build_states(performance)
    performance = add_period_datetime(performance)
    performance = build_transitions(performance)

    validate_states(performance)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    performance.to_parquet(
        PANEL_PATH,
        index=False,
    )

    print(f"\nSaved: {PANEL_PATH}")


if __name__ == "__main__":
    main()