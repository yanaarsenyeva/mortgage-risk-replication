import pandas as pd

from src.config import (
    MODELING_DATASET_WITH_LAGS_PATH,
    REPORTS_TABLES_DIR,
)


def build_transition_matrix(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    transition_counts = (
        panel
        .groupby(["state", "next_state"])
        .size()
        .reset_index(name="n")
    )

    transition_counts["row_total"] = (
        transition_counts
        .groupby("state")["n"]
        .transform("sum")
    )

    transition_counts["probability"] = (
        transition_counts["n"]
        / transition_counts["row_total"]
    )

    transition_matrix = transition_counts.pivot(
        index="state",
        columns="next_state",
        values="probability",
    ).fillna(0)

    return transition_matrix


def main() -> None:
    panel = pd.read_parquet(
        MODELING_DATASET_WITH_LAGS_PATH,
    )

    transition_matrix = build_transition_matrix(panel)

    print("Empirical one-month transition matrix:")
    print(
        transition_matrix.round(4).to_string()
    )

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    transition_matrix.to_csv(
        REPORTS_TABLES_DIR / "empirical_transition_matrix.csv"
    )

    print(
        "\nSaved: "
        f"{REPORTS_TABLES_DIR / 'empirical_transition_matrix.csv'}"
    )


if __name__ == "__main__":
    main()