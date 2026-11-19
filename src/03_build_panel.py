import pandas as pd

from src.config import (
    ACTIVE_STATES,
    MODEL_STATES,
    PANEL_PATH,
    PERFORMANCE_WITH_STATES_PATH,
)


def build_panel(
    performance: pd.DataFrame,
) -> pd.DataFrame:
    panel = performance.loc[
        performance["valid_next_month"]
        & performance["state"].isin(ACTIVE_STATES)
        & performance["next_state"].isin(MODEL_STATES)
    ].copy()

    return panel


def validate_panel(panel: pd.DataFrame) -> None:
    print("Panel shape:", panel.shape)

    print("\nState distribution:")
    print(panel["state"].value_counts())

    print("\nNext-state distribution:")
    print(panel["next_state"].value_counts())

    print("\nTransition table:")
    transition_table = (
        panel
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
    performance = pd.read_parquet(
        PERFORMANCE_WITH_STATES_PATH,
    )

    print("Performance shape:", performance.shape)

    panel = build_panel(performance)

    validate_panel(panel)

    panel.to_parquet(
        PANEL_PATH,
        index=False,
    )

    print(f"\nSaved: {PANEL_PATH}")


if __name__ == "__main__":
    main()