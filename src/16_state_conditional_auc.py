import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    MODELS_DIR,
    MODELING_DATASET_WITH_LAGS_PATH,
    REPORTS_TABLES_DIR,
    TARGET_COLUMN,
)


def load_model():
    return joblib.load(
        MODELS_DIR / "baselinewithlags.joblib"
    )


def calculate_state_conditional_auc(
    model,
    test: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = model.predict_proba(
        test[
            model.named_steps[
                "preprocessor"
            ].feature_names_in_
        ]
    )

    classes = model.named_steps[
        "classifier"
    ].classes_

    probability_frame = pd.DataFrame(
        probabilities,
        columns=classes,
        index=test.index,
    )

    results = []

    for current_state in test["state"].unique():
        state_data = test.loc[
            test["state"] == current_state
        ]

        state_probabilities = probability_frame.loc[
            state_data.index
        ]

        for next_state in classes:
            target = (
                state_data[TARGET_COLUMN] == next_state
            ).astype(int)

            if target.nunique() < 2:
                continue

            auc = roc_auc_score(
                target,
                state_probabilities[next_state],
            )

            results.append(
                {
                    "current_state": current_state,
                    "next_state": next_state,
                    "observations": len(state_data),
                    "positive_rate": target.mean(),
                    "auc": auc,
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    model = load_model()

    test = pd.read_parquet(
        MODELING_DATASET_WITH_LAGS_PATH,
    )

    test = test.loc[
        test["split"] == "test"
    ]

    results = calculate_state_conditional_auc(
        model,
        test,
    )

    results = results.sort_values(
        ["current_state", "next_state"],
        ignore_index=True,
    )

    print(results.to_string(index=False))

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORTS_TABLES_DIR / "state_conditional_auc.csv",
        index=False,
    )

    print(
        "\nSaved: "
        f"{REPORTS_TABLES_DIR / 'state_conditional_auc.csv'}"
    )


if __name__ == "__main__":
    main()