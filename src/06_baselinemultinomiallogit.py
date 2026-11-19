from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    MODELS_DIR,
    MODELING_DATASET_PATH,
    REPORTS_TABLES_DIR,
    TARGET_COLUMN,
)


NUMERIC_FEATURES = [
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
    "loan_age_months",
    "months_to_maturity",
    "current_rate_spread",
    "unpaid_principal_ratio",
    "prepayment_incentive",
]

CATEGORICAL_FEATURES = [
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

SENTINEL_VALUES = {
    "credit_score": [9999],
    "original_cltv": [999],
    "original_dti": [999],
    "original_ltv": [999],
}

TRAIN_SAMPLE_SIZE = 200000


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(
        MODELING_DATASET_PATH,
    )


def clean_sentinels(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    panel = panel.copy()

    for column, sentinels in SENTINEL_VALUES.items():
        panel[column] = panel[column].replace(
            sentinels,
            np.nan,
        )

    return panel


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def build_model() -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=100,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def evaluate_split(
    model: Pipeline,
    data: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    features = data[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    target = data[TARGET_COLUMN]

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    print(f"\n{split_name.capitalize()} report:")
    print(
        classification_report(
            target,
            predictions,
            zero_division=0,
        )
    )

    print(
        f"{split_name.capitalize()} log loss:",
        log_loss(
            target,
            probabilities,
            labels=model.named_steps[
                "classifier"
            ].classes_,
        ),
    )

    try:
        roc_auc = roc_auc_score(
            target,
            probabilities,
            multi_class="ovr",
            average="macro",
        )

        print(
            f"{split_name.capitalize()} macro ROC-AUC:",
            roc_auc,
        )
    except ValueError:
        print(
            f"{split_name.capitalize()} macro ROC-AUC: not available"
        )

    print(
        f"\n{split_name.capitalize()} confusion matrix:"
    )

    confusion = pd.DataFrame(
        confusion_matrix(
            target,
            predictions,
            labels=model.named_steps[
                "classifier"
            ].classes_,
        ),
        index=[
            f"true_{class_name}"
            for class_name in model.named_steps[
                "classifier"
            ].classes_
        ],
        columns=[
            f"pred_{class_name}"
            for class_name in model.named_steps[
                "classifier"
            ].classes_
        ],
    )

    print(confusion)

    return confusion


def main() -> None:
    panel = load_dataset()
    panel = clean_sentinels(panel)

    train = panel.loc[
        panel["split"] == "train"
    ]

    validation = panel.loc[
        panel["split"] == "validation"
    ]

    test = panel.loc[
        panel["split"] == "test"
    ]

    print("Train shape:", train.shape)
    print("Validation shape:", validation.shape)
    print("Test shape:", test.shape)

    train_sample = train.sample(
        n=TRAIN_SAMPLE_SIZE,
        random_state=42,
    )

    print("Train sample shape:", train_sample.shape)

    model = build_model()

    print("\nFitting multinomial logit on train sample...")

    model.fit(
        train_sample[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ],
        train_sample[TARGET_COLUMN],
    )

    validation_confusion = evaluate_split(
        model,
        validation,
        "validation",
    )

    test_confusion = evaluate_split(
        model,
        test,
        "test",
    )

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_confusion.to_csv(
        REPORTS_TABLES_DIR
        / "baselinemultinomiallogitconfusionmatrix.csv"
    )

    test_confusion.to_csv(
        REPORTS_TABLES_DIR
        / "baselinemultinomiallogittestconfusionmatrix.csv"
    )

    joblib.dump(
        model,
        MODELS_DIR / "baselinemultinomiallogit.joblib",
    )

    print(
        "\nSaved: "
        f"{REPORTS_TABLES_DIR / 'baselinemultinomiallogitconfusionmatrix.csv'}"
    )

    print(
        "Saved: "
        f"{REPORTS_TABLES_DIR / 'baselinemultinomiallogittestconfusionmatrix.csv'}"
    )

    print(
        "Saved: "
        f"{MODELS_DIR / 'baselinemultinomiallogit.joblib'}"
    )


if __name__ == "__main__":
    main()