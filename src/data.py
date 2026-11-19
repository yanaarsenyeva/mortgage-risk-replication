from pathlib import Path

import pandas as pd

from src.config import (
    ORIGINATION_PATH,
    PERFORMANCE_PATH,
)


ORIGINATION_COLUMNS = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer",
    "maturity_date",
    "msa",
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
    "loan_id",
    "loan_purpose",
    "original_loan_term",
    "num_borrowers",
    "seller_name",
    "super_conforming_flag",
    "pre_harp_loan_id",
    "special_eligibility_program",
    "harp_indicator",
    "property_valuation_method",
    "interest_only",
    "vantage_score",
]

PERFORMANCE_COLUMNS = [
    "loan_id",
    "period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "due_date_last_paid_installment",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "current_period_modification_costs",
    "current_interest_bearing_upb",
    "mi_cancellation_indicator",
    "servicer_name",
    "bankruptcy_cramdown_costs",
]


def read_pipe_file(
    path: Path,
    columns: list[str],
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )


def read_origination() -> pd.DataFrame:
    return read_pipe_file(
        ORIGINATION_PATH,
        ORIGINATION_COLUMNS,
    )


def read_performance() -> pd.DataFrame:
    return read_pipe_file(
        PERFORMANCE_PATH,
        PERFORMANCE_COLUMNS,
    )