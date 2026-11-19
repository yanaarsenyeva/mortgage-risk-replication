from src.data import (
    read_origination,
    read_performance,
)


def test_origination_shape():
    origination = read_origination()

    assert origination.shape[0] == 50000
    assert origination.shape[1] == 31

    assert "loan_id" in origination.columns
    assert "credit_score" in origination.columns
    assert "property_state" in origination.columns
    assert "postal_code" in origination.columns
    assert "vantage_score" in origination.columns


def test_performance_shape():
    performance = read_performance()

    assert "loan_id" in performance.columns
    assert "period" in performance.columns
    assert "current_loan_delinquency_status" in performance.columns
    assert "zero_balance_code" in performance.columns