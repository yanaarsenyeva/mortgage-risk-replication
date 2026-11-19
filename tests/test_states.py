from src.states import normalize_state


def test_current_state():
    assert normalize_state("00", "") == "CURRENT"
    assert normalize_state("0.00", "") == "CURRENT"


def test_delinquency_states():
    assert normalize_state("01", "") == "D30"
    assert normalize_state("1.00", "") == "D30"
    assert normalize_state("02", "") == "D60"
    assert normalize_state("2.00", "") == "D60"
    assert normalize_state("03", "") == "D90_PLUS"
    assert normalize_state("3.00", "") == "D90_PLUS"
    assert normalize_state("10", "") == "D90_PLUS"
    assert normalize_state("10.00", "") == "D90_PLUS"
    assert normalize_state("99", "") == "D90_PLUS"


def test_zero_balance_has_priority():
    assert normalize_state("00", "01") == "PREPAID"
    assert normalize_state("00", "09") == "REO_DISPOSITION"
    assert normalize_state("00", "02") == "TERMINATED_OTHER"


def test_reo_acquisition_and_disposition_are_separate():
    assert normalize_state("RA", "") == "REO_ACQUISITION"
    assert normalize_state("", "09") == "REO_DISPOSITION"
    assert normalize_state("", "9.00") == "REO_DISPOSITION"


def test_prepayment():
    assert normalize_state("", "01") == "PREPAID"
    assert normalize_state("", "1.00") == "PREPAID"


def test_other_termination():
    assert normalize_state("", "02") == "TERMINATED_OTHER"
    assert normalize_state("", "2.00") == "TERMINATED_OTHER"
    assert normalize_state("", "03") == "TERMINATED_OTHER"
    assert normalize_state("", "3.00") == "TERMINATED_OTHER"
    assert normalize_state("", "15") == "TERMINATED_OTHER"
    assert normalize_state("", "15.00") == "TERMINATED_OTHER"
    assert normalize_state("", "16") == "TERMINATED_OTHER"
    assert normalize_state("", "16.00") == "TERMINATED_OTHER"
    assert normalize_state("", "96") == "TERMINATED_OTHER"
    assert normalize_state("", "96.00") == "TERMINATED_OTHER"


def test_unknown_state():
    assert normalize_state("ZZ", "ZZ") == "UNKNOWN"