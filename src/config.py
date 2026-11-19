from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = PROJECT_ROOT / "data"


RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


RAW_SAMPLE_2010_DIR = RAW_DATA_DIR / "sample_2010"


ORIGINATION_PATH = RAW_SAMPLE_2010_DIR / "sample_orig_2010.txt"
PERFORMANCE_PATH = RAW_SAMPLE_2010_DIR / "sample_perf_2010.txt"


PERFORMANCE_WITH_STATES_PATH = (
    PROCESSED_DATA_DIR / "performance_with_states.parquet"
)


PANEL_PATH = PROCESSED_DATA_DIR / "panel.parquet"


PANEL_WITH_ORIGINATION_PATH = (
    PROCESSED_DATA_DIR / "panel_with_origination.parquet"
)


MODELING_DATASET_PATH = (
    PROCESSED_DATA_DIR / "modeling_dataset.parquet"
)


MODELING_DATASET_WITH_LAGS_PATH = (
    PROCESSED_DATA_DIR / "modeling_dataset_with_lags.parquet"
)


MACRO_DATA_DIR = DATA_DIR / "raw"

MORTGAGE30US_PATH = (
    MACRO_DATA_DIR / "freddiemac_mortgage30us.csv"
)

MODELING_DATASET_WITH_LAGS_AND_MACRO_PATH = (
    PROCESSED_DATA_DIR / "modeling_dataset_with_lags_and_macro.parquet"
)


REPORTS_TABLES_DIR = REPORTS_DIR / "tables"


ACTIVE_STATES = [
    "CURRENT",
    "D30",
    "D60",
    "D90_PLUS",
]


TERMINAL_STATES = [
    "PREPAID",
    "TERMINATED_OTHER",
    "REO_ACQUISITION",
    "REO_DISPOSITION",
]


MODEL_STATES = [
    "CURRENT",
    "D30",
    "D60",
    "D90_PLUS",
    "PREPAID",
    "TERMINATED_OTHER",
]


TARGET_COLUMN = "next_state"


TIME_COLUMN = "period"


ID_COLUMN = "loan_id"


TEST_SIZE = 0.2


VALIDATION_SIZE = 0.1


RANDOM_STATE = 42