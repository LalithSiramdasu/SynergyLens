from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INPUT_DATA_PATH = DATA_DIR / "input_data.csv"
METADATA_PATH = DATA_DIR / "metadata.csv"
FEATURE_COLUMNS_PATH = DATA_DIR / "feature_columns.json"
FEATURE_COLUMNS_PICKLE_PATH = DATA_DIR / "feature_columns.pkl"
DRUG_NAME_MAP_PATH = DATA_DIR / "drug_name_id_map.csv"
DRUG_FEATURES_PATH = DATA_DIR / "drug_fingerprints_lookup.csv"
CELL_LINE_FEATURES_PATH = DATA_DIR / "cell_line_features_lookup.csv"
DEPMAP_FEATURES_PATH = DATA_DIR / "depmap_features_lookup.csv"
LABEL_ENCODERS_PATH = DATA_DIR / "depmap_label_encoders.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"

DATASET_CANDIDATES = (INPUT_DATA_PATH, METADATA_PATH, CELL_LINE_FEATURES_PATH)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

MODEL_DISPLAY_NAME = "SingleModel"
MODEL_DISPLAY_PATH = "models/model.pkl"

PLACEHOLDER_DRUGS = [
    {"id": "1001", "name": "Demo Item 1001"},
    {"id": "1002", "name": "Demo Item 1002"},
    {"id": "1003", "name": "Demo Item 1003"},
]
PLACEHOLDER_CELL_LINES = ["General"]


def ensure_directories():
    for directory in (DATA_DIR, MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def project_relative(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
