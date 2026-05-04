import json
from functools import lru_cache

import joblib
import pandas as pd

from backend.config import (
    CELL_LINE_FEATURES_PATH,
    DATASET_CANDIDATES,
    DEPMAP_FEATURES_PATH,
    DRUG_FEATURES_PATH,
    DRUG_NAME_MAP_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_COLUMNS_PICKLE_PATH,
    LABEL_ENCODERS_PATH,
    MODEL_REGISTRY_PATH,
    OLD_DRUG_FEATURES_PATH,
    OLD_FEATURE_COLUMNS_PATH,
    PLACEHOLDER_CELL_LINES,
    PLACEHOLDER_DRUGS,
    REFERENCE_DATASET_CANDIDATES,
    project_relative,
)


class DataAssetError(Exception):
    """Raised when configured data assets cannot support a requested operation."""


DRUG1_ALIASES = ("NSC1", "drug1_id", "Drug1", "drug1", "nsc1", "item_1", "item1", "compound_1")
DRUG2_ALIASES = ("NSC2", "drug2_id", "Drug2", "drug2", "nsc2", "item_2", "item2", "compound_2")
CELL_ALIASES = ("CELLNAME", "cell_line", "cellLine", "cellname", "cell", "cell line", "context", "sample", "category")
SINGLE_DRUG_ALIASES = ("NSC", "nsc", "drug_id", "item_id", "compound_id", "id")
DRUG_NAME_ALIASES = ("drug_name", "name", "label", "compound_name", "item_name")
SCORE_ALIASES = (
    "COMBOSCORE",
    "ComboScore",
    "combo_score",
    "synergy_score",
    "score",
    "target",
    "final_predicted_COMBOSCORE",
)
CELL_LINE_COLUMN = "cell line"
DEPMAP_ID_COLUMN = "depmap_id"
DRUG_ID_COLUMN = "drug_id"
PAIR_IDENTIFIER_COLUMNS = {"cell line", "depmap_id", "drug_combination"}
DEFAULT_FEATURE_VALUES = {
    # The organized v3 asset bundle has this feature in the trained model order
    # but not in the lookup CSV. Keep the default explicit and easy to revisit.
    "seneitive10": 0.0,
}


def _normalize_name(value):
    return str(value or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _clean_identifier(value):
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _find_column(frame, aliases):
    if frame is None or frame.empty:
        return None

    normalized = {_normalize_name(column): column for column in frame.columns}
    for alias in aliases:
        match = normalized.get(_normalize_name(alias))
        if match:
            return match
    return None


def get_dataset_path():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    return None


def organized_lookup_assets_available():
    return CELL_LINE_FEATURES_PATH.exists() and DRUG_FEATURES_PATH.exists()


def old_feature_assets_available():
    return OLD_DRUG_FEATURES_PATH.exists() and OLD_FEATURE_COLUMNS_PATH.exists()


@lru_cache(maxsize=1)
def load_dataset():
    path = get_dataset_path()
    if not path:
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DataAssetError(f"Could not read {project_relative(path)}: {exc}") from exc


@lru_cache(maxsize=1)
def load_drug_name_map():
    if not DRUG_NAME_MAP_PATH.exists():
        return None
    return _read_csv(DRUG_NAME_MAP_PATH)


@lru_cache(maxsize=1)
def load_drug_features():
    if not DRUG_FEATURES_PATH.exists():
        return None
    return _read_csv(DRUG_FEATURES_PATH)


@lru_cache(maxsize=1)
def load_old_drug_features():
    if not OLD_DRUG_FEATURES_PATH.exists():
        return None
    frame = _read_csv(OLD_DRUG_FEATURES_PATH)
    id_column = _find_column(frame, ("NSC", "nsc"))
    if not id_column:
        raise DataAssetError(f"Old drug feature file must include an NSC column: {project_relative(OLD_DRUG_FEATURES_PATH)}.")
    if id_column != "NSC":
        frame = frame.rename(columns={id_column: "NSC"})
    frame["NSC"] = pd.to_numeric(frame["NSC"], errors="raise").astype(int)
    return frame


@lru_cache(maxsize=1)
def load_cell_line_features():
    if not CELL_LINE_FEATURES_PATH.exists():
        return None
    return _read_csv(CELL_LINE_FEATURES_PATH)


@lru_cache(maxsize=1)
def load_depmap_features():
    if not DEPMAP_FEATURES_PATH.exists():
        return None
    return _read_csv(DEPMAP_FEATURES_PATH)


@lru_cache(maxsize=None)
def get_feature_columns(feature_mode="organized_lookup"):
    if feature_mode == "old_d1_d2":
        return _load_feature_columns_from_path(OLD_FEATURE_COLUMNS_PATH, None)

    return _load_feature_columns_from_path(FEATURE_COLUMNS_PATH, FEATURE_COLUMNS_PICKLE_PATH)


def _load_feature_columns_from_path(json_path, pickle_path=None):
    payload = None

    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DataAssetError(f"Could not read {project_relative(json_path)}: {exc}") from exc
    elif pickle_path and pickle_path.exists():
        try:
            payload = joblib.load(pickle_path)
        except Exception as exc:
            raise DataAssetError(f"Could not read {project_relative(pickle_path)}: {exc}") from exc
    else:
        return []

    if isinstance(payload, list):
        columns = payload
    elif isinstance(payload, tuple):
        columns = list(payload)
    elif isinstance(payload, dict):
        columns = (
            payload.get("feature_columns")
            or payload.get("features")
            or payload.get("columns")
            or payload.get("feature_names")
            or []
        )
    else:
        columns = []

    return [str(column) for column in columns if str(column).strip()]


def get_identifier_columns(frame=None):
    table = frame if frame is not None else load_dataset()
    return {
        "nsc1": _find_column(table, DRUG1_ALIASES),
        "nsc2": _find_column(table, DRUG2_ALIASES),
        "cell": _find_column(table, CELL_ALIASES),
        "single_drug": _find_column(table, SINGLE_DRUG_ALIASES),
        "drug_name": _find_column(table, DRUG_NAME_ALIASES),
    }


def get_drugs():
    mapped = _drugs_from_name_map()
    if mapped:
        return mapped

    drug_features = load_drug_features()
    if drug_features is not None and DRUG_ID_COLUMN in drug_features.columns:
        ids = [_clean_identifier(value) for value in drug_features[DRUG_ID_COLUMN].dropna().astype(str)]
        return [
            {"id": item_id, "name": f"NSC {item_id}" if item_id.isdigit() else f"Item {item_id}"}
            for item_id in sorted(set(filter(None, ids)), key=_sort_key)
        ]

    old_drug_features = load_old_drug_features()
    if old_drug_features is not None and "NSC" in old_drug_features.columns:
        ids = [_clean_identifier(value) for value in old_drug_features["NSC"].dropna().astype(str)]
        return [
            {"id": item_id, "name": f"NSC {item_id}" if item_id.isdigit() else f"Item {item_id}"}
            for item_id in sorted(set(filter(None, ids)), key=_sort_key)
        ]

    table = load_dataset()
    if table is None or table.empty:
        return list(PLACEHOLDER_DRUGS)

    columns = get_identifier_columns(table)
    ids_to_names = {}

    for column in (columns["nsc1"], columns["nsc2"], columns["single_drug"]):
        if not column:
            continue
        for value in table[column].dropna().astype(str):
            item_id = _clean_identifier(value)
            if item_id:
                ids_to_names.setdefault(item_id, f"NSC {item_id}" if item_id.isdigit() else f"Item {item_id}")

    if columns["single_drug"] and columns["drug_name"]:
        for _, row in table[[columns["single_drug"], columns["drug_name"]]].dropna(how="all").iterrows():
            item_id = _clean_identifier(row.get(columns["single_drug"], ""))
            name = str(row.get(columns["drug_name"], "")).strip()
            if item_id and name:
                ids_to_names[item_id] = name

    if not ids_to_names:
        return list(PLACEHOLDER_DRUGS)

    return [
        {"id": item_id, "name": ids_to_names[item_id]}
        for item_id in sorted(ids_to_names, key=_sort_key)
    ]


def get_cell_lines():
    cell_features = load_cell_line_features()
    if cell_features is not None and CELL_LINE_COLUMN in cell_features.columns:
        values = [
            str(value).strip()
            for value in cell_features[CELL_LINE_COLUMN].dropna().astype(str).unique()
            if str(value).strip()
        ]
        return sorted(values) or list(PLACEHOLDER_CELL_LINES)

    table = load_dataset()
    if table is None or table.empty:
        return list(PLACEHOLDER_CELL_LINES)

    cell_column = get_identifier_columns(table)["cell"]
    if not cell_column:
        return list(PLACEHOLDER_CELL_LINES)

    values = [
        str(value).strip()
        for value in table[cell_column].dropna().astype(str).unique()
        if str(value).strip()
    ]
    return sorted(values) or list(PLACEHOLDER_CELL_LINES)


def build_feature_frame(nsc1, nsc2, cellname, feature_columns, feature_mode="organized_lookup"):
    if feature_mode == "old_d1_d2" or _is_old_d1_d2_feature_layout(feature_columns):
        return _build_old_d1_d2_feature_frame(nsc1, nsc2, feature_columns)
    if organized_lookup_assets_available():
        return _build_lookup_feature_frame(nsc1, nsc2, cellname, feature_columns)
    return _build_dataset_feature_frame(nsc1, nsc2, cellname, feature_columns)


def prediction_asset_errors():
    errors = []

    old_assets = old_feature_assets_available()
    if not organized_lookup_assets_available() and not old_assets and get_dataset_path() is None:
        errors.append("Prediction data is missing: add organized lookup CSV files, old drug_features.csv, or data/input_data.csv.")
    if not DRUG_NAME_MAP_PATH.exists() and not DRUG_FEATURES_PATH.exists() and not OLD_DRUG_FEATURES_PATH.exists():
        errors.append("Drug lookup is missing: add data/drug_name_id_map.csv, data/drug_fingerprints_lookup.csv, or data/drug_features.csv.")
    if not CELL_LINE_FEATURES_PATH.exists() and not MODEL_REGISTRY_PATH.exists() and get_dataset_path() is None:
        errors.append("Cell-line/context lookup is missing: add data/cell_line_features_lookup.csv or results/step6_final_model_registry.csv.")
    if (
        not FEATURE_COLUMNS_PATH.exists()
        and not FEATURE_COLUMNS_PICKLE_PATH.exists()
        and not OLD_FEATURE_COLUMNS_PATH.exists()
    ):
        errors.append("Feature order is missing: add data/feature_columns.json or data/step6_final_model_feature_columns.json.")

    return errors


def asset_messages():
    messages = []

    for label, path in (
        ("Drug names", DRUG_NAME_MAP_PATH),
        ("Drug fingerprints", DRUG_FEATURES_PATH),
        ("Cell-line features", CELL_LINE_FEATURES_PATH),
        ("DepMap features", DEPMAP_FEATURES_PATH),
        ("Label encoders", LABEL_ENCODERS_PATH),
        ("Feature order JSON", FEATURE_COLUMNS_PATH),
        ("Feature order pickle", FEATURE_COLUMNS_PICKLE_PATH),
        ("Old 263-column drug features", OLD_DRUG_FEATURES_PATH),
        ("Old 526-feature order JSON", OLD_FEATURE_COLUMNS_PATH),
    ):
        if path.exists():
            messages.append(f"{label}: {project_relative(path)}")

    dataset_path = get_dataset_path()
    if dataset_path and dataset_path not in {CELL_LINE_FEATURES_PATH}:
        messages.append(f"Generic dataset source: {project_relative(dataset_path)}")

    for path in REFERENCE_DATASET_CANDIDATES:
        if path.exists():
            messages.append(f"Dataset reference source: {project_relative(path)}")

    if not messages:
        messages.append("No data assets are configured yet.")

    return messages


def _is_old_d1_d2_feature_layout(feature_columns):
    if len(feature_columns) != 526:
        return False
    return all(str(column).startswith(("D1_", "D2_")) for column in feature_columns)


def _build_old_d1_d2_feature_frame(nsc1, nsc2, feature_columns):
    table = load_old_drug_features()
    if table is None or table.empty:
        raise DataAssetError(
            "Old 526-feature prediction mode needs data/drug_features.csv with an NSC column."
        )

    if not feature_columns:
        raise DataAssetError(
            "Old 526-feature prediction mode needs data/step6_final_model_feature_columns.json."
        )

    if len(feature_columns) != 526:
        raise DataAssetError(
            f"Old D1_/D2_ feature mode expected 526 feature columns, found {len(feature_columns)}."
        )

    drug1_row = _get_old_drug_row(table, nsc1)
    drug2_row = _get_old_drug_row(table, nsc2)
    base_feature_columns = [column for column in table.columns if column != "NSC"]

    if len(base_feature_columns) != 263:
        raise DataAssetError(
            f"Old drug feature table has {len(base_feature_columns)} feature columns, expected 263."
        )

    values = {}
    for column in base_feature_columns:
        values[f"D1_{column}"] = drug1_row[column]
        values[f"D2_{column}"] = drug2_row[column]

    missing = [column for column in feature_columns if column not in values]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise DataAssetError(f"Old 526-feature vector is missing required columns: {preview}{suffix}.")

    frame = pd.DataFrame([{column: values[column] for column in feature_columns}], columns=feature_columns)
    try:
        return frame.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise DataAssetError(f"Old 526-feature vector contains non-numeric values: {exc}") from exc


def _get_old_drug_row(table, nsc):
    try:
        requested = int(float(str(nsc).strip()))
    except (TypeError, ValueError) as exc:
        raise DataAssetError(f"NSC value '{nsc}' must be numeric for old 526-feature mode.") from exc

    matches = table[table["NSC"] == requested]
    if matches.empty:
        examples = ", ".join(table["NSC"].head(8).astype(str).tolist())
        raise DataAssetError(f"NSC {requested} was not found in data/drug_features.csv. Available examples: {examples}.")
    return matches.iloc[0]


def _build_lookup_feature_frame(nsc1, nsc2, cellname, feature_columns):
    cell_row = _get_cell_row(cellname)
    drug_a = _get_drug_feature_row(nsc1)
    drug_b = _get_drug_feature_row(nsc2)

    values = {}
    missing = []

    for feature in feature_columns:
        if feature.endswith("_sum"):
            base = feature[:-4]
            values[feature] = _fingerprint_value(drug_a, base) + _fingerprint_value(drug_b, base)
        elif feature.endswith("_diff"):
            base = feature[:-5]
            values[feature] = _fingerprint_value(drug_a, base) - _fingerprint_value(drug_b, base)
        elif feature.endswith("_mul"):
            base = feature[:-4]
            values[feature] = _fingerprint_value(drug_a, base) * _fingerprint_value(drug_b, base)
        elif feature.endswith("_A"):
            values[feature] = _fingerprint_value(drug_a, feature[:-2])
        elif feature.endswith("_B"):
            values[feature] = _fingerprint_value(drug_b, feature[:-2])
        elif feature in cell_row.index:
            values[feature] = cell_row[feature]
        elif feature in DEFAULT_FEATURE_VALUES:
            values[feature] = DEFAULT_FEATURE_VALUES[feature]
        else:
            missing.append(feature)

    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise DataAssetError(
            f"The organized lookup assets cannot build these configured features: {preview}{suffix}."
        )

    frame = pd.DataFrame([values], columns=feature_columns)
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _build_dataset_feature_frame(nsc1, nsc2, cellname, feature_columns):
    table = load_dataset()
    if table is None or table.empty:
        raise DataAssetError(
            "Feature lookup data is missing. Add organized lookup files in data/ or add "
            "data/input_data.csv with NSC1, NSC2, CELLNAME, and configured feature columns."
        )

    columns = get_identifier_columns(table)
    if not columns["nsc1"] or not columns["nsc2"]:
        raise DataAssetError(
            "The dataset must include pair identifier columns such as NSC1 and NSC2 "
            "so the backend can find the feature row for a requested pair."
        )

    missing_features = [column for column in feature_columns if column not in table.columns]
    if missing_features:
        preview = ", ".join(missing_features[:8])
        suffix = "..." if len(missing_features) > 8 else ""
        raise DataAssetError(
            f"The dataset is missing configured feature columns: {preview}{suffix}. "
            "Update data/feature_columns.json or add these columns to the dataset."
        )

    mask = (
        table[columns["nsc1"]].astype(str).str.strip().eq(str(nsc1).strip())
        & table[columns["nsc2"]].astype(str).str.strip().eq(str(nsc2).strip())
    )

    if columns["cell"]:
        mask = mask & table[columns["cell"]].astype(str).str.strip().eq(str(cellname).strip())

    matches = table.loc[mask]
    if matches.empty:
        context = f" in context '{cellname}'" if cellname else ""
        raise DataAssetError(f"No feature row was found for {nsc1} + {nsc2}{context}.")

    return matches.iloc[[0]][feature_columns].reset_index(drop=True)


def _drugs_from_name_map():
    table = load_drug_name_map()
    if table is None or table.empty:
        return []

    id_column = _find_column(table, SINGLE_DRUG_ALIASES) or table.columns[0]
    name_column = _find_column(table, DRUG_NAME_ALIASES)
    ids_to_names = {}

    for _, row in table.iterrows():
        item_id = _clean_identifier(row.get(id_column, ""))
        if not item_id:
            continue
        name = str(row.get(name_column, "")).strip() if name_column else ""
        ids_to_names[item_id] = name or (f"NSC {item_id}" if item_id.isdigit() else f"Item {item_id}")

    return [
        {"id": item_id, "name": ids_to_names[item_id]}
        for item_id in sorted(ids_to_names, key=_sort_key)
    ]


def _get_cell_row(cellname):
    table = load_cell_line_features()
    if table is None or table.empty or CELL_LINE_COLUMN not in table.columns:
        raise DataAssetError(f"Cell-line feature file is missing or invalid: {project_relative(CELL_LINE_FEATURES_PATH)}.")

    requested = str(cellname or "").strip()
    lowered = requested.lower()
    mask = table[CELL_LINE_COLUMN].astype(str).str.strip().str.lower().eq(lowered)

    if not mask.any() and DEPMAP_ID_COLUMN in table.columns:
        mask = table[DEPMAP_ID_COLUMN].astype(str).str.strip().str.lower().eq(lowered)

    if not mask.any():
        examples = ", ".join(get_cell_lines()[:8])
        raise DataAssetError(f"Unknown cell line/context '{requested}'. Available examples: {examples}.")

    row = table.loc[mask].iloc[0].copy()
    depmap_id = row.get(DEPMAP_ID_COLUMN)
    depmap_row = _depmap_row(depmap_id)
    if depmap_row is not None:
        for column, value in depmap_row.items():
            row[column] = value
    return row


def _get_drug_feature_row(drug_id):
    table = load_drug_features()
    if table is None or table.empty or DRUG_ID_COLUMN not in table.columns:
        raise DataAssetError(f"Drug fingerprint file is missing or invalid: {project_relative(DRUG_FEATURES_PATH)}.")

    requested = _clean_identifier(drug_id)
    mask = table[DRUG_ID_COLUMN].astype(str).map(_clean_identifier).eq(requested)
    if not mask.any():
        examples = ", ".join([drug["id"] for drug in get_drugs()[:8]])
        raise DataAssetError(f"Unknown drug/item ID '{requested}'. Available examples: {examples}.")

    return table.loc[mask].iloc[0]


def _depmap_row(depmap_id):
    if depmap_id is None or pd.isna(depmap_id):
        return None

    table = load_depmap_features()
    if table is None or table.empty or "ModelID" not in table.columns:
        return None

    requested = str(depmap_id).strip().lower()
    mask = table["ModelID"].astype(str).str.strip().str.lower().eq(requested)
    if not mask.any():
        return None

    return table.loc[mask].iloc[0]


def lookup_dataset_reference(nsc1, nsc2, cellname):
    """Find an exact observed ComboScore row for transparent debugging only.

    The returned reference is never used as the model prediction. It is used by
    prediction_service to calculate optional error metadata when a project
    dataset such as data/model_matrix.csv is present.
    """
    for path in REFERENCE_DATASET_CANDIDATES:
        table = _load_reference_dataset(path)
        if table is None or table.empty:
            continue

        columns = get_identifier_columns(table)
        score_column = _find_column(table, SCORE_ALIASES)
        if not columns["nsc1"] or not columns["nsc2"] or not score_column:
            continue

        match = _match_reference_rows(table, columns, nsc1, nsc2, cellname)
        direction = "forward"
        if match.empty:
            match = _match_reference_rows(table, columns, nsc2, nsc1, cellname)
            direction = "reverse"

        if match.empty:
            continue

        value = pd.to_numeric(match.iloc[0][score_column], errors="coerce")
        if pd.isna(value):
            continue

        return {
            "dataset_reference_checked": True,
            "dataset_reference_available": True,
            "dataset_reference_COMBOSCORE": float(value),
            "dataset_reference_source": project_relative(path),
            "dataset_reference_direction": direction,
        }

    return {
        "dataset_reference_checked": True,
        "dataset_reference_available": False,
        "dataset_reference_COMBOSCORE": None,
        "dataset_reference_source": "",
        "dataset_reference_direction": "",
    }


@lru_cache(maxsize=None)
def _load_reference_dataset(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _match_reference_rows(table, columns, nsc1, nsc2, cellname):
    nsc1_requested = _clean_identifier(nsc1)
    nsc2_requested = _clean_identifier(nsc2)
    mask = (
        table[columns["nsc1"]].astype(str).map(_clean_identifier).eq(nsc1_requested)
        & table[columns["nsc2"]].astype(str).map(_clean_identifier).eq(nsc2_requested)
    )
    if columns["cell"]:
        requested_cell = str(cellname or "").strip().lower()
        mask = mask & table[columns["cell"]].astype(str).str.strip().str.lower().eq(requested_cell)
    return table.loc[mask]


def _fingerprint_value(row, feature):
    if feature not in row.index:
        raise DataAssetError(f"Drug fingerprint feature '{feature}' is missing from {project_relative(DRUG_FEATURES_PATH)}.")
    value = row[feature]
    if pd.isna(value):
        return 0.0
    return float(value)


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DataAssetError(f"Could not read {project_relative(path)}: {exc}") from exc


def _sort_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text.lower())
