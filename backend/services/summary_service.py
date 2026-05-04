import pandas as pd

from backend.config import (
    FEATURE_COLUMNS_PATH,
    FEATURE_COLUMNS_PICKLE_PATH,
    MODEL_DISPLAY_NAME,
    MODEL_DISPLAY_PATH,
    MODEL_REGISTRY_PATH,
    MODEL_PATH,
    OLD_DRUG_FEATURES_PATH,
    OLD_FEATURE_COLUMNS_PATH,
    STEP5_AVERAGE_PERFORMANCE_PATH,
    STEP6_MODEL_SUMMARY_PATH,
    project_relative,
)
from backend.services import data_service, model_service


def health_summary():
    errors = []

    try:
        drugs = data_service.get_drugs()
        cell_lines = data_service.get_cell_lines()
    except Exception as exc:
        drugs = []
        cell_lines = []
        errors.append(str(exc))

    try:
        feature_columns = data_service.get_feature_columns()
    except Exception as exc:
        feature_columns = []
        errors.append(str(exc))

    errors.extend(data_service.prediction_asset_errors())
    if not MODEL_PATH.exists():
        try:
            registry = model_service.load_model_registry()
        except Exception as exc:
            registry = None
            errors.append(str(exc))
        if registry is None or registry.empty:
            errors.append("Model file is missing: add models/model.pkl or configure old final_step6 model assets.")

    cell_line_model_count = _safe_cell_line_model_count(errors)
    return {
        "status": "success" if not errors else "error",
        "available_drugs": len(drugs),
        "available_cell_lines": len(cell_lines),
        "feature_column_count": len(feature_columns),
        "model_count": cell_line_model_count or (1 if model_service.single_model_exists() else 0),
        "single_model_available": model_service.single_model_exists(),
        "cell_line_specific_model_count": cell_line_model_count,
        "errors": errors,
    }


def model_performance_summary():
    drugs = data_service.get_drugs()
    cell_lines = data_service.get_cell_lines()
    feature_columns = _safe_feature_columns()
    total_models = _safe_cell_line_model_count() or (1 if model_service.single_model_exists() else 0)

    return {
        "status": "success",
        "assets": {
            "total_cell_lines": len(cell_lines),
            "total_drugs": len(drugs),
            "feature_vector": len(feature_columns),
            "final_model_count": total_models,
        },
        "model_summary": {
            "model_types": _model_types(),
            "count_per_model_type": _model_type_counts(),
            "total_models": total_models,
        },
        "performance": _performance_payload(),
        "explanation": (
            "This project uses the single deployed model when models/model.pkl is present. "
            "If old final_step6 registry/model assets are added, matching cell lines use their specific model."
        ),
    }


def system_summary():
    health = health_summary()
    return {
        "status": "success",
        "project_name": "SynergyLens",
        "summary": "Clean Flask backend for the extracted SynergyLens UI and a new single-model project.",
        "model_strategy": "single deployed model with optional old cell-line-specific compatibility",
        "model_path": MODEL_DISPLAY_PATH,
        "dataset_paths": [
            "data/drug_name_id_map.csv",
            "data/drug_fingerprints_lookup.csv",
            "data/cell_line_features_lookup.csv",
            "data/depmap_features_lookup.csv",
            "data/input_data.csv",
            "data/metadata.csv",
            "data/model_matrix.csv",
        ],
        "feature_columns_path": "data/feature_columns.json",
        "old_compatibility_paths": {
            "drug_features": project_relative(OLD_DRUG_FEATURES_PATH),
            "feature_columns": project_relative(OLD_FEATURE_COLUMNS_PATH),
            "model_registry": project_relative(MODEL_REGISTRY_PATH),
        },
        "asset_messages": data_service.asset_messages(),
        "configured": {
            "dataset": data_service.organized_lookup_assets_available() or data_service.get_dataset_path() is not None,
            "feature_columns": FEATURE_COLUMNS_PATH.exists() or FEATURE_COLUMNS_PICKLE_PATH.exists(),
            "model": model_service.single_model_exists() or _safe_cell_line_model_count() > 0,
            "old_d1_d2_feature_mode": data_service.old_feature_assets_available(),
            "cell_line_specific_models": _safe_cell_line_model_count() > 0,
        },
        "health": health,
        "notes": [
            "The backend keeps old UI field names like NSC1, NSC2, and CELLNAME for compatibility.",
            "Feature engineering is isolated in backend/services/prediction_service.py.",
            "Dataset reference values, when present, are debugging metadata and never replace model predictions.",
            "Molecule and performance details are graceful placeholders until matching assets are configured.",
        ],
    }


def _safe_feature_columns():
    try:
        return data_service.get_feature_columns()
    except Exception:
        return []


def _safe_cell_line_model_count(errors=None):
    try:
        return model_service.cell_line_model_count()
    except Exception as exc:
        if errors is not None:
            errors.append(str(exc))
        return 0


def _model_types():
    counts = _model_type_counts()
    return list(counts) if counts else [MODEL_DISPLAY_NAME]


def _model_type_counts():
    try:
        registry = model_service.load_model_registry()
    except Exception:
        registry = None
    if registry is not None and not registry.empty:
        counts = registry["selected_model"].dropna().astype(str).value_counts().to_dict()
        return {str(model): int(count) for model, count in counts.items()}
    return {MODEL_DISPLAY_NAME: 1 if model_service.single_model_exists() else 0}


def _performance_payload():
    rows = []
    deployed_average = {}

    if STEP6_MODEL_SUMMARY_PATH.exists():
        rows = _read_performance_rows(STEP6_MODEL_SUMMARY_PATH)
    if STEP5_AVERAGE_PERFORMANCE_PATH.exists():
        average_rows = _read_performance_rows(STEP5_AVERAGE_PERFORMANCE_PATH)
        if average_rows:
            deployed_average = average_rows[0]

    return {
        "by_model_type": rows,
        "deployed_final_average": deployed_average,
    }


def _read_performance_rows(path):
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    return frame.head(50).where(pd.notna(frame), None).to_dict(orient="records")
