from backend.config import (
    FEATURE_COLUMNS_PATH,
    FEATURE_COLUMNS_PICKLE_PATH,
    MODEL_DISPLAY_NAME,
    MODEL_DISPLAY_PATH,
    MODEL_PATH,
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
        errors.append("Model file is missing: add models/model.pkl.")

    return {
        "status": "success" if not errors else "error",
        "available_drugs": len(drugs),
        "available_cell_lines": len(cell_lines),
        "feature_column_count": len(feature_columns),
        "model_count": 1 if model_service.model_exists() else 0,
        "errors": errors,
    }


def model_performance_summary():
    drugs = data_service.get_drugs()
    cell_lines = data_service.get_cell_lines()
    feature_columns = _safe_feature_columns()

    return {
        "status": "success",
        "assets": {
            "total_cell_lines": len(cell_lines),
            "total_drugs": len(drugs),
            "feature_vector": len(feature_columns),
            "final_model_count": 1 if model_service.model_exists() else 0,
        },
        "model_summary": {
            "model_types": [MODEL_DISPLAY_NAME],
            "count_per_model_type": {MODEL_DISPLAY_NAME: 1 if model_service.model_exists() else 0},
            "total_models": 1 if model_service.model_exists() else 0,
        },
        "performance": {
            "by_model_type": [],
            "deployed_final_average": {},
        },
        "explanation": "This new project uses one deployed model. Add performance metrics later if you want this panel to show validation results.",
    }


def system_summary():
    health = health_summary()
    return {
        "status": "success",
        "project_name": "SynergyLens",
        "summary": "Clean Flask backend for the extracted SynergyLens UI and a new single-model project.",
        "model_strategy": "single deployed model",
        "model_path": MODEL_DISPLAY_PATH,
        "dataset_paths": [
            "data/drug_name_id_map.csv",
            "data/drug_fingerprints_lookup.csv",
            "data/cell_line_features_lookup.csv",
            "data/depmap_features_lookup.csv",
            "data/input_data.csv",
            "data/metadata.csv",
        ],
        "feature_columns_path": "data/feature_columns.json",
        "asset_messages": data_service.asset_messages(),
        "configured": {
            "dataset": data_service.organized_lookup_assets_available() or data_service.get_dataset_path() is not None,
            "feature_columns": FEATURE_COLUMNS_PATH.exists() or FEATURE_COLUMNS_PICKLE_PATH.exists(),
            "model": model_service.model_exists(),
        },
        "health": health,
        "notes": [
            "The backend keeps old UI field names like NSC1, NSC2, and CELLNAME for compatibility.",
            "Feature engineering is isolated in backend/services/prediction_service.py.",
            "Molecule and performance details are placeholders until new project assets are configured.",
        ],
    }


def _safe_feature_columns():
    try:
        return data_service.get_feature_columns()
    except Exception:
        return []
