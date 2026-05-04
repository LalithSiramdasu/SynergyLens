from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.config import (
    MODEL_DISPLAY_NAME,
    MODEL_DISPLAY_PATH,
    MODEL_PATH,
    MODEL_REGISTRY_PATH,
    MODELS_DIR,
    project_relative,
)


class ModelAssetError(Exception):
    """Raised when configured model assets are unavailable or invalid."""


class ModelPredictionError(Exception):
    """Raised when a loaded model cannot produce a prediction."""


REQUIRED_REGISTRY_COLUMNS = {"cell_line", "safe_cell_line", "selected_model", "model_path"}


def single_model_exists():
    return MODEL_PATH.exists()


def cell_line_model_count():
    registry = load_model_registry()
    if registry is not None and not registry.empty:
        return int(registry["cell_line"].dropna().astype(str).str.strip().nunique())
    return len(list(MODELS_DIR.glob("final_step6_*.pkl"))) if MODELS_DIR.exists() else 0


def model_exists():
    return single_model_exists() or cell_line_model_count() > 0


def model_count():
    count = cell_line_model_count()
    if count:
        return count
    return 1 if single_model_exists() else 0


@lru_cache(maxsize=1)
def load_model():
    """Load the current SynergyLens single deployed model."""
    if not MODEL_PATH.exists():
        raise ModelAssetError(
            f"Model file is missing. Place the trained single model at {project_relative(MODEL_PATH)}."
        )
    return _load_model_from_path(str(MODEL_PATH))


@lru_cache(maxsize=None)
def _load_model_from_path(local_model_path):
    try:
        model = joblib.load(local_model_path)
    except ModuleNotFoundError as exc:
        package_name = exc.name or "a required model package"
        raise ModelAssetError(
            f"Model dependency {package_name} is not installed. Install requirements.txt before prediction."
        ) from exc
    except Exception as exc:
        raise ModelAssetError(f"Could not load {project_relative(Path(local_model_path))}: {exc}") from exc

    if not hasattr(model, "predict"):
        raise ModelAssetError("The loaded model must expose a sklearn-like .predict() method.")

    return _normalize_loaded_model(model)


def _normalize_loaded_model(model):
    """Apply runtime compatibility fixes without retraining or editing model files."""
    for estimator in getattr(model, "estimators_", []):
        if not hasattr(estimator, "monotonic_cst"):
            estimator.monotonic_cst = None
    if hasattr(model, "estimators_") and hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


@lru_cache(maxsize=1)
def load_model_registry():
    if not MODEL_REGISTRY_PATH.exists():
        return None

    try:
        registry = pd.read_csv(MODEL_REGISTRY_PATH)
    except Exception as exc:
        raise ModelAssetError(f"Could not read {project_relative(MODEL_REGISTRY_PATH)}: {exc}") from exc

    missing = sorted(REQUIRED_REGISTRY_COLUMNS.difference(registry.columns))
    if missing:
        raise ModelAssetError(f"Model registry is missing required columns: {', '.join(missing)}.")

    registry["cell_line"] = registry["cell_line"].astype(str).str.strip()
    registry["selected_model"] = registry["selected_model"].astype(str).str.strip()
    return registry


def get_cell_line_model_info(cell_line):
    registry = load_model_registry()
    if registry is None or registry.empty:
        return None

    requested = str(cell_line or "").strip()
    if not requested:
        raise ModelAssetError("CELLNAME is required.")

    matches = registry[registry["cell_line"].astype(str).str.strip().str.lower().eq(requested.lower())]
    if matches.empty:
        return None

    row = matches.iloc[0]
    model_filename = Path(str(row["model_path"]).strip()).name
    if not model_filename:
        raise ModelAssetError(f"Model path for cell line {requested} is empty in the model registry.")

    local_model_path = MODELS_DIR / model_filename
    return {
        "cell_line": requested,
        "safe_cell_line": str(row["safe_cell_line"]),
        "model_used": str(row["selected_model"]),
        "model_name": str(row["selected_model"]),
        "model_path": (Path("models") / model_filename).as_posix(),
        "local_model_path": local_model_path,
        "model_file_exists": local_model_path.is_file(),
        "feature_mode": "old_d1_d2",
        "model_selection_mode": "cell_line_specific",
    }


def select_model_for_cell_line(cell_line):
    """Select a model without silently faking a model run.

    If old Step 6 registry assets are configured and include the requested cell
    line, that specific model is used with the old 526-feature path. Otherwise
    the current single-model SynergyLens flow is used when models/model.pkl is
    present.
    """
    model_info = get_cell_line_model_info(cell_line)
    if model_info:
        if not model_info["model_file_exists"]:
            raise ModelAssetError(
                f"Model file for cell line {model_info['cell_line']} was not found: {model_info['model_path']}."
            )
        model = _load_model_from_path(str(model_info["local_model_path"]))
        return _selection_payload(model, model_info)

    if single_model_exists():
        model = load_model()
        return _selection_payload(
            model,
            {
                "cell_line": str(cell_line or "General").strip() or "General",
                "safe_cell_line": str(cell_line or "General").strip() or "General",
                "model_used": MODEL_DISPLAY_NAME,
                "model_name": MODEL_DISPLAY_NAME,
                "model_path": MODEL_DISPLAY_PATH,
                "local_model_path": MODEL_PATH,
                "model_file_exists": True,
                "feature_mode": "organized_lookup",
                "model_selection_mode": "single_model",
            },
        )

    registry = load_model_registry()
    if registry is not None and not registry.empty:
        raise ModelAssetError(
            f"No model registry entry was found for cell line '{cell_line}', and {MODEL_DISPLAY_PATH} is missing."
        )

    raise ModelAssetError(
        f"No model assets are available. Add {MODEL_DISPLAY_PATH} or configure old final_step6 models and registry."
    )


def _selection_payload(model, model_info):
    return {
        **model_info,
        "model": model,
        "model_type": type(model).__name__,
        "selected_model_path": model_info["model_path"],
    }


def predict(feature_frame):
    return predict_with_model(load_model(), feature_frame)


def predict_with_model(model, feature_frame):
    try:
        raw_prediction = model.predict(feature_frame)
    except Exception as exc:
        raise ModelPredictionError(f"Model prediction failed: {exc}") from exc

    values = np.asarray(raw_prediction).reshape(-1)
    if values.size == 0:
        raise ModelPredictionError("Model returned an empty prediction.")

    try:
        return float(values[0])
    except (TypeError, ValueError) as exc:
        raise ModelPredictionError(f"Model returned a non-numeric prediction: {values[0]!r}") from exc
