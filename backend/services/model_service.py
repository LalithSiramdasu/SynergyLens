from functools import lru_cache

import joblib
import numpy as np

from backend.config import MODEL_PATH, project_relative


class ModelAssetError(Exception):
    """Raised when the configured model file is unavailable or invalid."""


class ModelPredictionError(Exception):
    """Raised when the loaded model cannot produce a prediction."""


def model_exists():
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise ModelAssetError(
            f"Model file is missing. Place the trained single model at {project_relative(MODEL_PATH)}."
        )

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise ModelAssetError(f"Could not load {project_relative(MODEL_PATH)}: {exc}") from exc

    if not hasattr(model, "predict"):
        raise ModelAssetError("The loaded model must expose a sklearn-like .predict() method.")

    return model


def predict(feature_frame):
    model = load_model()
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
