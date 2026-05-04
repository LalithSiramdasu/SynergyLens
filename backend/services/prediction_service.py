import pandas as pd

from backend.config import MODEL_DISPLAY_NAME, MODEL_DISPLAY_PATH
from backend.services import data_service, model_service
from backend.services.data_service import DataAssetError


class PredictionInputError(Exception):
    """Raised when a prediction request is missing required UI fields."""


class FeatureBuildError(Exception):
    """Raised when feature engineering assets cannot build a model input row."""


SYNERGY_THRESHOLD = 20.0
ANTAGONISM_THRESHOLD = -20.0


def normalize_prediction_input(payload):
    payload = payload or {}
    nsc1 = payload.get("NSC1") or payload.get("drug1_id") or payload.get("drug1") or payload.get("nsc1")
    nsc2 = payload.get("NSC2") or payload.get("drug2_id") or payload.get("drug2") or payload.get("nsc2")
    cellname = payload.get("CELLNAME") or payload.get("cell_line") or payload.get("cellname") or payload.get("context")

    nsc1 = str(nsc1 or "").strip()
    nsc2 = str(nsc2 or "").strip()
    cellname = str(cellname or "").strip() or "General"

    missing = []
    if not nsc1:
        missing.append("NSC1")
    if not nsc2:
        missing.append("NSC2")
    if not cellname:
        missing.append("CELLNAME")
    if missing:
        raise PredictionInputError(f"Missing required prediction field(s): {', '.join(missing)}.")

    return nsc1, nsc2, cellname


def build_feature_vector(nsc1, nsc2, cellname):
    """Build the one-row model input DataFrame for the deployed single model.

    Adaptation point for the new dataset:
    replace the organized lookup builder with your real feature engineering if
    model inputs are not derived from the files currently stored in data/.
    Keep the returned DataFrame columns in the exact order from
    data/feature_columns.json.
    """
    try:
        feature_columns = data_service.get_feature_columns()
    except DataAssetError as exc:
        raise FeatureBuildError(str(exc)) from exc

    if not feature_columns:
        raise FeatureBuildError(
            "Feature engineering is not configured. Add data/feature_columns.json "
            "as a JSON list containing the exact model feature order."
        )

    try:
        frame = data_service.build_feature_frame(nsc1, nsc2, cellname, feature_columns)
    except DataAssetError as exc:
        raise FeatureBuildError(str(exc)) from exc

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FeatureBuildError("Feature engineering returned no model input rows.")

    return frame


def predict_single(payload):
    nsc1, nsc2, cellname = normalize_prediction_input(payload)

    forward_features = build_feature_vector(nsc1, nsc2, cellname)
    forward_prediction = model_service.predict(forward_features)

    reverse_note = ""
    try:
        reverse_features = build_feature_vector(nsc2, nsc1, cellname)
        reverse_prediction = model_service.predict(reverse_features)
    except FeatureBuildError:
        reverse_prediction = forward_prediction
        reverse_note = (
            " Reverse feature row was not found, so the backend treated the deployed model "
            "as order-independent for UI compatibility."
        )

    final_prediction = float((forward_prediction + reverse_prediction) / 2.0)
    label = label_for_score(final_prediction)
    category = category_for_score(final_prediction)

    explanation = (
        f"The deployed single model predicted a final score of {final_prediction:.3f} "
        f"for {nsc1} + {nsc2} in {cellname}. "
        f"Using the configured thresholds, this is labeled {label}."
        f"{reverse_note}"
    )

    return {
        "status": "success",
        "input": {
            "NSC1": nsc1,
            "NSC2": nsc2,
            "CELLNAME": cellname,
        },
        "NSC1": nsc1,
        "NSC2": nsc2,
        "CELLNAME": cellname,
        "model_used": MODEL_DISPLAY_NAME,
        "model_name": MODEL_DISPLAY_NAME,
        "model_path": MODEL_DISPLAY_PATH,
        "prediction_1_to_2": forward_prediction,
        "prediction_2_to_1": reverse_prediction,
        "prediction_NSC1_to_NSC2": forward_prediction,
        "prediction_NSC2_to_NSC1": reverse_prediction,
        "final_predicted_COMBOSCORE": final_prediction,
        "final_prediction": final_prediction,
        "predicted_comboscore": final_prediction,
        "score": final_prediction,
        "label": label,
        "prediction_label": label,
        "category": category,
        "prediction_category": category,
        "explanation": explanation,
        "suggestion": "Use this as a screening signal and validate important findings experimentally.",
        "feature_count": int(forward_features.shape[1]),
    }


def label_for_score(score):
    value = float(score)
    if value >= SYNERGY_THRESHOLD:
        return "synergistic"
    if value <= ANTAGONISM_THRESHOLD:
        return "antagonistic"
    return "neutral"


def category_for_score(score):
    value = float(score)
    if value >= SYNERGY_THRESHOLD:
        return "synergistic"
    if value <= ANTAGONISM_THRESHOLD:
        return "antagonistic"
    return "neutral"
