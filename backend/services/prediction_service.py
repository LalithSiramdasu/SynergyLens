import pandas as pd

from backend.services import data_service, model_service
from backend.services.data_service import DataAssetError


class PredictionInputError(Exception):
    """Raised when a prediction request is missing required UI fields."""


class FeatureBuildError(Exception):
    """Raised when feature engineering assets cannot build a model input row."""


SYNERGY_THRESHOLD = 20.0
ANTAGONISM_THRESHOLD = -20.0
FIELD_ALIASES = {
    "NSC1": ("NSC1", "drug1_id", "Drug1", "drug1", "nsc1"),
    "NSC2": ("NSC2", "drug2_id", "Drug2", "drug2", "nsc2"),
    "CELLNAME": ("CELLNAME", "cell_line", "cellLine", "cell", "cellname", "context"),
}


def normalize_prediction_input(payload):
    if not isinstance(payload, dict):
        raise PredictionInputError("Request body must be a JSON object.")

    nsc1 = _required_field(payload, "NSC1")
    nsc2 = _required_field(payload, "NSC2")
    cellname = _required_field(payload, "CELLNAME")

    return _clean_identifier(nsc1), _clean_identifier(nsc2), str(cellname).strip()


def _required_field(payload, field_name):
    for candidate in FIELD_ALIASES[field_name]:
        value = payload.get(candidate)
        if value is not None and str(value).strip() != "":
            return value
    raise PredictionInputError(f"{field_name} is required.")


def _clean_identifier(value):
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def build_feature_vector(nsc1, nsc2, cellname, feature_mode="organized_lookup"):
    """Build the one-row model input DataFrame.

    Current SynergyLens assets use a single deployed model with a 1,600-column
    feature order from data/feature_columns.json. If old NCI ALMANAC Step 6
    assets are added later, feature_mode="old_d1_d2" builds the old 526-column
    D1_/D2_ vector from data/drug_features.csv and
    data/step6_final_model_feature_columns.json.
    """
    return build_feature_bundle(nsc1, nsc2, cellname, feature_mode)["frame"]


def build_feature_bundle(nsc1, nsc2, cellname, feature_mode="organized_lookup"):
    try:
        feature_columns = data_service.get_feature_columns(feature_mode)
    except DataAssetError as exc:
        raise FeatureBuildError(str(exc)) from exc

    if not feature_columns:
        if feature_mode == "old_d1_d2":
            raise FeatureBuildError(
                "Old 526-feature mode is not configured. Add data/step6_final_model_feature_columns.json."
            )
        raise FeatureBuildError(
            "Feature engineering is not configured. Add data/feature_columns.json "
            "as a JSON list containing the exact model feature order."
        )

    try:
        frame = data_service.build_feature_frame(nsc1, nsc2, cellname, feature_columns, feature_mode)
    except DataAssetError as exc:
        raise FeatureBuildError(str(exc)) from exc

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FeatureBuildError("Feature engineering returned no model input rows.")

    return {
        "frame": frame,
        "feature_columns": feature_columns,
        "feature_count": int(frame.shape[1]),
        "feature_mode": feature_mode,
    }


def predict_single(payload):
    nsc1, nsc2, cellname = normalize_prediction_input(payload)
    model_selection = model_service.select_model_for_cell_line(cellname)
    feature_mode = model_selection["feature_mode"]

    forward_bundle = build_feature_bundle(nsc1, nsc2, cellname, feature_mode)
    forward_prediction = model_service.predict_with_model(model_selection["model"], forward_bundle["frame"])

    reverse_note = ""
    reverse_feature_available = True
    try:
        reverse_bundle = build_feature_bundle(nsc2, nsc1, cellname, feature_mode)
        reverse_prediction = model_service.predict_with_model(model_selection["model"], reverse_bundle["frame"])
    except FeatureBuildError:
        reverse_bundle = forward_bundle
        reverse_prediction = forward_prediction
        reverse_feature_available = False
        reverse_note = (
            " Reverse feature row was not found, so the backend treated this model as "
            "order-independent for UI compatibility."
        )

    final_prediction = float((forward_prediction + reverse_prediction) / 2.0)
    label = label_for_score(final_prediction)
    category = category_for_score(final_prediction)
    reference = _reference_debug_fields(nsc1, nsc2, cellname, final_prediction)

    explanation = (
        f"The deployed model predicted a final ComboScore of {final_prediction:.3f} "
        f"for {nsc1} + {nsc2} in {cellname}. "
        f"Using the configured thresholds, this is labeled {label}."
        f"{reverse_note}"
    )
    if reference["dataset_reference_available"]:
        explanation += (
            " A matching dataset ComboScore was found for debugging only; it was not used "
            "as the prediction."
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
        "model_used": model_selection["model_used"],
        "model_name": model_selection["model_name"],
        "model_type": model_selection["model_type"],
        "model_path": model_selection["model_path"],
        "selected_model_path": model_selection["selected_model_path"],
        "model_selection_mode": model_selection["model_selection_mode"],
        "feature_mode_used": feature_mode,
        "feature_count": forward_bundle["feature_count"],
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
        "reverse_feature_available": reverse_feature_available,
        "calibration_applied": False,
        "calibrated_predicted_COMBOSCORE": None,
        **reference,
        "debug": {
            "feature_count": forward_bundle["feature_count"],
            "feature_mode_used": feature_mode,
            "model_type": model_selection["model_type"],
            "model_used": model_selection["model_used"],
            "selected_model_path": model_selection["selected_model_path"],
            "model_selection_mode": model_selection["model_selection_mode"],
            "forward_prediction": forward_prediction,
            "reverse_prediction": reverse_prediction,
            "reverse_feature_available": reverse_feature_available,
            "dataset_reference_available": reference["dataset_reference_available"],
            "dataset_reference_source": reference["dataset_reference_source"],
        },
    }


def _reference_debug_fields(nsc1, nsc2, cellname, final_prediction):
    reference = data_service.lookup_dataset_reference(nsc1, nsc2, cellname)
    if reference["dataset_reference_available"]:
        actual = reference["dataset_reference_COMBOSCORE"]
        signed_error = float(final_prediction - actual)
        return {
            **reference,
            "absolute_error": abs(signed_error),
            "signed_error": signed_error,
        }

    return {
        **reference,
        "absolute_error": None,
        "signed_error": None,
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
