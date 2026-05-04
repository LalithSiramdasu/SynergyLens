import numpy as np

from backend.config import MODEL_DISPLAY_NAME
from backend.services import model_service, prediction_service


def explain(payload):
    try:
        prediction = prediction_service.predict_single(payload)
    except Exception as exc:
        return _fallback_response(
            payload,
            prediction=None,
            message=(
                f"Feature explanation is not available because prediction assets are not ready: {exc}"
            ),
        )

    try:
        import shap  # type: ignore
    except Exception:
        return _fallback_response(
            payload,
            prediction=prediction,
            message="Feature explanation is not available for this model yet.",
        )

    try:
        feature_frame = prediction_service.build_feature_vector(
            prediction["NSC1"],
            prediction["NSC2"],
            prediction["CELLNAME"],
        )
        model = model_service.load_model()
        explainer = shap.Explainer(model, feature_frame)
        shap_result = explainer(feature_frame)
        impacts = _extract_impacts(shap_result)
        base_value = _extract_base_value(shap_result)
        contributors = _rank_contributors(feature_frame, impacts)
    except Exception as exc:
        return _fallback_response(
            payload,
            prediction=prediction,
            message="Feature explanation is not available for this model yet.",
            detail=str(exc),
        )

    return {
        "status": "success",
        "input": prediction["input"],
        "model_used": prediction.get("model_used", MODEL_DISPLAY_NAME),
        "final_predicted_COMBOSCORE": prediction["final_predicted_COMBOSCORE"],
        "prediction": prediction["final_predicted_COMBOSCORE"],
        "base_value": base_value,
        "top_positive_contributors": contributors["positive"],
        "top_negative_contributors": contributors["negative"],
        "plain_english_explanation": (
            "SHAP values estimate which configured features pushed the single-model "
            "prediction upward toward synergy or downward toward antagonism for this input row."
        ),
        "explanation": "SHAP explanation generated from the deployed single model.",
    }


def _fallback_response(payload, prediction=None, message="Feature explanation is not available for this model yet.", detail=""):
    try:
        nsc1, nsc2, cellname = prediction_service.normalize_prediction_input(payload)
    except Exception:
        nsc1, nsc2, cellname = "", "", "General"

    score = 0.0
    input_payload = {"NSC1": nsc1, "NSC2": nsc2, "CELLNAME": cellname}
    if prediction:
        score = float(prediction.get("final_predicted_COMBOSCORE", 0.0))
        input_payload = prediction.get("input", input_payload)

    explanation = (
        f"{message} Positive contributors would be interpreted as pushing the score upward toward synergy; "
        "negative contributors would be interpreted as pushing the score downward toward antagonism."
    )
    if detail:
        explanation = f"{explanation} Technical note: {detail[:180]}"

    return {
        "status": "success",
        "input": input_payload,
        "model_used": prediction.get("model_used", MODEL_DISPLAY_NAME) if prediction else MODEL_DISPLAY_NAME,
        "final_predicted_COMBOSCORE": score,
        "prediction": score,
        "base_value": None,
        "top_positive_contributors": [],
        "top_negative_contributors": [],
        "plain_english_explanation": explanation,
        "explanation": explanation,
        "explanation_summary": explanation,
        "suggestion": "Add SHAP-compatible model assets later if feature attribution is required.",
    }


def _extract_impacts(shap_result):
    values = np.asarray(getattr(shap_result, "values", shap_result))
    if values.ndim == 3:
        values = values[0, :, 0]
    elif values.ndim > 1:
        values = values[0]
    return values.reshape(-1)


def _extract_base_value(shap_result):
    base_values = getattr(shap_result, "base_values", None)
    if base_values is None:
        return None
    values = np.asarray(base_values).reshape(-1)
    if values.size == 0:
        return None
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return None


def _rank_contributors(feature_frame, impacts):
    row = feature_frame.iloc[0]
    rows = []
    for feature_name, impact in zip(feature_frame.columns, impacts):
        try:
            numeric_impact = float(impact)
        except (TypeError, ValueError):
            numeric_impact = 0.0
        rows.append({
            "readable_feature": str(feature_name),
            "feature_value": _clean_value(row[feature_name]),
            "impact": numeric_impact,
            "direction": (
                "pushes score upward toward synergy"
                if numeric_impact >= 0
                else "pushes score downward toward antagonism"
            ),
        })

    positive = sorted((item for item in rows if item["impact"] > 0), key=lambda item: item["impact"], reverse=True)[:7]
    negative = sorted((item for item in rows if item["impact"] < 0), key=lambda item: item["impact"])[:7]
    return {"positive": positive, "negative": negative}


def _clean_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value
