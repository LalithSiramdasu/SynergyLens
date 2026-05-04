from backend.services import summary_service


PROJECT_SUGGESTIONS = [
    "Where do I place the new dataset?",
    "Where do I place model.pkl?",
    "What feature file is required?",
    "What does the prediction label mean?",
]

PREDICTION_SUGGESTIONS = [
    "What does this score mean?",
    "Why are there two directional scores?",
    "What should I validate next?",
    "Can I use this clinically?",
]


def answer(payload):
    payload = payload or {}
    mode = str(payload.get("mode") or "project").lower()
    question = str(payload.get("question") or "").strip()

    if mode == "prediction":
        return _prediction_answer(question, payload.get("prediction") or {}, payload.get("explanation"))

    return _project_answer(question)


def _project_answer(question):
    lower = question.lower()
    system = summary_service.system_summary()

    if "model" in lower:
        text = (
            "This clean backend uses one deployed model at models/model.pkl. "
            "It uses the same deployed model for every valid context, with no per-context "
            "model registry. The model receives a feature DataFrame ordered by "
            "data/feature_columns.json."
        )
    elif "dataset" in lower or "data" in lower:
        text = (
            "The organized data bundle lives in data/: drug_name_id_map.csv, "
            "drug_fingerprints_lookup.csv, cell_line_features_lookup.csv, "
            "depmap_features_lookup.csv, and feature_columns.json. The frontend still "
            "sends NSC1, NSC2, and CELLNAME for compatibility."
        )
    elif "feature" in lower:
        text = (
            "Feature construction is isolated in backend/services/prediction_service.py. "
            "Edit build_feature_vector() when your new dataset needs custom feature engineering."
        )
    elif "batch" in lower or "csv" in lower:
        text = "Batch uploads need a CSV with NSC1, NSC2, and CELLNAME columns. Aliases from the UI are normalized before prediction."
    elif "molecule" in lower or "structure" in lower:
        text = "Molecule rendering is currently a placeholder. Add a real molecule lookup later if the new project has structure assets."
    elif "safety" in lower or "clinical" in lower or "trust" in lower:
        text = "Predictions are screening estimates, not biological proof or clinical advice. Important results should be validated experimentally."
    elif "comboscore" in lower or "synergy" in lower or "antagon" in lower:
        text = "Positive scores are labeled synergistic, scores near zero are neutral, and negative scores are antagonistic using thresholds in prediction_service.py."
    else:
        text = (
            "SynergyLens is now wired to a clean Flask backend for a new single-model project. "
            f"Configured assets: {', '.join(system.get('asset_messages', []))}."
        )

    return {
        "status": "success",
        "answer": text,
        "suggested_questions": PROJECT_SUGGESTIONS,
        "llm_used": False,
        "provider_label": "Built-in Guide",
    }


def _prediction_answer(question, prediction, explanation):
    score = prediction.get("final_predicted_COMBOSCORE")
    label = prediction.get("label") or prediction.get("prediction_label") or "unknown"
    model = prediction.get("model_used") or "SingleModel"
    input_payload = prediction.get("input") or {}
    nsc1 = input_payload.get("NSC1", "")
    nsc2 = input_payload.get("NSC2", "")
    cellname = input_payload.get("CELLNAME", "General")

    text = (
        f"For {nsc1} + {nsc2} in {cellname}, the latest model output is {score} "
        f"and the current label is {label}. The backend used {model}, the single deployed model."
    )

    lower = question.lower()
    if "direction" in lower or "both" in lower:
        text += " The UI expects two directional scores; this backend averages them for compatibility. If no reverse feature row exists, both scores may be identical."
    elif "feature" in lower or "xai" in lower or "caused" in lower:
        has_explanation = bool(explanation and explanation.get("features"))
        text += " Explain AI contributors are available only when SHAP and compatible model assets are configured." if not has_explanation else " The latest explanation payload includes ranked feature contributors."
    elif "clinical" in lower or "trust" in lower or "advice" in lower:
        text += " Treat this as a screening estimate only, not clinical advice."
    else:
        text += " Positive scores indicate synergy, near-zero scores indicate neutral behavior, and negative scores indicate antagonism."

    return {
        "status": "success",
        "answer": text,
        "suggested_questions": PREDICTION_SUGGESTIONS,
        "llm_used": False,
        "provider_label": "Built-in Guide",
    }
