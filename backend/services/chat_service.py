from backend.services import summary_service


PROJECT_SUGGESTIONS = [
    "What is ComboScore?",
    "How does prediction work?",
    "What are 263 and 526 features?",
    "Can I use this clinically?",
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
            "SynergyLens currently uses the single deployed model at models/model.pkl when that is the configured asset. "
            "It also supports the old Drug-Synergy-Prediction registry if final_step6 model files and "
            "results/step6_final_model_registry.csv are added. The response always reports model_used, model_path, "
            "model_selection_mode, and feature_mode_used so the UI does not guess."
        )
    elif "263" in lower or "526" in lower:
        text = (
            "The old NCI ALMANAC deployment used 263 prepared features per drug and built a 526-column pair vector "
            "with D1_ and D2_ prefixes. This repo keeps that path as optional compatibility when old assets are present. "
            "The current single model uses the feature order in data/feature_columns.json."
        )
    elif "dataset" in lower or "data" in lower:
        text = (
            "The project is compatible with NCI ALMANAC-style NSC drug-pair inputs. Current organized files live in data/: "
            "drug_name_id_map.csv, drug_fingerprints_lookup.csv, cell_line_features_lookup.csv, depmap_features_lookup.csv, "
            "and feature_columns.json. If data/model_matrix.csv is provided, exact observed ComboScore rows are shown only "
            "as dataset_reference_COMBOSCORE debugging metadata, never as the model prediction."
        )
    elif "feature" in lower:
        text = (
            "Feature construction is isolated in backend/services/prediction_service.py. "
            "Current SynergyLens builds the configured single-model feature vector; old assets can build the 526-column "
            "D1_/D2_ vector. In both modes, columns are ordered exactly by the saved feature-column file."
        )
    elif "batch" in lower or "csv" in lower:
        text = "Batch uploads need a CSV with NSC1, NSC2, and CELLNAME columns. Aliases from the UI are normalized before prediction."
    elif "molecule" in lower or "structure" in lower:
        text = "Molecule rendering is currently a placeholder. Add a real molecule lookup later if the new project has structure assets."
    elif "safety" in lower or "clinical" in lower or "trust" in lower:
        text = "Predictions are screening estimates, not biological proof or clinical advice. Important results should be validated experimentally."
    elif "comboscore" in lower or "synergy" in lower or "antagon" in lower:
        text = "ComboScore labels use transparent thresholds: score >= 20 is synergistic, score <= -20 is antagonistic, and values between -20 and 20 are neutral."
    else:
        text = (
            "SynergyLens is wired to a clean Flask backend for drug synergy screening with NSC1, NSC2, and CELLNAME inputs. "
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

    mode = prediction.get("model_selection_mode") or "single_model"
    feature_mode = prediction.get("feature_mode_used") or "configured feature"
    text = (
        f"For {nsc1} + {nsc2} in {cellname}, the latest model output is {score} "
        f"and the current label is {label}. The backend used {model} in {mode} mode with {feature_mode} features."
    )

    lower = question.lower()
    if "direction" in lower or "both" in lower:
        text += " SynergyLens predicts NSC1 to NSC2 and NSC2 to NSC1, then averages the two ComboScores. If a model is order-independent or reverse features cannot be built, both values may be identical and the response says so."
    elif "reference" in lower or "actual" in lower or "error" in lower:
        if prediction.get("dataset_reference_available"):
            text += (
                f" A dataset reference ComboScore of {prediction.get('dataset_reference_COMBOSCORE')} was found for debugging. "
                f"The signed error is {prediction.get('signed_error')} and the absolute error is {prediction.get('absolute_error')}. "
                "That reference value is not used as the model prediction."
            )
        else:
            text += " No exact dataset reference row was available, so no observed ComboScore error metadata is shown."
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
