from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory

from backend.config import MODEL_DISPLAY_NAME, MODEL_DISPLAY_PATH, OUTPUTS_DIR
from backend.services import (
    batch_service,
    chat_service,
    data_service,
    demo_service,
    explanation_service,
    model_service,
    molecule_service,
    prediction_service,
    summary_service,
)
from backend.services.model_service import ModelAssetError, ModelPredictionError
from backend.services.prediction_service import FeatureBuildError, PredictionInputError


api_bp = Blueprint("api", __name__, url_prefix="/api")


def error_response(message, status_code=400, **extra):
    payload = {
        "status": "error",
        "message": str(message),
        "error": str(message),
    }
    payload.update(extra)
    return jsonify(payload), status_code


def _drug_item_matches(item, query):
    if isinstance(item, dict):
        haystack = f"{item.get('id', '')} {item.get('name', '')}".lower()
    else:
        haystack = str(item).lower()
    return query in haystack


@api_bp.get("/health")
def health():
    return jsonify(summary_service.health_summary())


@api_bp.get("/drugs")
def drugs():
    items = data_service.get_drugs()
    query = request.args.get("q", "").strip().lower().replace("nsc", "").strip()
    if query:
        items = [item for item in items if _drug_item_matches(item, query)][:20]
    return jsonify({
        "status": "success",
        "count": len(items),
        "drugs": items,
    })


@api_bp.get("/cell-lines")
def cell_lines():
    items = data_service.get_cell_lines()
    return jsonify({
        "status": "success",
        "count": len(items),
        "cell_lines": items,
    })


@api_bp.get("/model-info/<path:cell_line>")
def model_info(cell_line):
    exists = model_service.model_exists()
    return jsonify({
        "status": "success",
        "cell_line": cell_line,
        "context": cell_line,
        "model_used": MODEL_DISPLAY_NAME,
        "model_name": MODEL_DISPLAY_NAME,
        "model_path": MODEL_DISPLAY_PATH,
        "model_file_exists": exists,
        "model_count": 1 if exists else 0,
        "message": (
            "This new SynergyLens project uses one deployed model for every valid context."
            if exists
            else "Model file is missing. Place the deployed model at models/model.pkl."
        ),
    })


@api_bp.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(prediction_service.predict_single(payload))
    except PredictionInputError as exc:
        return error_response(exc, 400)
    except FeatureBuildError as exc:
        return error_response(exc, 422)
    except ModelAssetError as exc:
        return error_response(exc, 503)
    except ModelPredictionError as exc:
        return error_response(exc, 500)


@api_bp.post("/batch-predict")
def batch_predict():
    upload = request.files.get("file")
    try:
        return jsonify(batch_service.process_upload(upload))
    except batch_service.BatchServiceError as exc:
        return error_response(exc, 400)


@api_bp.get("/download/<path:filename>")
def download(filename):
    requested = Path(filename)
    if requested.name != filename:
        return error_response("Invalid download filename.", 400)

    output_path = (OUTPUTS_DIR / requested.name).resolve()
    output_root = OUTPUTS_DIR.resolve()
    if output_path.parent != output_root or not output_path.exists() or not output_path.is_file():
        return error_response("Requested output file was not found.", 404)

    return send_from_directory(output_root, requested.name, as_attachment=True)


@api_bp.post("/explain")
def explain():
    payload = request.get_json(silent=True) or {}
    return jsonify(explanation_service.explain(payload))


@api_bp.post("/molecule-pair")
def molecule_pair():
    payload = request.get_json(silent=True) or {}
    return jsonify(molecule_service.lookup_pair(payload))


@api_bp.get("/molecule/<nsc>")
def molecule(nsc):
    return jsonify(molecule_service.lookup_single(nsc))


@api_bp.post("/molecules")
def molecules():
    payload = request.get_json(silent=True) or {}
    return jsonify(molecule_service.lookup_pair(payload))


@api_bp.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode") or "project").strip().lower()
    if mode not in {"project", "prediction"}:
        return error_response("Only project and prediction chat modes are available.", 400)
    payload["mode"] = mode
    return jsonify(chat_service.answer(payload))


@api_bp.get("/demo-cases")
def demo_cases():
    cases = demo_service.get_demo_cases()
    return jsonify({
        "status": "success",
        "count": len(cases),
        "demo_cases": cases,
    })


@api_bp.get("/model-performance-summary")
def model_performance_summary():
    return jsonify(summary_service.model_performance_summary())


@api_bp.get("/system-summary")
def system_summary():
    return jsonify(summary_service.system_summary())
