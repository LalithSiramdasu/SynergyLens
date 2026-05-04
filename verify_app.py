import io
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _maybe_reexec_with_venv():
    if os.environ.get("SYNERGYLENS_VERIFY_NO_REEXEC") == "1":
        return
    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    target = VENV_PYTHON.resolve()
    if current != target:
        os.environ["SYNERGYLENS_VERIFY_NO_REEXEC"] = "1"
        os.execv(str(target), [str(target), str(Path(__file__).resolve())])


_maybe_reexec_with_venv()

from app import create_app  # noqa: E402
from backend.config import OUTPUTS_DIR, UPLOADS_DIR  # noqa: E402


PREDICTION_KEYS = (
    "status",
    "NSC1",
    "NSC2",
    "CELLNAME",
    "model_used",
    "model_path",
    "prediction_1_to_2",
    "prediction_2_to_1",
    "final_predicted_COMBOSCORE",
    "label",
    "prediction_label",
    "explanation",
    "feature_count",
    "model_selection_mode",
    "feature_mode_used",
    "model_type",
    "dataset_reference_checked",
    "dataset_reference_available",
    "dataset_reference_COMBOSCORE",
    "absolute_error",
    "signed_error",
    "debug",
)


class Verifier:
    def __init__(self, client):
        self.client = client
        self.passed = 0
        self.warnings = 0
        self.failures = 0

    def get(self, path, keys=(), allow_asset_error=False, label=None):
        response = self.client.get(path)
        return self._record("GET", path, response, keys, allow_asset_error, label)

    def post_json(self, path, payload, keys=(), allow_asset_error=False, label=None):
        response = self.client.post(path, json=payload)
        return self._record("POST", path, response, keys, allow_asset_error, label)

    def post_file(self, path, csv_text, keys=(), label=None):
        data = {"file": (io.BytesIO(csv_text.encode("utf-8")), "verify_batch.csv")}
        response = self.client.post(path, data=data, content_type="multipart/form-data")
        return self._record("POST_FILE", path, response, keys, True, label)

    def home(self):
        response = self.client.get("/")
        body = response.get_data(as_text=True)
        ok = response.status_code == 200 and "<html" in body.lower()
        self._print("OK" if ok else "FAIL", "GET", "/", response, "HTML loaded" if ok else body[:180])
        if ok:
            self.passed += 1
        else:
            self.failures += 1

    def get_static(self, path, expected_content_type):
        response = self.client.get(path)
        content_type = response.headers.get("Content-Type", "")
        ok = response.status_code == 200 and expected_content_type in content_type
        self._print("OK" if ok else "FAIL", "GET", path, response, content_type or _response_preview(response))
        if ok:
            self.passed += 1
        else:
            self.failures += 1

    def download(self, filename):
        response = self.client.get(f"/api/download/{filename}")
        ok = response.status_code == 200 and response.get_data()
        self._print("OK" if ok else "FAIL", "GET", f"/api/download/{filename}", response, _response_preview(response))
        if ok:
            self.passed += 1
        else:
            self.failures += 1

    def expect_blocked_download(self):
        response = self.client.get("/api/download/../app.py")
        ok = response.status_code in {400, 404}
        self._print("OK" if ok else "FAIL", "GET", "/api/download/../app.py", response, _response_preview(response))
        if ok:
            self.passed += 1
        else:
            self.failures += 1

    def _record(self, method, path, response, keys, allow_asset_error, label):
        payload = response.get_json(silent=True)
        status = "OK"
        message = _response_preview(response)

        if response.status_code >= 500:
            status = "FAIL"
        elif payload is None:
            status = "FAIL"
            message = "Expected JSON response."
        elif _contains_traceback(payload):
            status = "FAIL"
            message = "Response includes traceback-like text."
        elif payload.get("status") == "error":
            if allow_asset_error and _asset_limited(payload):
                status = "WARN"
                message = _compact_json(payload)
            else:
                status = "FAIL"
                message = _compact_json(payload)
        else:
            missing = [key for key in keys if key not in payload]
            if missing:
                status = "FAIL"
                message = f"Missing response field(s): {', '.join(missing)}"

        self._print(status, method, label or path, response, message)
        if status == "OK":
            self.passed += 1
        elif status == "WARN":
            self.warnings += 1
        else:
            self.failures += 1
        return payload or {}

    @staticmethod
    def _print(status, method, path, response, preview):
        print(f"[{status}] {method} {path} -> HTTP {response.status_code} :: {preview}")


def main():
    app = create_app()
    snapshots = _snapshot_generated_files()

    try:
        with app.test_client() as client:
            verifier = Verifier(client)

            verifier.home()
            verifier.get_static("/static/css/app.css", "text/css")
            verifier.get_static("/static/js/app.js", "javascript")
            health = verifier.get(
                "/api/health",
                ("status", "available_drugs", "available_cell_lines", "feature_column_count", "model_count", "errors"),
                allow_asset_error=True,
            )
            drugs = verifier.get("/api/drugs", ("status", "count", "drugs"), allow_asset_error=True)
            cell_lines = verifier.get("/api/cell-lines", ("status", "count", "cell_lines"), allow_asset_error=True)
            demos = verifier.get("/api/demo-cases", ("status", "count", "demo_cases"), allow_asset_error=True)

            payload = _build_prediction_payload(drugs, cell_lines, demos)
            first_drug = payload["NSC1"]
            if first_drug:
                verifier.get(f"/api/drugs?q={first_drug}", ("status", "count", "drugs"), allow_asset_error=True)

            prediction = verifier.post_json("/api/predict", payload, PREDICTION_KEYS, allow_asset_error=True)
            prediction_context = _prediction_context(payload, prediction)
            alias_payload = {
                "Drug1": payload["NSC1"],
                "Drug2": payload["NSC2"],
                "cellLine": payload["CELLNAME"],
            }
            alias_prediction = verifier.post_json(
                "/api/predict",
                alias_payload,
                PREDICTION_KEYS,
                allow_asset_error=True,
                label="/api/predict alias payload",
            )
            if prediction.get("status") == "success" and alias_prediction.get("status") == "success":
                _assert_same_prediction(verifier, prediction, alias_prediction)

            explanation = verifier.post_json(
                "/api/explain",
                payload,
                ("status", "prediction", "base_value", "top_positive_contributors", "top_negative_contributors", "explanation"),
                allow_asset_error=True,
            )
            verifier.post_json(
                "/api/molecule-pair",
                {"NSC1": payload["NSC1"], "NSC2": payload["NSC2"]},
                ("status", "NSC1", "NSC2"),
                allow_asset_error=True,
            )
            verifier.get(f"/api/molecule/{payload['NSC1']}", ("status", "requested_nsc", "molecule_found"), allow_asset_error=True)

            verifier.post_json(
                "/api/chat",
                {"mode": "project", "question": "What model is used?"},
                ("status", "answer", "suggested_questions"),
            )
            verifier.post_json(
                "/api/chat",
                {
                    "mode": "prediction",
                    "question": "What does this score mean?",
                    "prediction": prediction_context,
                    "explanation": explanation,
                },
                ("status", "answer", "suggested_questions"),
            )

            batch_csv = _batch_csv(payload)
            batch = verifier.post_file(
                "/api/batch-predict",
                batch_csv,
                ("status", "total_rows", "successful_rows", "failed_rows", "preview", "download_filename"),
            )
            download_filename = str(batch.get("download_filename") or batch.get("output_file") or "").strip()
            if download_filename:
                verifier.download(download_filename)
            else:
                verifier.warnings += 1
                print("[WARN] GET /api/download/<filename> -> skipped :: batch response did not include a filename")
            verifier.expect_blocked_download()

            verifier.get(
                "/api/model-performance-summary",
                ("status", "assets", "model_summary", "performance", "explanation"),
                allow_asset_error=True,
            )
            verifier.get(
                "/api/system-summary",
                ("status", "project_name", "summary", "model_strategy", "model_path", "health"),
                allow_asset_error=True,
            )
            verifier.get(f"/api/model-info/{payload['CELLNAME']}", ("status", "model_used", "model_path", "model_count"), allow_asset_error=True)

            print(
                f"\nVerification complete: {verifier.passed} passed, "
                f"{verifier.warnings} warnings, {verifier.failures} failures."
            )
            if verifier.failures:
                raise SystemExit(1)
    finally:
        _cleanup_generated_files(snapshots)


def _build_prediction_payload(drugs, cell_lines, demos):
    ids = []
    for item in drugs.get("drugs") or []:
        item_id = _drug_id(item)
        if item_id and item_id not in ids:
            ids.append(item_id)

    contexts = [_cell_line_value(item) for item in (cell_lines.get("cell_lines") or [])]
    contexts = [item for item in contexts if item]
    demo_cases = demos.get("demo_cases") or []

    if len(ids) >= 2:
        return {"NSC1": ids[0], "NSC2": ids[1], "CELLNAME": contexts[0] if contexts else "General"}

    for case in demo_cases:
        nsc1 = str(case.get("NSC1") or "").strip()
        nsc2 = str(case.get("NSC2") or "").strip()
        cellname = str(case.get("CELLNAME") or (contexts[0] if contexts else "General")).strip()
        if nsc1 and nsc2:
            return {"NSC1": nsc1, "NSC2": nsc2, "CELLNAME": cellname or "General"}

    while len(ids) < 2:
        ids.append(str(1001 + len(ids)))
    return {"NSC1": ids[0], "NSC2": ids[1], "CELLNAME": contexts[0] if contexts else "General"}


def _prediction_context(payload, prediction):
    if prediction.get("status") == "success":
        return prediction
    return {
        "status": "success",
        "input": payload,
        "NSC1": payload["NSC1"],
        "NSC2": payload["NSC2"],
        "CELLNAME": payload["CELLNAME"],
        "model_used": "SingleModel",
        "model_selection_mode": "single_model",
        "feature_mode_used": "organized_lookup",
        "final_predicted_COMBOSCORE": None,
        "label": "unavailable",
        "prediction_label": "unavailable",
    }


def _batch_csv(payload):
    return "\n".join([
        "Drug1,Drug2,cellLine",
        f"{payload['NSC1']},{payload['NSC2']},{payload['CELLNAME']}",
        f"VERIFY_UNKNOWN_DRUG,{payload['NSC2']},{payload['CELLNAME']}",
        "",
    ])


def _assert_same_prediction(verifier, prediction, alias_prediction):
    original = prediction.get("final_predicted_COMBOSCORE")
    alias = alias_prediction.get("final_predicted_COMBOSCORE")
    try:
        ok = abs(float(original) - float(alias)) < 1e-9
    except (TypeError, ValueError):
        ok = False

    if ok:
        verifier.passed += 1
        print("[OK] ALIAS /api/predict -> matching model prediction for Drug1/Drug2/cellLine aliases")
    else:
        verifier.failures += 1
        print("[FAIL] ALIAS /api/predict -> alias payload changed prediction result")


def _drug_id(item):
    if isinstance(item, dict):
        return str(item.get("id") or item.get("NSC") or item.get("drug_id") or "").strip()
    return str(item or "").strip()


def _cell_line_value(item):
    if isinstance(item, dict):
        return str(item.get("id") or item.get("name") or item.get("CELLNAME") or item.get("cell_line") or "").strip()
    return str(item or "").strip()


def _snapshot_generated_files():
    return {
        "outputs": set(OUTPUTS_DIR.glob("*")) if OUTPUTS_DIR.exists() else set(),
        "uploads": set(UPLOADS_DIR.glob("*")) if UPLOADS_DIR.exists() else set(),
    }


def _cleanup_generated_files(snapshot):
    _cleanup_new_files(OUTPUTS_DIR, snapshot.get("outputs", set()), "batch_predictions_")
    _cleanup_new_files(UPLOADS_DIR, snapshot.get("uploads", set()), "verify_")


def _cleanup_new_files(directory, before, required_prefix):
    if not directory.exists():
        return
    for path in directory.glob("*"):
        if path in before or not path.is_file():
            continue
        if required_prefix and required_prefix not in path.name:
            continue
        try:
            path.unlink()
        except OSError:
            pass


def _asset_limited(payload):
    text = _compact_json(payload).lower()
    markers = (
        "missing",
        "not configured",
        "not ready",
        "not found",
        "unknown drug",
        "unknown cell",
        "feature engineering",
        "feature order",
        "prediction data",
        "model file",
        "lookup",
    )
    return any(marker in text for marker in markers)


def _contains_traceback(payload):
    text = _compact_json(payload).lower()
    return "traceback" in text or "stack trace" in text


def _response_preview(response):
    payload = response.get_json(silent=True)
    if payload is not None:
        return _compact_json(payload)
    return response.get_data(as_text=True)[:220].replace("\n", " ")


def _compact_json(payload):
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)[:360]


if __name__ == "__main__":
    main()
