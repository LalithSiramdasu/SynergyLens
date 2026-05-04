import io
import json

from app import create_app


def main():
    app = create_app()
    failures = 0

    with app.test_client() as client:
        prediction_payload = _verification_payload(client)
        batch_csv = (
            "NSC1,NSC2,CELLNAME\n"
            f"{prediction_payload['NSC1']},{prediction_payload['NSC2']},{prediction_payload['CELLNAME']}\n"
        )
        checks = [
            ("GET", "/", None),
            ("GET", "/api/health", None),
            ("GET", "/api/drugs", None),
            ("GET", "/api/cell-lines", None),
            ("GET", "/api/demo-cases", None),
            ("GET", "/api/model-performance-summary", None),
            ("GET", "/api/system-summary", None),
            ("POST_JSON", "/api/predict", prediction_payload),
            ("POST_JSON", "/api/explain", prediction_payload),
            ("POST_JSON", "/api/molecule-pair", {"NSC1": prediction_payload["NSC1"], "NSC2": prediction_payload["NSC2"]}),
            ("POST_JSON", "/api/chat", {"mode": "project", "question": "Where do I place model.pkl?"}),
            ("POST_FILE", "/api/batch-predict", batch_csv),
        ]

        for method, path, payload in checks:
            response = _request(client, method, path, payload)
            ok = response.status_code < 500
            failures += 0 if ok else 1
            print(_format_result(method, path, response, ok))

    if failures:
        raise SystemExit(1)


def _verification_payload(client):
    demos = client.get("/api/demo-cases").get_json(silent=True) or {}
    cases = demos.get("demo_cases") or []
    if cases:
        case = cases[0]
        return {
            "NSC1": str(case.get("NSC1") or "1001"),
            "NSC2": str(case.get("NSC2") or "1002"),
            "CELLNAME": str(case.get("CELLNAME") or "General"),
        }
    return {"NSC1": "1001", "NSC2": "1002", "CELLNAME": "General"}


def _request(client, method, path, payload):
    if method == "GET":
        return client.get(path)
    if method == "POST_JSON":
        return client.post(path, json=payload)
    if method == "POST_FILE":
        data = {"file": (io.BytesIO(payload.encode("utf-8")), "verify_batch.csv")}
        return client.post(path, data=data, content_type="multipart/form-data")
    raise ValueError(method)


def _format_result(method, path, response, ok):
    status = "OK" if ok else "FAIL"
    payload = response.get_json(silent=True)
    if payload is None:
        preview = response.get_data(as_text=True)[:120].replace("\n", " ")
    else:
        preview = json.dumps(payload, ensure_ascii=True)[:260]
    return f"[{status}] {method} {path} -> HTTP {response.status_code} :: {preview}"


if __name__ == "__main__":
    main()
