from datetime import datetime, timezone

import pandas as pd
from werkzeug.utils import secure_filename

from backend.config import OUTPUTS_DIR, UPLOADS_DIR
from backend.services import prediction_service


class BatchServiceError(Exception):
    """Raised when an uploaded batch CSV cannot be processed."""


REQUIRED_COLUMNS = ("NSC1", "NSC2", "CELLNAME")
COLUMN_ALIASES = {
    "drug1_id": "NSC1",
    "drug1": "NSC1",
    "nsc1": "NSC1",
    "drug2_id": "NSC2",
    "drug2": "NSC2",
    "nsc2": "NSC2",
    "cell_line": "CELLNAME",
    "cellline": "CELLNAME",
    "cell": "CELLNAME",
    "cellname": "CELLNAME",
}


def process_upload(upload):
    if upload is None or not getattr(upload, "filename", ""):
        raise BatchServiceError("Upload a CSV file using the form field named 'file'.")

    safe_name = secure_filename(upload.filename) or "batch.csv"
    if not safe_name.lower().endswith(".csv"):
        raise BatchServiceError("Batch prediction only accepts .csv files.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    upload_path = UPLOADS_DIR / f"{timestamp}_{safe_name}"
    upload.save(upload_path)

    try:
        frame = pd.read_csv(upload_path)
    except Exception as exc:
        raise BatchServiceError(f"Could not read uploaded CSV: {exc}") from exc

    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise BatchServiceError(f"CSV must include required columns: {', '.join(REQUIRED_COLUMNS)}.")

    records = []
    successful_rows = 0
    failed_rows = 0

    for index, row in frame.iterrows():
        nsc1 = _clean_cell(row.get("NSC1"))
        nsc2 = _clean_cell(row.get("NSC2"))
        cellname = _clean_cell(row.get("CELLNAME")) or "General"
        base = {
            "row_index": int(index) + 1,
            "NSC1": nsc1,
            "NSC2": nsc2,
            "CELLNAME": cellname,
        }

        try:
            prediction = prediction_service.predict_single(base)
            records.append({
                **base,
                "model_used": prediction["model_used"],
                "model_path": prediction["model_path"],
                "model_selection_mode": prediction.get("model_selection_mode", ""),
                "feature_mode_used": prediction.get("feature_mode_used", ""),
                "feature_count": prediction.get("feature_count", ""),
                "prediction_1_to_2": prediction["prediction_1_to_2"],
                "prediction_2_to_1": prediction["prediction_2_to_1"],
                "prediction_NSC1_to_NSC2": prediction["prediction_NSC1_to_NSC2"],
                "prediction_NSC2_to_NSC1": prediction["prediction_NSC2_to_NSC1"],
                "final_predicted_COMBOSCORE": prediction["final_predicted_COMBOSCORE"],
                "dataset_reference_available": prediction.get("dataset_reference_available", False),
                "dataset_reference_COMBOSCORE": prediction.get("dataset_reference_COMBOSCORE"),
                "absolute_error": prediction.get("absolute_error"),
                "signed_error": prediction.get("signed_error"),
                "label": prediction["label"],
                "prediction_label": prediction["prediction_label"],
                "prediction_category": prediction["prediction_category"],
                "status": "success",
                "error": "",
            })
            successful_rows += 1
        except Exception as exc:
            records.append({
                **base,
                "model_used": "",
                "model_path": "",
                "model_selection_mode": "",
                "feature_mode_used": "",
                "feature_count": "",
                "prediction_1_to_2": "",
                "prediction_2_to_1": "",
                "prediction_NSC1_to_NSC2": "",
                "prediction_NSC2_to_NSC1": "",
                "final_predicted_COMBOSCORE": "",
                "dataset_reference_available": False,
                "dataset_reference_COMBOSCORE": "",
                "absolute_error": "",
                "signed_error": "",
                "label": "",
                "prediction_label": "",
                "prediction_category": "",
                "status": "error",
                "error": str(exc),
            })
            failed_rows += 1

    output_filename = f"batch_predictions_{timestamp}.csv"
    output_path = OUTPUTS_DIR / output_filename
    pd.DataFrame(records).to_csv(output_path, index=False)

    return {
        "status": "success",
        "total_rows": int(len(frame)),
        "successful_rows": int(successful_rows),
        "failed_rows": int(failed_rows),
        "preview": _json_records(records[:50]),
        "download_filename": output_filename,
        "output_file": output_filename,
    }


def _normalize_columns(frame):
    rename = {}
    for column in frame.columns:
        normalized = str(column).strip()
        lower = normalized.lower()
        rename[column] = COLUMN_ALIASES.get(lower, normalized)
    return frame.rename(columns=rename)


def _clean_cell(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _json_records(records):
    cleaned = []
    for record in records:
        item = {}
        for key, value in record.items():
            if pd.isna(value):
                item[key] = ""
            elif hasattr(value, "item"):
                item[key] = value.item()
            else:
                item[key] = value
        cleaned.append(item)
    return cleaned
