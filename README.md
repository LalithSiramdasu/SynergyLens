# SynergyLens Clean Flask Backend

This project wraps the extracted SynergyLens UI with a new, clean Flask backend. The frontend design is preserved, but the backend is written for a new dataset and one single deployed model.

## Project Layout

- `app.py`: Flask app factory and local run entry point
- `backend/routes/api_routes.py`: all API endpoints used by the UI
- `backend/services/`: modular data, model, prediction, batch, explanation, molecule, chat, demo, and summary services
- `templates/index.html`: copied extracted UI template
- `static/css/app.css`: copied extracted UI stylesheet
- `static/js/app.js`: copied extracted UI behavior
- `data/`: place the new dataset and feature order
- `models/`: place the single trained model
- `uploads/`: saved batch uploads
- `outputs/`: generated batch CSV files

## Required Assets

The current organized data bundle lives in `data/`:

- `data/drug_name_id_map.csv`
- `data/drug_fingerprints_lookup.csv`
- `data/cell_line_features_lookup.csv`
- `data/depmap_features_lookup.csv`
- `data/depmap_label_encoders.pkl`
- `data/feature_columns.json`
- `data/feature_columns.pkl`

The single deployed model lives here:

- `models/model.pkl`

`feature_columns.json` is a JSON list of 1,600 feature column names in the exact order expected by the model.

## Feature Engineering Hook

The implementation builds a model-ready row from the organized lookup files using `NSC1`, `NSC2`, and `CELLNAME`. If your new dataset needs custom encoding, joins, fingerprints, descriptors, or other feature work, edit:

`backend/services/prediction_service.py` -> `build_feature_vector()`

Keep it returning a one-row pandas DataFrame ordered by `data/feature_columns.json`.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open:

`http://127.0.0.1:5000/`

## Test

```bash
python -m compileall backend app.py verify_app.py
python verify_app.py
```

If Node.js is available:

```bash
node --check static/js/app.js
```

## UI Endpoints

The extracted UI depends on:

- `GET /`
- `GET /api/health`
- `GET /api/drugs`
- `GET /api/cell-lines`
- `POST /api/predict`
- `POST /api/batch-predict`
- `GET /api/download/<filename>`
- `POST /api/explain`
- `POST /api/molecule-pair`
- `POST /api/chat`
- `GET /api/demo-cases`
- `GET /api/model-performance-summary`
- `GET /api/system-summary`

Additional compatibility routes are also available for old UI/tooling checks:

- `GET /api/model-info/<cell_line>`
- `GET /api/molecule/<nsc>`
- `POST /api/molecules`

## Current Placeholders

Molecule rendering, model performance metrics, and SHAP explanations are graceful placeholders until matching new project assets are configured. The prediction backend is connected to the organized single-model asset bundle.
