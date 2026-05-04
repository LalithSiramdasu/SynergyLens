# Feature Restore Checklist

Comparison source: `LalithSiramdasu/Drug-Synergy-Prediction` tag `v2`.

Target: this `SynergyLens` branch `restore-working-features-from-v2`.

## A. Core App And Routing

- [x] Flask app factory exists and starts.
- [x] `/` renders the active UI.
- [x] API routes use a Flask Blueprint.
- [x] Static CSS and JS load from Flask `static/`.
- [x] Strengthen JSON error handling for all HTTP errors.
- [x] Keep compatibility routes from v2 where harmless.

## B. Backend Health Badge

- [x] `/api/health` returns `status`, counts, model count, feature count, and `errors`.
- [x] New project reports one model when `models/model.pkl` exists.
- [x] Counts come from organized new data assets.
- [x] Errors are readable strings.

## C. Drug List And Autocomplete

- [x] `/api/drugs` returns IDs and names.
- [x] Drug IDs come from `data/drug_name_id_map.csv`.
- [x] Frontend can normalize returned objects.
- [x] Add optional `q` filtering compatibility from v2.

## D. Cell-Line/Context Dropdown

- [x] `/api/cell-lines` returns available contexts.
- [x] Prediction and explain forms share frontend state.
- [x] Values come from `data/cell_line_features_lookup.csv`.

## E. Prediction

- [x] `/api/predict` uses `NSC1`, `NSC2`, `CELLNAME`.
- [x] Single model loads from `models/model.pkl`.
- [x] Feature order follows `data/feature_columns.json`.
- [x] Response includes frontend-compatible prediction fields.
- [x] Threshold labels use `>= 20`, neutral band, `<= -20`.
- [x] Forward/reverse predictions use same feature builder.

## F. Gauge And Animation

- [x] Gauge ticks and labels render.
- [x] Needle and score animate with `requestAnimationFrame`.
- [x] SVG needle transform is applied directly for reliability.
- [x] User-requested display range is currently `-120` to `+80`.
- [x] Node mocked-DOM runtime check confirms gauge range, ticks, and needle transform.
- [x] Live HTTP/static checks passed after starting Flask.

## G. Demo Cases

- [x] `/api/demo-cases` returns three cases.
- [x] Demo cases use valid available drugs/context.
- [x] No fake score labels are shown on demo cards.

## H. Explain AI

- [x] `/api/explain` returns a UI-compatible response.
- [x] Missing/incompatible SHAP does not break the UI.
- [x] Improve fallback wording and empty-list handling.
- [x] Use real SHAP if installed and compatible.

## I. Molecules

- [x] `/api/molecule-pair` does not crash.
- [x] Placeholder cards are returned when molecule assets are absent.
- [x] Add optional SMILES CSV support.
- [x] Keep RDKit optional.

## J. Batch Prediction

- [x] `/api/batch-predict` accepts CSV upload.
- [x] Uses `predict_single()` row by row.
- [x] One bad row does not stop the batch.
- [x] Outputs are saved under `outputs/`.
- [x] `/api/download/<filename>` blocks traversal.
- [x] Add download verification to `verify_app.py`.

## K. Chatbot

- [x] `/api/chat` supports project mode.
- [x] `/api/chat` supports prediction mode.
- [x] No external key is required.
- [x] Add stricter invalid-mode error.
- [x] Add prediction-chat verification.

## L. Project/About Section

- [x] `/api/model-performance-summary` works.
- [x] `/api/system-summary` works.
- [x] About page text references one deployed model.
- [x] Search again for old 60-model/Step wording.

## M. Frontend Polish

- [x] Theme/navigation/history/report code retained from UI.
- [x] Sample CSV uses valid new project examples.
- [x] Run JS syntax check.
- [x] Run Flask verifier.
- [x] Start app and check root/static/health over HTTP.
