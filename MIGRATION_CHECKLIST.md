# Migration Checklist

Branch: `feature/migrate-working-drug-synergy-features`

Reference inspected:
- Current `LalithSiramdasu/SynergyLens` after fetching and pushing merged `main`
- Old `LalithSiramdasu/Drug-Synergy-Prediction` reference checkout at `v2`

## Git Safety

- [x] Checked current branch and status before editing.
- [x] Fetched latest SynergyLens state from GitHub.
- [x] Merged local `restore-working-features-from-v2` into `main`.
- [x] Pushed merged `main` to GitHub.
- [x] Created `feature/migrate-working-drug-synergy-features`.

## Backend Migration

- [x] Support robust prediction input aliases from the old app.
- [x] Preserve current single-model 1,600-feature flow.
- [x] Add optional old 526-feature `D1_`/`D2_` flow when old assets are present.
- [x] Add optional cell-line-specific model loading when old registry/assets are present.
- [x] Add transparent dataset-reference debugging fields without faking predictions.
- [x] Add model/feature/debug metadata to prediction responses.
- [x] Keep batch prediction row-level success/failure behavior.
- [x] Keep Explain AI graceful when SHAP is unavailable/incompatible.
- [x] Keep molecule lookup graceful when structures are unavailable.
- [x] Improve health/system/model summaries for single-model and optional old-asset modes.
- [x] Update chatbot answers for ComboScore, NSC, old 263/526 mode, current single model, thresholds, and safety.

## Frontend Compatibility

- [x] Preserve SynergyLens UI structure and routes.
- [x] Keep demo cards score-free.
- [x] Preserve gauge animation and `-120` to `+80` display range.
- [x] Respect reduced-motion preference for gauge animation.
- [x] Display dataset-reference/debug information separately from model prediction when present.
- [x] Keep prediction status messages honest and non-fake.

## Verification

- [x] Update `verify_app.py` for aliases, debug fields, dataset-reference fields, and downloads.
- [x] Run `python -m compileall .`.
- [x] Run `node --check static/js/app.js`.
- [x] Run `python verify_app.py`.
- [x] Check `/`, `/api/health`, `/static/css/app.css`, `/static/js/app.js` through Flask test client. Background server launch was blocked by the tool reviewer.
- [x] Run a gauge runtime check.
