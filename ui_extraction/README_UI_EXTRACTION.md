# SynergyLens UI Extraction

This folder contains only the active frontend/UI files from the current SynergyLens Flask app.

## Extracted Files

- `templates/index.html` - Flask/Jinja template that controls the page layout and all visible UI sections.
- `static/css/app.css` - Main stylesheet that controls the visual design, responsive layout, light/dark theme, cards, forms, gauge, chat panels, and tables.
- `static/js/app.js` - Frontend behavior, API calls, form validation, result rendering, molecule rendering, batch upload handling, chat UI, local history, and local report/download helpers.
- `README_UI_EXTRACTION.md` - This reuse guide.
- `API_CONTRACT.md` - Backend API contract expected by the extracted JavaScript.

No backend service code, ML code, datasets, trained models, result CSV files, molecule assets, SHAP logic, or batch prediction core files are included.

## How To Reuse In A New Flask Project

1. Copy this folder's `templates/index.html` into the new project's `templates/index.html`.
2. Copy this folder's `static/css/app.css` into the new project's `static/css/app.css`.
3. Copy this folder's `static/js/app.js` into the new project's `static/js/app.js`.
4. Keep Flask static routing enabled.
5. Keep these template references unless you intentionally rename the files:

```html
<link rel="stylesheet" href="{{ url_for('static', filename='css/app.css') }}">
<script src="{{ url_for('static', filename='js/app.js') }}" defer></script>
```

6. Implement the backend endpoints documented in `API_CONTRACT.md`.

## File Responsibilities

- Layout: `templates/index.html`
- Styling: `static/css/app.css`
- Frontend behavior and API calls: `static/js/app.js`

## External Browser Dependencies

The template uses:

- Google Fonts CDN for `Space Grotesk` and `IBM Plex Mono`.
- Chart.js CDN for the Explain AI bar chart.

If the new environment is offline, replace those CDN references with locally hosted equivalents.

## Intentional Exclusions

The extracted UI does not include:

- trained model files
- datasets
- generated prediction files
- result CSV files
- molecule structure assets
- Flask backend services
- model loading logic
- feature building logic
- SHAP implementation
- batch prediction implementation
- chat backend implementation

The new project should provide its own backend, dataset, model, and prediction logic while preserving the same endpoint names where convenient.
