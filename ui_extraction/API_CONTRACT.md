# SynergyLens UI API Contract

The extracted UI calls the endpoints below from `static/js/app.js`. In the new project, these endpoint names can be kept the same while replacing the backend implementation with the new dataset, new model, and new prediction logic. This allows the same UI to work with the new backend with minimal frontend changes.

All JSON endpoints should avoid raw Python tracebacks. Return a clear `error` or `message` field on failure.

## GET `/api/health`

Purpose: Populate the backend status card and top-level asset metrics.

Expected response summary:

```json
{
  "status": "success",
  "available_drugs": 123,
  "available_cell_lines": 45,
  "model_count": 45,
  "feature_column_count": 526,
  "errors": []
}
```

The UI treats `status: "success"` as ready. If unavailable, return `status: "error"` with `message` or `errors`.

## GET `/api/drugs`

Purpose: Populate autocomplete lists for prediction, Explain AI, and molecule lookup.

Accepted response shapes:

```json
["740", "750"]
```

or:

```json
{
  "drugs": [
    { "id": "740", "name": "NSC 740" },
    { "NSC": 750, "drug_name": "Example Drug" }
  ]
}
```

The UI also accepts `items` instead of `drugs`.

## GET `/api/cell-lines`

Purpose: Populate cell-line dropdowns.

Accepted response shapes:

```json
["786-0", "A498"]
```

or:

```json
{
  "cell_lines": ["786-0", "A498"]
}
```

The UI also accepts `items` instead of `cell_lines`.

## POST `/api/predict`

Purpose: Run a single drug-pair prediction.

Request body:

```json
{
  "NSC1": 740,
  "NSC2": 750,
  "CELLNAME": "786-0",
  "drug1_id": "740",
  "drug2_id": "750",
  "cell_line": "786-0"
}
```

Expected response summary:

```json
{
  "status": "success",
  "NSC1": 740,
  "NSC2": 750,
  "CELLNAME": "786-0",
  "model_used": "CatBoost",
  "model_path": "models/example.pkl",
  "prediction_NSC1_to_NSC2": -15.48,
  "prediction_NSC2_to_NSC1": -18.26,
  "final_predicted_COMBOSCORE": -16.87,
  "prediction_label": "Neutral / Weak effect",
  "prediction_category": "neutral",
  "explanation": "Short explanation.",
  "suggestion": "Short suggestion."
}
```

The UI also tolerates `input.NSC1`, `input.NSC2`, `input.CELLNAME`, `final_prediction`, `predicted_comboscore`, `synergy_score`, `score`, `label`, `category`, and `model_name`.

## POST `/api/explain`

Purpose: Generate Explain AI/XAI output only when requested.

Request body is the same model payload as `/api/predict`.

Expected response summary:

```json
{
  "status": "success",
  "input": {
    "NSC1": 740,
    "NSC2": 750,
    "CELLNAME": "786-0"
  },
  "model_used": "CatBoost",
  "final_predicted_COMBOSCORE": -16.87,
  "base_value": 0.12,
  "top_positive_contributors": [
    {
      "readable_feature": "Drug 1 LogP",
      "feature_value": 2.4,
      "impact": 0.31
    }
  ],
  "top_negative_contributors": [
    {
      "readable_feature": "Drug 2 fingerprint pattern 90",
      "feature_value": 1,
      "impact": -0.28
    }
  ],
  "plain_english_explanation": "Readable explanation."
}
```

The UI also accepts `top_synergy_drivers`, `top_antagonism_drivers`, `shap_value`, `shap`, `feature_name`, `feature`, `value`, `prediction`, `expected_value`, `explanation_summary`, and `suggestion`.

## POST `/api/molecule-pair`

Purpose: Load two molecule cards with structure SVG and metadata.

Request body:

```json
{
  "NSC1": 740,
  "NSC2": 750
}
```

Expected response summary:

```json
{
  "status": "success",
  "NSC1": {
    "status": "success",
    "requested_nsc": 740,
    "used_nsc": 740,
    "alias_used": false,
    "molecular_formula": "C10H12N2O",
    "source": "drug_mols.pkl",
    "structure_svg": "<svg>...</svg>"
  },
  "NSC2": {
    "status": "success",
    "requested_nsc": 750,
    "used_nsc": 761431,
    "alias_used": true,
    "molecular_formula": "C20H20O4",
    "source": "sdf",
    "structure_svg": "<svg>...</svg>"
  }
}
```

The UI also accepts `molecule_1`, `molecule_2`, `resolved_nsc`, `used_alias`, `svg`, `molecule_found`, and `found`.

## POST `/api/batch-predict`

Purpose: Upload a CSV and render/download batch prediction output.

Request: `multipart/form-data` with one file field named `file`.

The UI normalizes input columns to:

- `NSC1`
- `NSC2`
- `CELLNAME`

It also accepts uploaded aliases `drug1_id`, `drug1`, `nsc1`, `drug2_id`, `drug2`, `nsc2`, `cell_line`, and `cellname`.

Expected JSON response summary:

```json
{
  "status": "success",
  "total_rows": 10,
  "successful_rows": 9,
  "failed_rows": 1,
  "output_file": "batch_prediction_output.csv",
  "preview": [
    {
      "row_index": 1,
      "NSC1": 740,
      "NSC2": 750,
      "CELLNAME": "786-0",
      "model_used": "CatBoost",
      "prediction_NSC1_to_NSC2": -15.48,
      "prediction_NSC2_to_NSC1": -18.26,
      "final_predicted_COMBOSCORE": -16.87,
      "prediction_label": "Neutral / Weak effect",
      "prediction_category": "neutral",
      "status": "success",
      "error": ""
    }
  ]
}
```

The UI can also handle a direct CSV/blob response instead of JSON.

## POST `/api/chat`

Purpose: Serve the floating Project Assistant and prediction-aware assistant.

Project chat request:

```json
{
  "mode": "project",
  "question": "What is ComboScore?"
}
```

Prediction chat request:

```json
{
  "mode": "prediction",
  "question": "Explain this result",
  "prediction": {
    "input": {
      "NSC1": 740,
      "NSC2": 750,
      "CELLNAME": "786-0"
    },
    "prediction_NSC1_to_NSC2": -15.48,
    "prediction_NSC2_to_NSC1": -18.26,
    "final_predicted_COMBOSCORE": -16.87,
    "label": "neutral",
    "model_used": "CatBoost"
  },
  "explanation": null
}
```

Expected response summary:

```json
{
  "status": "success",
  "answer": "Readable assistant answer.",
  "suggested_questions": ["What should I check next?"],
  "llm_used": false,
  "provider_label": "Built-in Guide"
}
```

## GET `/api/demo-cases`

Purpose: Populate the three demo buttons and optionally prefill the first demo.

Expected response summary:

```json
{
  "status": "success",
  "demo_cases": [
    {
      "case_type": "strong_synergy",
      "NSC1": 740,
      "NSC2": 750,
      "CELLNAME": "786-0",
      "predicted_score": 120.5,
      "label": "Strong Synergy",
      "description": "Most positive demo case."
    },
    {
      "case_type": "neutral",
      "NSC1": 740,
      "NSC2": 752,
      "CELLNAME": "A498"
    },
    {
      "case_type": "antagonism",
      "NSC1": 750,
      "NSC2": 755,
      "CELLNAME": "A549/ATCC"
    }
  ]
}
```

`demo_cases` may be an array or an object whose values are case objects.

## GET `/api/model-performance-summary`

Purpose: Populate the About page performance transparency panel.

Expected response summary:

```json
{
  "status": "success",
  "explanation": "Model summary loaded.",
  "assets": {
    "total_cell_lines": 45,
    "total_drugs": 100,
    "feature_vector": 526,
    "final_model_count": 45
  },
  "model_summary": {
    "total_models": 45,
    "count_per_model_type": {
      "CatBoost": 20,
      "LightGBM": 10
    }
  },
  "performance": {
    "deployed_final_average": {
      "mean_r2_score": 0.42,
      "mean_pearson_rp": 0.68,
      "mean_rmse": 12.4,
      "mean_mae": 8.7
    },
    "by_model_type": [
      {
        "model": "CatBoost",
        "mean_r2_score": 0.45,
        "mean_pearson_rp": 0.7,
        "mean_rmse": 11.9,
        "mean_mae": 8.2
      }
    ]
  }
}
```

## GET `/api/download/<filename>`

Purpose: Download a previously generated batch output file when `/api/batch-predict` returns `output_file`.

Expected behavior: Return a downloadable CSV response for the requested filename. The UI builds the URL by URL-encoding the basename from `output_file`.

## Root Route `/`

Purpose: Render the Flask template.

Expected behavior:

```python
return render_template("index.html")
```
