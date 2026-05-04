# Data Assets

This folder now contains the organized SynergyLens data assets used by the clean Flask backend:

- `drug_name_id_map.csv`: drug/item IDs and display names for frontend dropdowns
- `drug_fingerprints_lookup.csv`: per-drug fingerprint features
- `cell_line_features_lookup.csv`: per-context/cell-line model features
- `depmap_features_lookup.csv`: DepMap metadata features used as supplemental context
- `depmap_label_encoders.pkl`: saved encoders kept with the data bundle for reproducibility
- `feature_columns.json`: model feature order used at prediction time
- `feature_columns.pkl`: original feature-order pickle retained as a source asset

Prediction input still uses the frontend-compatible names:

- `NSC1`: first item/drug ID
- `NSC2`: second item/drug ID
- `CELLNAME`: context/cell line

The backend builds one model row by combining the selected cell-line row, drug A fingerprints, drug B fingerprints, and pairwise fingerprint sum/diff/mul features. The final DataFrame is ordered exactly by `feature_columns.json`.

To replace this with a different dataset, update these files or edit:

`backend/services/prediction_service.py` -> `build_feature_vector()`

Keep that function returning a one-row pandas DataFrame with columns ordered exactly like `feature_columns.json`.
