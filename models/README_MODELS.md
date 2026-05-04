# Model Asset

The single deployed model lives here:

- `models/model.pkl`

This organized bundle currently uses one joblib-loadable `XGBRegressor` model. It exposes a sklearn-like `.predict()` method and receives a one-row pandas DataFrame.

Prediction columns and order come from:

- `data/feature_columns.json`

The backend does not use a registry or multiple cell-line-specific models. It always loads one deployed model from `models/model.pkl` and caches it after first use.
