# PathSnap Phase 2 — Model Artifact & Integration Guide for Member 2

## Saved Artifact
- **Path**: `artifacts/best_model.joblib`
- **Format**: Joblib dictionary saved via `model_utils.save_model(...)` with keys:
  - `model`: Fitted `sklearn.pipeline.Pipeline` (`SimpleImputer(strategy="median")` + winning classifier)
  - `feature_columns`: Ordered list of the 7 numeric input feature names
  - `model_name`: String name of the selected model
  - `metrics`: Dictionary of 5-fold trip-wise `GroupKFold` CV metrics

## Required Input Columns (Exact Order)
1. `distance_to_road_m`
2. `perpendicular_distance_m`
3. `road_length_m`
4. `gps_movement_distance_m`
5. `gps_movement_bearing_deg`
6. `gps_speed_mps`
7. `bearing_difference_deg`

## Loading & Inference Snippet
```python
from model_utils import load_model

artifact = load_model("artifacts/best_model.joblib")
model = artifact["model"]
feature_cols = artifact["feature_columns"]

# X_candidates is the DataFrame from feature_engineering.extract_features(...)
candidate_probs = model.predict_proba(X_candidates[feature_cols])[:, 1]
```
