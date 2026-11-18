
"""Train, compare, select and save the best ML model."""

import os
import pandas as pd

from model_selection import evaluate_models
from model_training import build_models
from model_utils import save_model


DATA_FILE = "training_features.csv"
RESULT_FILE = "model_comparison.csv"

# These columns identify rows or contain information unavailable
# when making predictions. They must not be model inputs.
EXCLUDED_COLUMNS = {
    "target_label",
    "trip_id",
    "observation_id",
    "observation_index",
    "candidate_road_id",
    "alignment_error_m",
    "ground_truth_road_id"
}


def main():
    print("=" * 60)
    print("PATHSNAP - ML MODEL TRAINING")
    print("Training data only")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Wait for build_training_data.py."
        )

    data = pd.read_csv(DATA_FILE)

    if "target_label" not in data.columns:
        raise ValueError("target_label column is missing.")

    if "trip_id" not in data.columns:
        raise ValueError("trip_id column is missing for grouped CV.")

    data = data.dropna(subset=["target_label", "trip_id"]).copy()
    data["target_label"] = data["target_label"].astype(int)

    if data["target_label"].nunique() != 2:
        raise ValueError("Training data must contain both labels 0 and 1.")

    # Select numeric prediction features only.
    feature_columns = []

    for column in data.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        converted = pd.to_numeric(data[column], errors="coerce")

        if converted.notna().any():
            data[column] = converted
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No usable numeric feature columns found.")

    X = data[feature_columns].replace(
        [float("inf"), float("-inf")], float("nan")
    )
    y = data["target_label"]
    groups = data["trip_id"].astype(str)

    print(f"Training rows: {len(data)}")
    print(f"Training trips: {groups.nunique()}")
    print(f"Features: {len(feature_columns)}")
    print(f"Positive labels: {(y == 1).sum()}")
    print(f"Negative labels: {(y == 0).sum()}")

    # Compare five models using trip-wise cross-validation.
    results = evaluate_models(X, y, groups, n_splits=5)

    results.to_csv(RESULT_FILE, index=False)

    print("\nMODEL COMPARISON")
    print(results.to_string(index=False))
    print(f"\nComparison saved to: {RESULT_FILE}")

    # Select the highest mean cross-validation F1 model.
    best_name = results.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    # Refit selected model on all available training trips.
    best_model = build_models()[best_name]
    best_model.fit(X, y)

    metrics = results.iloc[0].to_dict()

    save_model(
        best_model,
        feature_columns,
        best_name,
        metrics
    )

    print("\nTRAINING COMPLETE")
    print("Saved model: artifacts/best_model.joblib")
    print("Saved comparison: model_comparison.csv")


if __name__ == "__main__":
    main()
