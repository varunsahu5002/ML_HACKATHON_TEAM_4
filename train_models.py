
"""Train, compare, select and save the best ML model."""

import os
import json
import shutil
import hashlib
import platform
from time import perf_counter
import pandas as pd
import sklearn

from model_selection import evaluate_models
from model_training import build_models
from model_utils import save_model


DATA_FILE = "training_features.csv"
RESULT_FILE = "model_comparison.csv"
LEGACY_RESULT_FILE = "model_comparison_phase1_preserved.csv"
EXPERIMENT_CONFIG_FILE = "baseline_experiment_config.json"
N_SPLITS = 5
PRIMARY_METRIC = "macro_f1"


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

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

LEAKAGE_NAME_PARTS = (
    "ground_truth", "true_road", "matched_road", "reference_road",
    "target", "route_", "alignment_error", "match", "label",
)


def select_feature_columns(data):
    """Select numeric inputs while excluding identifiers and target-derived fields."""
    feature_columns = []
    for column in data.columns:
        name = str(column).lower()
        if column in EXCLUDED_COLUMNS or name.endswith("_id") or name in {"id", "trip", "observation", "candidate"}:
            continue
        if any(part in name for part in LEAKAGE_NAME_PARTS):
            continue
        converted = pd.to_numeric(data[column], errors="coerce").replace(
            [float("inf"), float("-inf")], float("nan")
        )
        if converted.notna().any():
            data[column] = converted
            feature_columns.append(column)
    return feature_columns


def main():
    print("=" * 60)
    print("PATHSNAP - ML MODEL TRAINING")
    print("Training data only")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. This checkout has no training-matrix builder; "
            "restore the training-only label/feature-generation workflow before training."
        )

    data = pd.read_csv(DATA_FILE)

    if "partition" in data.columns:
        partitions = set(data["partition"].dropna().astype(str).str.lower())
        if partitions != {"train"}:
            raise ValueError(
                "training_features.csv must contain training trips only; "
                f"found partition values {sorted(partitions)}."
            )

    if "target_label" not in data.columns:
        raise ValueError("target_label column is missing.")

    if "trip_id" not in data.columns:
        raise ValueError("trip_id column is missing for grouped CV.")

    data = data.dropna(subset=["target_label", "trip_id"]).copy()
    data["target_label"] = data["target_label"].astype(int)

    if data["target_label"].nunique() != 2:
        raise ValueError("Training data must contain both labels 0 and 1.")

    # Select numeric prediction features only.
    feature_columns = select_feature_columns(data)

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

    # Candidate-level top-1 can be calculated from out-of-fold scores when the
    # grouping columns survived training-data construction. Miss rate cannot be
    # recovered from a feature table that has already dropped candidate misses.
    observation_ids = None
    candidate_road_ids = None
    if {"observation_id", "candidate_road_id"}.issubset(data.columns):
        observation_ids = (
            data["trip_id"].astype(str) + "::" + data["observation_id"].astype(str)
        ).to_numpy()
        candidate_road_ids = data["candidate_road_id"].to_numpy()

    # Compare five models using complete-trip GroupKFold and pooled OOF metrics.
    models = build_models()
    results = evaluate_models(
        X,
        y,
        groups,
        n_splits=N_SPLITS,
        observation_ids=observation_ids,
        candidate_road_ids=candidate_road_ids,
    )

    # Keep the checked-in Phase 1 table intact before the first corrected run.
    if os.path.exists(RESULT_FILE) and not os.path.exists(LEGACY_RESULT_FILE):
        shutil.copy2(RESULT_FILE, LEGACY_RESULT_FILE)

    results.to_csv(RESULT_FILE, index=False)

    print("\nMODEL COMPARISON")
    print(results.to_string(index=False))
    print(f"\nComparison saved to: {RESULT_FILE}")

    # Select by pooled out-of-fold Macro F1 (both classes weighted equally).
    best_name = results.iloc[0]["model"]
    print(f"\nBest baseline by pooled OOF Macro F1: {best_name}")

    # Refit selected model on all available training trips.
    best_model = build_models()[best_name]
    refit_start = perf_counter()
    best_model.fit(X, y)
    final_refit_time_seconds = perf_counter() - refit_start

    metrics = results.iloc[0].to_dict()

    save_model(
        best_model,
        feature_columns,
        best_name,
        metrics
    )

    experiment_config = {
        "training_data_file": DATA_FILE,
        "training_data_sha256": _sha256_file(DATA_FILE),
        "partition": "train only",
        "student_test_used": False,
        "evaluator_test_used": False,
        "partition_column_present": "partition" in data.columns,
        "partition_values": sorted(set(data["partition"].dropna().astype(str).tolist()))
        if "partition" in data.columns else None,
        "partition_source_note": "If no partition column exists, provenance must be guaranteed by the training-only matrix builder.",
        "group_column": "trip_id",
        "validation": {
            "method": "GroupKFold",
            "n_splits": N_SPLITS,
            "shuffle": False,
            "metric_aggregation": "pooled out-of-fold predictions",
        },
        "primary_selection_metric": PRIMARY_METRIC,
        "metric_scope": {
            "top1_accuracy": "computed only when observation_id and candidate_road_id are present",
            "candidate_miss_rate": "not estimable from filtered candidate rows",
            "route_continuity": "not estimable without decoded routes and road adjacency",
        },
        "feature_columns": feature_columns,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "model_configurations": {
            name: model.get_params(deep=True) for name, model in models.items()
        },
        "selected_model": best_name,
        "selected_model_final_refit_time_seconds": final_refit_time_seconds,
    }
    with open(EXPERIMENT_CONFIG_FILE, "w", encoding="utf-8") as config_file:
        json.dump(experiment_config, config_file, indent=2, default=str)

    print("\nTRAINING COMPLETE")
    print("Saved model: artifacts/best_model.joblib")
    print(f"Saved corrected comparison: {RESULT_FILE}")
    print(f"Saved model/fold configuration: {EXPERIMENT_CONFIG_FILE}")


if __name__ == "__main__":
    main()
