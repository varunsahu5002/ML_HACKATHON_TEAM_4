"""Generate a pooled out-of-fold confusion matrix for the RF baseline."""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from evaluation.evaluation_metrics import evaluate_predictions
from model_selection import make_group_splits
from models.random_forest_model import build_model

DATA_PATH = "training_features.csv"
FEATURE_COLUMNS = [
    "distance_to_road_m",
    "perpendicular_distance_m",
    "road_length_m",
    "gps_movement_distance_m",
    "gps_movement_bearing_deg",
    "gps_speed_mps",
    "bearing_difference_deg",
]
TARGET_COLUMN = "target_label"
GROUP_COLUMN = "trip_id"
N_SPLITS = 5


def _preserve_legacy(path):
    """Keep the checked-in Phase 1 artifact before replacing it on a real run."""
    preserved = path.with_name(f"{path.stem}_phase1_preserved{path.suffix}")
    if path.exists() and not preserved.exists():
        shutil.copy2(path, preserved)


def main():
    print("Loading development training features...")
    df = pd.read_csv(DATA_PATH)
    if "partition" in df.columns:
        partitions = set(df["partition"].dropna().astype(str).str.lower())
        if partitions != {"train"}:
            raise ValueError(
                "training_features.csv must contain training trips only; "
                f"found partition values {sorted(partitions)}."
            )
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Training feature matrix is missing columns: {sorted(missing)}")
    df = df.dropna(subset=[TARGET_COLUMN, GROUP_COLUMN]).copy()
    X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    groups = df[GROUP_COLUMN].astype(str).to_numpy()
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Training feature matrix must contain both labels 0 and 1.")

    splits = make_group_splits(groups, n_splits=N_SPLITS)
    print("Rows:", len(df))
    print("Trips:", pd.Series(groups).nunique())
    print("Classes:", sorted(np.unique(y).tolist()))

    oof_predictions = np.empty(y.size, dtype=y.dtype)
    predicted_mask = np.zeros(y.size, dtype=bool)
    oof_probabilities = np.full((y.size, 2), np.nan, dtype=float)
    training_time_seconds = 0.0
    prediction_time_seconds = 0.0
    validation_trip_counts = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        model = build_model()
        validation_trip_counts.append(int(pd.Series(groups[val_idx]).nunique()))
        print(
            f"Fold {fold}/{len(splits)}: "
            f"training trips={pd.Series(groups[train_idx]).nunique()}, "
            f"validation trips={validation_trip_counts[-1]}"
        )
        start = perf_counter()
        model.fit(X.iloc[train_idx], y[train_idx])
        training_time_seconds += perf_counter() - start

        start = perf_counter()
        predictions = model.predict(X.iloc[val_idx])
        probabilities = model.predict_proba(X.iloc[val_idx])
        prediction_time_seconds += perf_counter() - start
        oof_predictions[val_idx] = predictions
        predicted_mask[val_idx] = True
        classes = list(model.classes_)
        oof_probabilities[val_idx, 0] = probabilities[:, classes.index(0)]
        oof_probabilities[val_idx, 1] = probabilities[:, classes.index(1)]

    if not predicted_mask.all():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")
    metrics = evaluate_predictions(
        y,
        oof_predictions,
        labels=[0, 1],
        y_probabilities=oof_probabilities,
    )
    cm = confusion_matrix(y, oof_predictions, labels=[0, 1])
    for name in (
        "accuracy", "binary_precision", "binary_recall", "binary_f1",
        "macro_precision", "macro_recall", "macro_f1", "log_loss",
    ):
        print(f"{name}: {metrics[name]}")
    print(f"training_time_seconds: {training_time_seconds}")
    print(f"prediction_time_seconds: {prediction_time_seconds}")
    print(f"prediction_time_seconds_per_row: {prediction_time_seconds / len(X)}")

    cm_csv_path = Path("confusion_matrix.csv")
    cm_png_path = Path("confusion_matrix.png")
    config_path = Path("model_config.json")
    for path in (cm_csv_path, cm_png_path, config_path):
        _preserve_legacy(path)

    pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    ).to_csv(cm_csv_path)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest - 5-Fold GroupKFold Out-of-Fold")
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=200)
    plt.close()

    classifier = model.named_steps["classifier"]
    config = {
        "model_evaluated": "Random Forest",
        "model_type": type(classifier).__name__,
        "model_parameters": classifier.get_params(),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "validation_method": "5-fold GroupKFold with pooled out-of-fold predictions",
        "cv_folds": len(splits),
        "validation_trip_counts_by_fold": validation_trip_counts,
        "training_partition": "train only, through training_features.csv",
        "student_test_accessed": False,
        "evaluator_test_accessed": False,
        "metrics": {
            key: metrics[key]
            for key in (
                "accuracy", "binary_precision", "binary_recall", "binary_f1",
                "macro_precision", "macro_recall", "macro_f1", "log_loss",
            )
        },
        "top1_accuracy": None,
        "candidate_miss_rate": None,
        "route_continuity": None,
        "training_time_seconds": training_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
        "prediction_time_seconds_per_row": prediction_time_seconds / len(X),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "confusion_matrix_labels": {"0": "Negative candidate", "1": "Positive candidate"},
        "interpretation": "Cross-validation diagnostics on training trips; not an independent test score.",
    }
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, default=str)

    print("\nCreated:")
    print(f"- {cm_csv_path}")
    print(f"- {cm_png_path}")
    print(f"- {config_path}")


if __name__ == "__main__":
    main()
