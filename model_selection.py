"""Reproducible trip-grouped cross-validation and baseline comparison."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from evaluation.evaluation_metrics import evaluate_predictions
from model_training import build_models


def make_group_splits(groups, n_splits=5):
    """Return deterministic GroupKFold splits and enforce trip-level isolation."""
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or group_values.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional sequence.")
    group_series = pd.Series(group_values)
    if group_series.isna().any():
        raise ValueError("Trip groups cannot contain missing values.")
    n_groups = int(group_series.nunique())
    if n_groups < 2:
        raise ValueError("At least two different trips are required.")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    n_splits = min(n_splits, n_groups)

    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(np.zeros(group_values.size), groups=group_values))
    validation_counts = np.zeros(group_values.size, dtype=np.int8)
    for train_idx, val_idx in splits:
        train_groups = set(group_values[train_idx].tolist())
        val_groups = set(group_values[val_idx].tolist())
        if train_groups.intersection(val_groups):
            raise RuntimeError("GroupKFold placed the same trip in train and validation.")
        validation_counts[val_idx] += 1
    if not np.all(validation_counts == 1):
        raise RuntimeError("Every row must appear in exactly one validation fold.")
    return splits


def _observation_top1_accuracy(y, positive_probabilities, observation_ids):
    """Top-scoring candidate accuracy on observations represented in candidate rows."""
    groups = {}
    for row_idx, observation_id in enumerate(observation_ids):
        groups.setdefault(observation_id, []).append(row_idx)
    if not groups:
        return np.nan
    correct = 0
    labels = np.asarray(y)
    scores = np.asarray(positive_probabilities)
    for indices in groups.values():
        # np.argmax is stable, so equal scores select the first candidate row.
        best_idx = indices[int(np.argmax(scores[indices]))]
        correct += int(labels[best_idx] == 1)
    return correct / len(groups)


def evaluate_models(
    X,
    y,
    groups,
    n_splits=5,
    observation_ids=None,
    candidate_road_ids=None,
):
    """Compare all configured models using out-of-fold, row-level aggregates.

    GroupKFold holds out complete trips. Each row receives one out-of-fold
    prediction, and aggregate metrics are calculated once over all held-out rows
    (rather than averaging folds of unequal size). ``top1_accuracy`` is computed
    only when candidate observations are supplied. Candidate-miss rate and route
    continuity remain NaN here because filtered candidate rows do not identify
    omitted-reference-road observations and no decoder/network is passed in.
    """
    y_array = np.asarray(y)
    group_array = np.asarray(groups)
    if len(X) != y_array.size or y_array.size != group_array.size:
        raise ValueError("X, y, and groups must have equal row counts.")
    if y_array.ndim != 1 or y_array.size == 0:
        raise ValueError("y must be a non-empty one-dimensional sequence.")
    if set(np.unique(y_array).tolist()) != {0, 1}:
        raise ValueError("Baseline candidate classification expects labels 0 and 1.")
    if (observation_ids is None) != (candidate_road_ids is None):
        raise ValueError("observation_ids and candidate_road_ids must be provided together.")
    if observation_ids is not None:
        observation_ids = np.asarray(observation_ids, dtype=object)
        candidate_road_ids = np.asarray(candidate_road_ids, dtype=object)
        if len(observation_ids) != y_array.size or len(candidate_road_ids) != y_array.size:
            raise ValueError("Candidate metadata must have one value per feature row.")
        if pd.isna(observation_ids).any() or pd.isna(candidate_road_ids).any():
            raise ValueError("Candidate observation and road IDs cannot be missing.")

    splits = make_group_splits(group_array, n_splits=n_splits)
    models = build_models()
    results = []

    for model_name, template in models.items():
        print(f"\nEvaluating: {model_name}")
        oof_predictions = np.empty(y_array.size, dtype=y_array.dtype)
        predicted_mask = np.zeros(y_array.size, dtype=bool)
        oof_probabilities = np.full((y_array.size, 2), np.nan, dtype=float)
        fold_macro_f1 = []
        training_time = 0.0
        prediction_time = 0.0

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            model = clone(template)
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            start = perf_counter()
            model.fit(X_train, y_train)
            training_time += perf_counter() - start

            start = perf_counter()
            predictions = model.predict(X_val)
            probabilities = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
            prediction_time += perf_counter() - start
            oof_predictions[val_idx] = predictions
            predicted_mask[val_idx] = True
            if probabilities is not None:
                classes = list(model.classes_)
                for col_idx, label in enumerate((0, 1)):
                    if label in classes:
                        oof_probabilities[val_idx, col_idx] = probabilities[:, classes.index(label)]

            fold_metrics = evaluate_predictions(y_val, predictions, labels=[0, 1])
            fold_macro_f1.append(fold_metrics["macro_f1"])
            print(
                f"  Fold {fold}/{len(splits)}: "
                f"Macro-F1={fold_metrics['macro_f1']:.4f}, "
                f"Binary-F1={fold_metrics['binary_f1']:.4f}"
            )

        if not predicted_mask.all():
            raise RuntimeError(f"Incomplete out-of-fold predictions for {model_name}.")
        probabilities_for_metrics = (
            oof_probabilities if np.isfinite(oof_probabilities).all() else None
        )
        metrics = evaluate_predictions(
            y_array,
            oof_predictions,
            labels=[0, 1],
            y_probabilities=probabilities_for_metrics,
        )
        top1 = (
            _observation_top1_accuracy(y_array, oof_probabilities[:, 1], observation_ids)
            if observation_ids is not None and np.isfinite(oof_probabilities).all()
            else np.nan
        )
        results.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["binary_precision"],
            "recall": metrics["binary_recall"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "binary_f1": metrics["binary_f1"],
            "f1": metrics["binary_f1"],  # Legacy column alias; explicitly binary.
            "log_loss": metrics["log_loss"],
            "top1_accuracy": top1,
            "candidate_miss_rate": np.nan,
            "route_continuity": np.nan,
            "macro_f1_fold_mean": float(np.mean(fold_macro_f1)),
            "macro_f1_fold_std": float(np.std(fold_macro_f1, ddof=0)),
            "training_time_seconds": training_time,
            "prediction_time_seconds": prediction_time,
            "prediction_time_seconds_per_row": prediction_time / y_array.size,
            "cv_folds": len(splits),
        })

    return pd.DataFrame(results).sort_values(
        by="macro_f1", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
