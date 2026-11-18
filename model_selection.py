
"""Trip-wise cross-validation and model comparison."""

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

from model_training import build_models


def evaluate_models(X, y, groups, n_splits=5):
    unique_groups = pd.Series(groups).nunique()

    if unique_groups < 2:
        raise ValueError("At least two different trips are required.")

    n_splits = min(n_splits, unique_groups)
    cv = GroupKFold(n_splits=n_splits)

    results = []
    models = build_models()

    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}")

        fold_scores = []
        total_train_time = 0
        total_predict_time = 0

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            start = time.time()
            model.fit(X_train, y_train)
            total_train_time += time.time() - start

            start = time.time()
            predictions = model.predict(X_val)
            total_predict_time += time.time() - start

            probabilities = model.predict_proba(X_val)

            positive_index = list(model.classes_).index(1)
            positive_probabilities = probabilities[:, positive_index]

            scores = {
                "accuracy": accuracy_score(y_val, predictions),
                "precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "log_loss": log_loss(
                    y_val,
                    probabilities,
                    labels=model.classes_
                )
            }

            fold_scores.append(scores)
            print(
                f"  Fold {fold}/{n_splits}: "
                f"F1={scores['f1']:.4f}, "
                f"Accuracy={scores['accuracy']:.4f}"
            )

        results.append({
            "model": model_name,
            "accuracy": np.mean([s["accuracy"] for s in fold_scores]),
            "precision": np.mean([s["precision"] for s in fold_scores]),
            "recall": np.mean([s["recall"] for s in fold_scores]),
            "f1": np.mean([s["f1"] for s in fold_scores]),
            "log_loss": np.mean([s["log_loss"] for s in fold_scores]),
            "training_time_seconds": total_train_time,
            "prediction_time_seconds": total_predict_time
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="f1", ascending=False
    ).reset_index(drop=True)

    return results_df
