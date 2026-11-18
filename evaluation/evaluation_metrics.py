"""Consistent classification metrics for candidate-road classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    y_true,
    y_pred,
    labels=None,
    y_probabilities=None,
    positive_label=1,
):
    """Calculate classification metrics with an explicit macro/binary split.

    ``y_probabilities`` must be a 2D array whose columns follow ``labels``.
    Log loss is omitted when probabilities are unavailable. Binary F1/precision/
    recall are ``None`` unless exactly two labels, including ``positive_label``,
    are evaluated. Macro metrics always cover the full, explicit label set.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or true.size == 0:
        raise ValueError("Actual and predicted labels must be non-empty 1D arrays.")
    if true.size != pred.size:
        raise ValueError("Actual and predicted labels must have equal lengths.")

    labels_were_provided = labels is not None
    if y_probabilities is not None and not labels_were_provided:
        raise ValueError("labels must be supplied to define probability-column order.")
    if labels is None:
        labels = list(dict.fromkeys(true.tolist() + pred.tolist()))
    else:
        labels = list(labels)
    if not labels:
        raise ValueError("At least one class label is required.")
    if any(label not in labels for label in true.tolist()):
        raise ValueError("labels must include every class present in y_true.")
    if any(label not in labels for label in pred.tolist()):
        raise ValueError("labels must include every class present in y_pred.")

    binary_applicable = len(labels) == 2 and positive_label in labels
    results = {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(precision_score(
            true, pred, labels=labels, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            true, pred, labels=labels, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            true, pred, labels=labels, average="macro", zero_division=0
        )),
        "binary_precision": None,
        "binary_recall": None,
        "binary_f1": None,
        "weighted_f1": float(f1_score(
            true, pred, labels=labels, average="weighted", zero_division=0
        )),
        "log_loss": None,
        "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
        "classification_report": classification_report(
            true, pred, labels=labels, output_dict=True, zero_division=0
        ),
    }
    if binary_applicable:
        results["binary_precision"] = float(precision_score(
            true, pred, average="binary", pos_label=positive_label, zero_division=0
        ))
        results["binary_recall"] = float(recall_score(
            true, pred, average="binary", pos_label=positive_label, zero_division=0
        ))
        results["binary_f1"] = float(f1_score(
            true, pred, average="binary", pos_label=positive_label, zero_division=0
        ))

    if y_probabilities is not None:
        probs = np.asarray(y_probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape != (true.size, len(labels)):
            raise ValueError(
                "y_probabilities must have shape (n_samples, len(labels)) "
                "with columns ordered like labels."
            )
        if not np.isfinite(probs).all():
            raise ValueError("y_probabilities must contain only finite values.")
        results["log_loss"] = float(log_loss(true, probs, labels=labels))
    return results
