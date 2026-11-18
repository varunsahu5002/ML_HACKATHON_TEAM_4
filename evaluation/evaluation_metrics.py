
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(y_true, y_pred, labels=None):
    """
    Evaluate model predictions using classification metrics.

    Parameters:
        y_true: Actual road segment labels.
        y_pred: Predicted road segment labels.
        labels: Optional list of class labels.

    Returns:
        Dictionary containing evaluation metrics.
    """

    if len(y_true) == 0:
        raise ValueError("The actual labels cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted labels must have equal lengths.")

    results = {
        "accuracy": accuracy_score(y_true, y_pred),

        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
    }

    return results
