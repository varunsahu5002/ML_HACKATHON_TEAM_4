
from evaluation.evaluation_metrics import evaluate_predictions


def evaluate_model(
    model_name,
    y_true,
    y_pred,
    labels=None,
    y_probabilities=None,
    positive_label=1,
    training_time_seconds=None,
    prediction_time_seconds=None,
):
    """
    Evaluate a model and return its performance results.
    """

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        labels=labels,
        y_probabilities=y_probabilities,
        positive_label=positive_label,
    )
    if training_time_seconds is not None:
        metrics["training_time_seconds"] = float(training_time_seconds)
    if prediction_time_seconds is not None:
        metrics["prediction_time_seconds"] = float(prediction_time_seconds)

    return {
        "model_name": model_name,
        "metrics": metrics
    }


def display_model_results(results):
    """
    Display model evaluation results.
    """

    print("\nMODEL EVALUATION RESULTS")
    print("------------------------")

    print("Model:", results["model_name"])

    for metric, value in results["metrics"].items():
        print(f"\n{metric}:")
        print(value)
