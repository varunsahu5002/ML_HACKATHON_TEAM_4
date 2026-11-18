
from evaluation.evaluation_metrics import evaluate_predictions


def evaluate_model(model_name, y_true, y_pred, labels=None):
    """
    Evaluate a model and return its performance results.
    """

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        labels=labels
    )

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
