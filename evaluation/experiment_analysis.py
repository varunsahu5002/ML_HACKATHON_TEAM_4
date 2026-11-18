
from datetime import datetime


def create_experiment_summary(
    experiment_name,
    model_results,
    ranking_metric="macro_f1"
):
    """
    Create a summary of an ML experiment.
    """

    if not model_results:
        raise ValueError("No experiment results provided.")

    summary = {
        "experiment_name": experiment_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "number_of_models": len(model_results),
        "ranking_metric": ranking_metric,
        "models": []
    }

    for result in model_results:
        metrics = result["metrics"]

        summary["models"].append({
            "model_name": result["model_name"],
            "accuracy": metrics.get("accuracy"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1")
        })

    summary["models"].sort(
        key=lambda model: model[ranking_metric],
        reverse=True
    )

    return summary


def display_experiment_summary(summary):
    """Display the experiment summary."""

    print("\nEXPERIMENT SUMMARY")
    print("=" * 60)

    print("Experiment:", summary["experiment_name"])
    print("Date:", summary["date"])
    print("Number of models:", summary["number_of_models"])
    print("Ranking metric:", summary["ranking_metric"])

    print("\nMODEL RESULTS")

    for rank, model in enumerate(summary["models"], start=1):
        print(f"\nRank: {rank}")
        print("Model:", model["model_name"])
        print("Accuracy:", model["accuracy"])
        print("Macro F1:", model["macro_f1"])
        print("Weighted F1:", model["weighted_f1"])

    print("\n" + "=" * 60)
