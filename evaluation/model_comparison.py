
def compare_models(model_results, ranking_metric="macro_f1"):
    """
    Compare multiple models using their evaluation metrics.

    model_results: List of results returned by evaluate_model().
    ranking_metric: Metric used to rank the models.
    """

    if not model_results:
        raise ValueError("No model results provided.")

    comparison = []

    for result in model_results:
        metrics = result["metrics"]

        if ranking_metric not in metrics:
            raise ValueError(
                f"Metric '{ranking_metric}' is not available."
            )

        comparison.append({
            "model_name": result["model_name"],
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("binary_precision"),
            "recall": metrics.get("binary_recall"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "macro_f1": metrics.get("macro_f1"),
            "binary_f1": metrics.get("binary_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "log_loss": metrics.get("log_loss"),
            "top1_accuracy": metrics.get("top1_accuracy"),
            "candidate_miss_rate": metrics.get("candidate_miss_rate"),
            "route_continuity": metrics.get("route_continuity"),
            "training_time_seconds": metrics.get("training_time_seconds"),
            "prediction_time_seconds": metrics.get("prediction_time_seconds"),
        })

    if any(row.get(ranking_metric) is None for row in comparison):
        raise ValueError(f"Metric '{ranking_metric}' is missing for one or more models.")
    comparison.sort(
        key=lambda row: float(row[ranking_metric]),
        reverse=True
    )

    return comparison


def display_comparison(comparison):
    """Display model comparison in a readable format."""

    print("\nMODEL COMPARISON")
    print("=" * 75)

    for rank, model in enumerate(comparison, start=1):
        print(f"\nRank: {rank}")
        print(f"Model: {model['model_name']}")

        for metric, value in model.items():
            if metric != "model_name":
                print(f"{metric}: {value}")

    print("\n" + "=" * 75)
