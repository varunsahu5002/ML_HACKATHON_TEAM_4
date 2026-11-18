import pandas as pd

from evaluation.model_evaluation import (
    evaluate_model,
    display_model_results
)

from evaluation.model_comparison import (
    compare_models,
    display_comparison
)


def run_evaluation(prediction_file):

    # Read actual model predictions
    df = pd.read_csv(prediction_file)

    required_columns = [
        "actual_road_id",
        "predicted_road_id"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    # Remove rows with missing labels or predictions
    df = df.dropna(subset=required_columns)

    y_true = df["actual_road_id"].tolist()
    y_pred = df["predicted_road_id"].tolist()

    # Evaluate the model
    results = evaluate_model(
        model_name="PathSnap Baseline Model",
        y_true=y_true,
        y_pred=y_pred
    )

    display_model_results(results)

    # Compare models
    comparison = compare_models(
        [results],
        ranking_metric="macro_f1"
    )

    display_comparison(comparison)

    return results, comparison


if __name__ == "__main__":

    run_evaluation(
        "results/validation_predictions.csv"
    )