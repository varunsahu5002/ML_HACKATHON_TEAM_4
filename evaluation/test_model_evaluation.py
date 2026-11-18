
from evaluation.model_evaluation import (
    evaluate_model,
    display_model_results
)

# Sample predictions for testing
y_true = [10, 20, 30, 40, 50, 10]
y_pred = [10, 20, 25, 40, 50, 10]

# Evaluate the sample model
results = evaluate_model(
    model_name="Sample Model",
    y_true=y_true,
    y_pred=y_pred
)

# Display evaluation results
display_model_results(results)
