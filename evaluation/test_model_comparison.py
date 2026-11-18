
from evaluation.model_evaluation import evaluate_model
from evaluation.model_comparison import (
    compare_models,
    display_comparison
)

# Sample actual labels
y_true = [10, 20, 30, 40, 50, 10]

# Sample predictions from two models
model_a_predictions = [10, 20, 25, 40, 50, 10]
model_b_predictions = [10, 20, 30, 40, 50, 10]

# Evaluate both models
model_a = evaluate_model(
    "Sample Model A",
    y_true,
    model_a_predictions
)

model_b = evaluate_model(
    "Sample Model B",
    y_true,
    model_b_predictions
)

# Compare models
comparison = compare_models(
    [model_a, model_b],
    ranking_metric="macro_f1"
)

# Display comparison
display_comparison(comparison)
