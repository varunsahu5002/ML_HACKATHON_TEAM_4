
from evaluation.model_evaluation import evaluate_model
from evaluation.experiment_analysis import (
    create_experiment_summary,
    display_experiment_summary
)

# Sample actual labels
y_true = [10, 20, 30, 40, 50, 10]

# Sample model predictions
model_a_predictions = [10, 20, 25, 40, 50, 10]
model_b_predictions = [10, 20, 30, 40, 50, 10]

# Evaluate models
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

# Create experiment summary
summary = create_experiment_summary(
    experiment_name="PathSnap Sample Experiment",
    model_results=[model_a, model_b]
)

# Display summary
display_experiment_summary(summary)
