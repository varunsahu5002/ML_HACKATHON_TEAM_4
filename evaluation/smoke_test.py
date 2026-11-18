from evaluation.evaluation_metrics import evaluate_predictions

# Sample actual and predicted road segment labels
y_true = [10, 20, 30, 40, 50, 10]
y_pred = [10, 20, 25, 40, 50, 10]

# Evaluate predictions
results = evaluate_predictions(y_true, y_pred)

# Display results
for metric, value in results.items():
    print(f"\n{metric}:")
    print(value)