
from evaluation.candidate_metrics import evaluate_candidate_generation

# Sample actual road segments
y_true = [101, 202, 303, 404, 505]

# Candidates generated for each observation
candidate_sets = [
    [101, 102, 103],
    [201, 202, 203],
    [301, 302, 303],
    [401, 402, 403],
    [501, 502, 503]
]

# Highest-ranked candidate predicted by the model
top1_predictions = [101, 202, 302, 402, 501]

results = evaluate_candidate_generation(
    y_true,
    candidate_sets,
    top1_predictions
)

print("\nCANDIDATE GENERATION EVALUATION")
print("--------------------------------")

for metric, value in results.items():
    print(f"{metric}: {value}")
