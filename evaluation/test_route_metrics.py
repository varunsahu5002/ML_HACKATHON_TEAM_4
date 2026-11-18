
from evaluation.route_metrics import evaluate_route_continuity

# Sample predicted route
predicted_sequences = [
    [101, 102, 103, 104, 105]
]

# Sample directed road-segment connections
connected_pairs = {
    (101, 102),
    (102, 103),
    (103, 104),
    # (104, 105) is intentionally missing
}

results = evaluate_route_continuity(
    predicted_sequences,
    connected_pairs
)

print("\nROUTE CONTINUITY EVALUATION")
print("---------------------------")

for metric, value in results.items():
    print(f"{metric}: {value}")
