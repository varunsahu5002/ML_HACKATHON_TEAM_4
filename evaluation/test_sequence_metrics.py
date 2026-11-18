
from evaluation.sequence_metrics import evaluate_sequences

# Sample actual routes for three trips
actual_sequences = [
    [101, 102, 103, 104],
    [201, 202, 203],
    [301, 302, 303]
]

# Sample predicted routes
predicted_sequences = [
    [101, 102, 103, 104],
    [201, 202, 204],
    [301, 302, 305]
]

results = evaluate_sequences(
    actual_sequences,
    predicted_sequences
)

print("\nSEQUENCE-LEVEL EVALUATION")
print("-------------------------")

for metric, value in results.items():
    print(f"{metric}: {value}")
