
def calculate_edit_distance(actual, predicted):
    """Calculate Levenshtein edit distance between two sequences."""

    rows = len(actual) + 1
    cols = len(predicted) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):

            if actual[i - 1] == predicted[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


def evaluate_sequences(actual_sequences, predicted_sequences):
    """
    Evaluate complete predicted routes against actual routes.

    Both inputs must contain one sequence per trip, in matching order.
    """

    if not actual_sequences:
        raise ValueError("Actual sequences cannot be empty.")

    if len(actual_sequences) != len(predicted_sequences):
        raise ValueError("Both inputs must contain the same number of trips.")

    if any(not sequence for sequence in actual_sequences):
        raise ValueError("Actual sequences cannot be empty.")

    if any(not sequence for sequence in predicted_sequences):
        raise ValueError("Predicted sequences cannot be empty.")

    exact_matches = 0
    similarities = []

    for actual, predicted in zip(
        actual_sequences, predicted_sequences
    ):

        if actual == predicted:
            exact_matches += 1

        distance = calculate_edit_distance(actual, predicted)

        max_length = max(len(actual), len(predicted))

        similarity = 1 - (distance / max_length)

        similarities.append(similarity)

    return {
        "number_of_trips": len(actual_sequences),
        "exact_sequence_match_rate": (
            exact_matches / len(actual_sequences)
        ),
        "average_sequence_similarity": (
            sum(similarities) / len(similarities)
        )
    }
