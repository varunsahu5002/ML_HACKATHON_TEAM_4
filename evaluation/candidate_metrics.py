
def evaluate_candidate_generation(
    y_true,
    candidate_sets,
    top1_predictions
):
    """
    Evaluate candidate generation for each observation.

    y_true: Actual road segment for each observation.
    candidate_sets: Generated candidate road segments.
    top1_predictions: Highest-ranked predicted candidate.
    """

    if not y_true:
        raise ValueError("Input observations cannot be empty.")

    if not (
        len(y_true) == len(candidate_sets)
        == len(top1_predictions)
    ):
        raise ValueError("All input lists must have equal lengths.")

    # Check whether the actual segment exists
    # in each generated candidate set.
    candidate_hits = [
        actual in candidates
        for actual, candidates in zip(y_true, candidate_sets)
    ]

    # Check whether the top-ranked prediction is correct.
    top1_hits = [
        actual == predicted
        for actual, predicted in zip(y_true, top1_predictions)
    ]

    candidate_miss_rate = (
        1 - sum(candidate_hits) / len(y_true)
    )

    top1_accuracy = (
        sum(top1_hits) / len(y_true)
    )

    return {
        "observation_count": len(y_true),
        "top1_accuracy": top1_accuracy,
        "candidate_miss_rate": candidate_miss_rate
    }
