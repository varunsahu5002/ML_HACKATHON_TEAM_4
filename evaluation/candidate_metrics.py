
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

    if len(y_true) == 0:
        raise ValueError("Input observations cannot be empty.")

    if not (
        len(y_true) == len(candidate_sets)
        == len(top1_predictions)
    ):
        raise ValueError("All input lists must have equal lengths.")

    if any(actual is None for actual in y_true):
        raise ValueError("Ground-truth road IDs cannot be missing.")

    normalized_sets = [set([] if candidates is None else list(candidates)) for candidates in candidate_sets]
    for candidates, prediction in zip(normalized_sets, top1_predictions):
        if candidates and prediction not in candidates:
            raise ValueError("Each top-1 prediction must be a member of its candidate set.")
        if not candidates and prediction is not None:
            raise ValueError("A prediction must be None when its candidate set is empty.")

    # Check whether the actual segment exists in each generated candidate set.
    candidate_hits = [
        actual in candidates
        for actual, candidates in zip(y_true, normalized_sets)
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
        "candidate_hit_count": sum(candidate_hits),
        "candidate_miss_count": len(y_true) - sum(candidate_hits),
        "top1_accuracy": top1_accuracy,
        "candidate_miss_rate": candidate_miss_rate
    }
