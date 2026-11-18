
def evaluate_route_continuity(predicted_sequences, connected_pairs):
    """
    Evaluate whether consecutive predicted road segments
    are connected in the road network.

    predicted_sequences:
        List of predicted road-segment sequences.

    connected_pairs:
        Set of directed (from_segment, to_segment) pairs
        that are connected in the road network.
    """

    if not predicted_sequences:
        raise ValueError("No predicted routes provided.")

    connected_pairs = set(connected_pairs)

    total_transitions = 0
    connected_transitions = 0

    for sequence in predicted_sequences:

        if len(sequence) < 2:
            continue

        for i in range(len(sequence) - 1):

            current_segment = sequence[i]
            next_segment = sequence[i + 1]

            total_transitions += 1

            if (current_segment, next_segment) in connected_pairs:
                connected_transitions += 1

    if total_transitions == 0:
        raise ValueError("No consecutive road segments to evaluate.")

    continuity = connected_transitions / total_transitions

    return {
        "total_transitions": total_transitions,
        "connected_transitions": connected_transitions,
        "disconnected_transitions": (
            total_transitions - connected_transitions
        ),
        "route_continuity": continuity
    }
