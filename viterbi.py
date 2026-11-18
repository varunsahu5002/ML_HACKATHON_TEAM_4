"""
PathSnap — Log-Space Viterbi Decoder (Deliverables Section 5)

Decodes the most likely sequence of hidden states (candidate road arc_ids)
given variable-length per-step emission log-probabilities and pairwise
transition log-probabilities.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def viterbi_decode(
    log_emissions: List[np.ndarray],
    log_transitions: List[np.ndarray],
    log_init: Optional[np.ndarray] = None,
) -> Tuple[List[int], float]:
    """
    Run Viterbi dynamic programming in log-space over a sequence of T steps,
    where step t has K_t candidate states.

    Parameters
    ----------
    log_emissions : list of 1D np.ndarray
        Length T; element t has shape (K_t,) with log emission scores.
        An empty array (K_t == 0) denotes an observation with no candidates.
    log_transitions : list of 2D np.ndarray
        Length max(0, T - 1); element t has shape (K_t, K_{t+1}) with
        log transition scores from step t to step t+1.
    log_init : 1D np.ndarray, optional
        Initial log state prior of shape (K_0,). Uniform if None.

    Returns
    -------
    (best_indices, best_score) : (list[int], float)
        Optimal candidate index at each step t (-1 if K_t == 0), and total log score.
    """
    T = len(log_emissions)
    if T == 0:
        return [], 0.0

    # Handle leading empty-candidate steps cleanly
    dp: List[np.ndarray] = []
    backpointers: List[np.ndarray] = []

    prev_valid_t: Optional[int] = None
    for t in range(T):
        k_t = len(log_emissions[t])
        if k_t == 0:
            dp.append(np.array([], dtype=np.float64))
            backpointers.append(np.array([], dtype=np.int64))
            continue

        if prev_valid_t is None:
            prior = log_init if (log_init is not None and len(log_init) == k_t) else np.zeros(k_t, dtype=np.float64)
            dp.append(prior + log_emissions[t])
            backpointers.append(np.full(k_t, -1, dtype=np.int64))
        elif prev_valid_t == t - 1 and len(log_transitions[t - 1]) > 0:
            # Standard transition from t-1 -> t: shape (K_{t-1}, K_t)
            trans = log_transitions[t - 1]
            scores = dp[t - 1][:, None] + trans  # (K_{t-1}, K_t)
            best_prev = np.argmax(scores, axis=0)
            best_val = scores[best_prev, np.arange(k_t)] + log_emissions[t]
            dp.append(best_val)
            backpointers.append(best_prev.astype(np.int64))
        else:
            # Preceding step(s) had 0 candidates: restart uniformly from emission
            dp.append(log_emissions[t].copy())
            backpointers.append(np.full(k_t, -1, dtype=np.int64))

        prev_valid_t = t

    if prev_valid_t is None:
        return [-1] * T, 0.0

    # Backtrack from the last step that had candidates
    best_indices = [-1] * T
    last_idx = int(np.argmax(dp[prev_valid_t]))
    best_score = float(dp[prev_valid_t][last_idx])

    cur_idx = last_idx
    for t in range(prev_valid_t, -1, -1):
        if len(dp[t]) == 0:
            best_indices[t] = -1
            continue
        if cur_idx < 0 or cur_idx >= len(dp[t]):
            cur_idx = int(np.argmax(dp[t]))
        best_indices[t] = cur_idx
        cur_idx = int(backpointers[t][cur_idx])

    return best_indices, best_score