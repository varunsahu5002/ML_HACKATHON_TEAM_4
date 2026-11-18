"""
PathSnap — HMM Route Reconstruction Module (Deliverables Section 5)

Combines ML candidate probabilities (from artifacts/best_model.joblib) with
directed road-network transition probabilities and segment-aware Viterbi
decoding (viterbi.py) to reconstruct continuous, realistic vehicle routes.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from candidate_generation import build_road_network, generate_candidates
from data_loader import TripData
from feature_engineering import extract_features
from model_utils import load_model
from preprocessing import haversine_distance, preprocess_trip
from viterbi import viterbi_decode


def _build_directed_graph(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, Tuple[int, int, float]]]:
    """
    Build adjacency list {u: [(v, length_m), ...]} and arc lookup
    {arc_id: (from_node, to_node, length_m)}.
    """
    node_lon = dict(zip(nodes_df["node_id"].astype(int), nodes_df["lon"].astype(float)))
    node_lat = dict(zip(nodes_df["node_id"].astype(int), nodes_df["lat"].astype(float)))

    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    arc_info: Dict[int, Tuple[int, int, float]] = {}

    for _, row in arcs_df.iterrows():
        aid = int(row["arc_id"])
        u = int(row["from_node_id"])
        v = int(row["to_node_id"])
        if u not in node_lon or v not in node_lon:
            continue
        length_m = float(haversine_distance(node_lon[u], node_lat[u], node_lon[v], node_lat[v]))
        adj[u].append((v, length_m))
        arc_info[aid] = (u, v, length_m)

    return adj, arc_info


def _bounded_shortest_path(
    adj: Dict[int, List[Tuple[int, float]]],
    start_node: int,
    target_node: int,
    max_dist_m: float = 1500.0,
) -> Optional[float]:
    """Bounded Dijkstra returning shortest directed distance in meters, or None if unreachable."""
    if start_node == target_node:
        return 0.0

    pq: List[Tuple[float, int]] = [(0.0, start_node)]
    best: Dict[int, float] = {start_node: 0.0}

    while pq:
        d, u = heapq.heappop(pq)
        if d > best.get(u, float("inf")):
            continue
        if u == target_node:
            return d
        if d > max_dist_m:
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd <= max_dist_m and nd < best.get(v, float("inf")):
                best[v] = nd
                heapq.heappush(pq, (nd, v))
    return None


def compute_route_continuity(arc_sequence: List[int], arc_info: Dict[int, Tuple[int, int, float]]) -> float:
    """
    Fraction of consecutive transitions (arc_i -> arc_{i+1}) that are valid
    (either staying on the same arc or arc_i.to_node == arc_{i+1}.from_node).
    """
    valid_seq = [a for a in arc_sequence if a >= 0 and a in arc_info]
    if len(valid_seq) < 2:
        return 1.0
    connected = 0
    for a1, a2 in zip(valid_seq[:-1], valid_seq[1:]):
        if a1 == a2 or arc_info[a1][1] == arc_info[a2][0]:
            connected += 1
    return connected / (len(valid_seq) - 1)


def reconstruct_route(
    trip_data: TripData,
    model_path: str = "artifacts/best_model.joblib",
    radius_m: float = 100.0,
    max_candidates: int = 8,
    transition_scale_m: float = 50.0,
    disconnected_penalty: float = -25.0,
) -> Dict[str, Any]:
    """
    Run full PathSnap HMM + Viterbi route reconstruction on a single trip.

    Parameters
    ----------
    trip_data : TripData
        Raw or preprocessed TripData object.
    model_path : str
        Path to serialized classifier artifact (artifacts/best_model.joblib).
    radius_m : float, default=100.0
        Candidate search radius in meters.
    max_candidates : int, default=8
        Maximum candidate roads per GPS observation.
    transition_scale_m : float, default=50.0
        Exponential scale parameter (beta) for |d_route - d_gps| / beta.
    disconnected_penalty : float, default=-25.0
        Log-transition penalty when consecutive candidates are unreachable in .arcs.

    Returns
    -------
    dict with keys:
        - trip_id
        - obs_predictions (DataFrame with obs_id, segment_id, classifier_arc_id, hmm_arc_id)
        - hmm_route_arcs (deduplicated consecutive arc sequence)
        - classifier_route_arcs (deduplicated consecutive classifier-only sequence)
        - classifier_continuity (float)
        - hmm_continuity (float)
    """
    clean_trip = preprocess_trip(trip_data) if "cleaning_report" not in trip_data.metadata else trip_data
    track = clean_trip.track.copy()
    if len(track) == 0:
        return {
            "trip_id": clean_trip.trip_id,
            "obs_predictions": pd.DataFrame(),
            "hmm_route_arcs": [],
            "classifier_route_arcs": [],
            "classifier_continuity": 1.0,
            "hmm_continuity": 1.0,
        }

    road_net = build_road_network(clean_trip.nodes, clean_trip.arcs)
    obs_df = track.copy()
    obs_df["trip_id"] = clean_trip.trip_id
    obs_df["observation_id"] = obs_df["obs_id"]

    candidates = generate_candidates(
        obs_df,
        road_net,
        radius_m=radius_m,
        max_candidates=max_candidates,
        observation_id_col="observation_id",
        trip_id_col="trip_id",
    )
    feat_df = extract_features(obs_df, candidates, road_net, include_labels=False)
    if isinstance(feat_df, list):
        feat_df = pd.DataFrame(feat_df)

    # Emission probabilities from trained classifier (fallback to distance decay if artifact missing)
    if Path(model_path).is_file() and len(feat_df) > 0:
        artifact = load_model(model_path)
        model = artifact["model"]
        feat_cols = artifact["feature_columns"]
        feat_df["prob"] = model.predict_proba(feat_df[feat_cols])[:, 1]
    elif len(feat_df) > 0:
        feat_df["prob"] = np.exp(-0.5 * (feat_df["distance_to_road_m"] / 25.0) ** 2)

    adj, arc_info = _build_directed_graph(clean_trip.nodes, clean_trip.arcs)

    # Group candidates by observation_id
    cands_by_obs: Dict[int, pd.DataFrame] = {}
    if len(feat_df) > 0:
        feat_df = feat_df[feat_df["candidate_road_id"].astype(int) >= 0].copy()
        for oid, grp in feat_df.groupby("observation_id"):
            cands_by_obs[int(oid)] = grp.sort_values("prob", ascending=False).reset_index(drop=True)

    # Segment-aware Viterbi decoding (never bridge across is_gap / segment_id breaks)
    hmm_arc_map: Dict[int, int] = {}
    clf_arc_map: Dict[int, int] = {}

    for _, seg_df in track.groupby("segment_id", sort=True):
        obs_ids = seg_df["obs_id"].astype(int).tolist()
        step_dists = seg_df["step_dist_m"].to_numpy()

        log_emissions: List[np.ndarray] = []
        cand_ids_per_step: List[List[int]] = []

        for oid in obs_ids:
            grp = cands_by_obs.get(oid)
            if grp is None or len(grp) == 0:
                log_emissions.append(np.array([], dtype=np.float64))
                cand_ids_per_step.append([])
                clf_arc_map[oid] = -1
            else:
                probs = np.clip(grp["prob"].to_numpy(dtype=np.float64), 1e-6, 1.0)
                cids = grp["candidate_road_id"].astype(int).tolist()
                log_emissions.append(np.log(probs))
                cand_ids_per_step.append(cids)
                clf_arc_map[oid] = cids[0]

        log_transitions: List[np.ndarray] = []
        path_cache: Dict[Tuple[int, int], Optional[float]] = {}

        for t in range(len(obs_ids) - 1):
            c1 = cand_ids_per_step[t]
            c2 = cand_ids_per_step[t + 1]
            if not c1 or not c2:
                log_transitions.append(np.empty((len(c1), len(c2)), dtype=np.float64))
                continue

            d_gps = float(step_dists[t + 1])
            mat = np.full((len(c1), len(c2)), disconnected_penalty, dtype=np.float64)
            for i, a1 in enumerate(c1):
                for j, a2 in enumerate(c2):
                    if a1 == a2:
                        mat[i, j] = -abs(0.0 - d_gps) / transition_scale_m
                        continue
                    if a1 not in arc_info or a2 not in arc_info:
                        continue
                    end_node = arc_info[a1][1]
                    start_node = arc_info[a2][0]
                    key = (end_node, start_node)
                    if key not in path_cache:
                        path_cache[key] = _bounded_shortest_path(adj, end_node, start_node, max_dist_m=max(1000.0, d_gps * 3.0))
                    d_route = path_cache[key]
                    if d_route is not None:
                        total_route = d_route + 0.5 * (arc_info[a1][2] + arc_info[a2][2])
                        mat[i, j] = -abs(total_route - d_gps) / transition_scale_m
            log_transitions.append(mat)

        best_idx, _ = viterbi_decode(log_emissions, log_transitions)
        for t, oid in enumerate(obs_ids):
            idx = best_idx[t]
            hmm_arc_map[oid] = cand_ids_per_step[t][idx] if (idx >= 0 and cand_ids_per_step[t]) else -1

    obs_ids_all = track["obs_id"].astype(int).tolist()
    clf_seq = [clf_arc_map.get(oid, -1) for oid in obs_ids_all]
    hmm_seq = [hmm_arc_map.get(oid, -1) for oid in obs_ids_all]

    obs_predictions = pd.DataFrame(
        {
            "obs_id": obs_ids_all,
            "segment_id": track["segment_id"].astype(int).tolist(),
            "classifier_arc_id": clf_seq,
            "hmm_arc_id": hmm_seq,
        }
    )

    def _dedup(seq: List[int]) -> List[int]:
        out: List[int] = []
        for a in seq:
            if a >= 0 and (not out or out[-1] != a):
                out.append(a)
        return out

    return {
        "trip_id": clean_trip.trip_id,
        "obs_predictions": obs_predictions,
        "classifier_route_arcs": _dedup(clf_seq),
        "hmm_route_arcs": _dedup(hmm_seq),
        "classifier_continuity": compute_route_continuity(clf_seq, arc_info),
        "hmm_continuity": compute_route_continuity(hmm_seq, arc_info),
    }