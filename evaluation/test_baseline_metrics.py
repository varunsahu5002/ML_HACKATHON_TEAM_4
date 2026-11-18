"""Regression tests for corrected baseline metrics and trip-level CV."""

import numpy as np
import pandas as pd

from evaluation.evaluation_metrics import evaluate_predictions
from evaluation.candidate_metrics import evaluate_candidate_generation
from model_selection import evaluate_models, make_group_splits
from train_models import select_feature_columns


def test_macro_f1_is_not_binary_f1_and_log_loss_is_reported():
    result = evaluate_predictions(
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        labels=[0, 1],
        y_probabilities=[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
    )
    assert result["binary_f1"] == 0.0
    np.testing.assert_allclose(result["macro_f1"], 3 / 7)
    assert result["log_loss"] is not None
    assert result["macro_f1"] != result["binary_f1"]


def test_candidate_miss_rate_and_observation_top1_include_missed_candidates():
    result = evaluate_candidate_generation(
        y_true=[10, 20, 30],
        candidate_sets=[[10, 11], [], [31, 32]],
        top1_predictions=[10, None, 31],
    )
    assert result["candidate_hit_count"] == 1
    assert result["candidate_miss_count"] == 2
    np.testing.assert_allclose(result["candidate_miss_rate"], 2 / 3)
    np.testing.assert_allclose(result["top1_accuracy"], 1 / 3)


def test_group_splits_are_trip_disjoint_and_repeatable():
    groups = np.repeat(["trip-a", "trip-b", "trip-c", "trip-d", "trip-e"], 3)
    first = make_group_splits(groups, n_splits=3)
    second = make_group_splits(groups, n_splits=3)
    assert len(first) == 3
    for (train_a, val_a), (train_b, val_b) in zip(first, second):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(val_a, val_b)
        assert set(groups[train_a]).isdisjoint(set(groups[val_a]))
    assert sorted(np.concatenate([val for _, val in first]).tolist()) == list(range(len(groups)))


def test_numeric_feature_selection_excludes_ground_truth_and_ids():
    frame = pd.DataFrame({
        "distance_to_road_m": [1.0, 2.0],
        "candidate_road_id": [10, 11],
        "ground_truth_road_id": [10, 12],
        "alignment_error_m": [0.1, 0.2],
        "route_seq_idx": [1, 2],
        "target_label": [1, 0],
        "trip_id": ["a", "b"],
        "all_infinite": [np.inf, -np.inf],
    })
    assert select_feature_columns(frame) == ["distance_to_road_m"]


def test_five_model_pipeline_returns_pooled_metrics_and_candidate_top1():
    rows = []
    labels = []
    groups = []
    observations = []
    roads = []
    for trip_idx in range(4):
        for obs_idx in range(4):
            observation = f"trip-{trip_idx}:{obs_idx}"
            for candidate_idx, is_match in enumerate((1, 0)):
                rows.append({
                    "distance_to_road_m": 1.0 if is_match else 35.0,
                    "bearing_difference_deg": 5.0 if is_match else 110.0,
                })
                labels.append(is_match)
                groups.append(f"trip-{trip_idx}")
                observations.append(observation)
                roads.append(obs_idx * 10 + candidate_idx)

    result = evaluate_models(
        pd.DataFrame(rows),
        pd.Series(labels),
        groups,
        n_splits=2,
        observation_ids=observations,
        candidate_road_ids=roads,
    )
    assert len(result) == 5
    required = {
        "accuracy", "precision", "recall", "macro_f1", "binary_f1",
        "log_loss", "top1_accuracy", "candidate_miss_rate",
        "route_continuity", "training_time_seconds", "prediction_time_seconds",
    }
    assert required.issubset(result.columns)
    assert result["macro_f1"].between(0, 1).all()
    assert result["top1_accuracy"].between(0, 1).all()
    assert result["candidate_miss_rate"].isna().all()
    assert result["route_continuity"].isna().all()
