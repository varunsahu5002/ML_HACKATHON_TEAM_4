"""
PathSnap Phase 2 — Hyperparameter Optimization & Model Selection

Compares Phase 1 Baseline Random Forest against:
  1. Tuned Random Forest (Bayesian HPO)
  2. Baseline HistGradientBoostingClassifier (alternative model family)
  3. Tuned HistGradientBoostingClassifier (Bayesian HPO)

Uses strict trip-wise GroupKFold cross-validation (grouped by `trip_id`)
to prevent observation leakage, evaluates both candidate-level metrics
(Accuracy, Precision, Recall, F1, Log Loss) and observation-level Top-1
Accuracy (selecting the highest-probability candidate per GPS observation),
and saves the winning model artifact via `model_utils.save_model`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from model_utils import save_model

NON_FEATURE_COLS = {
    "target_label",
    "trip_id",
    "observation_id",
    "observation_index",
    "candidate_road_id",
    "alignment_error_m",
    "ground_truth_road_id",
    "road_type",
}


def build_training_features_from_raw(
    data_dir: str = "Student folder",
    output_csv: str = "training_features.csv",
) -> pd.DataFrame:
    """
    Build training_features.csv from raw Student folder/train trips using
    Member 1 (data_loader, preprocessing) and Member 2 (candidate_generation,
    feature_engineering) modules. Never touches evaluator or student_test data.
    """
    from candidate_generation import build_road_network, generate_candidates
    from data_loader import load_dataset
    from feature_engineering import extract_features
    from preprocessing import preprocess_data

    print(f"Building '{output_csv}' from '{data_dir}/train'...")
    raw_train = load_dataset(data_dir, partition="train", load_route=True)
    clean_train = preprocess_data(raw_train)

    trip_frames: List[pd.DataFrame] = []
    for tid, trip in sorted(clean_train.items()):
        if trip.route is None or len(trip.route) == 0 or len(trip.track) == 0:
            continue

        road_net = build_road_network(trip.nodes, trip.arcs)
        obs_df = trip.track.copy()
        obs_df["trip_id"] = tid
        obs_df["observation_id"] = obs_df["obs_id"]

        candidates = generate_candidates(
            obs_df,
            road_net,
            radius_m=100.0,
            max_candidates=8,
            observation_id_col="observation_id",
            trip_id_col="trip_id",
        )

        # Map each observation to its closest candidate road that lies on the reference .route
        route_arcs = set(trip.route["arc_id"].astype(int).tolist())
        gt_by_obs: Dict[Any, int] = {}
        for c in candidates:
            oid = c["observation_id"]
            rid = c["candidate_road_id"]
            if rid in route_arcs and oid not in gt_by_obs:
                gt_by_obs[oid] = int(rid)

        obs_df["ground_truth_road_id"] = obs_df["observation_id"].map(gt_by_obs)
        obs_df = obs_df.dropna(subset=["ground_truth_road_id"]).copy()
        if len(obs_df) == 0:
            continue
        obs_df["ground_truth_road_id"] = obs_df["ground_truth_road_id"].astype(int)

        valid_obs = set(obs_df["observation_id"].tolist())
        valid_cands = [c for c in candidates if c["observation_id"] in valid_obs]
        feat_df = extract_features(obs_df, valid_cands, road_net, include_labels=True)
        if isinstance(feat_df, list):
            feat_df = pd.DataFrame(feat_df)
        if len(feat_df) > 0:
            trip_frames.append(feat_df)

    if not trip_frames:
        raise RuntimeError("No training feature rows could be generated from raw trips.")

    full_df = pd.concat(trip_frames, ignore_index=True)
    full_df.to_csv(output_csv, index=False)
    print(f"Saved {len(full_df)} feature rows across {full_df['trip_id'].nunique()} trips to '{output_csv}'.")
    return full_df


def load_training_table(
    csv_path: str = "training_features.csv",
    data_dir: str = "Student folder",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load training_features.csv; if missing, build it from Student folder/train
    or fall back to feature_matrix_sample.csv if available.
    """
    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
    elif Path(data_dir, "train").is_dir():
        df = build_training_features_from_raw(data_dir=data_dir, output_csv=csv_path)
    elif Path("feature_matrix_sample.csv").exists():
        print("Warning: 'training_features.csv' and 'Student folder/train' not found; using 'feature_matrix_sample.csv'.")
        df = pd.read_csv("feature_matrix_sample.csv")
    else:
        raise FileNotFoundError(
            f"Neither '{csv_path}', '{data_dir}/train', nor 'feature_matrix_sample.csv' was found."
        )

    if "target_label" not in df.columns or "trip_id" not in df.columns:
        raise ValueError("Feature table must contain 'target_label' and 'trip_id' columns.")

    feature_cols = [
        col
        for col in df.columns
        if col not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]
    return df, feature_cols


def compute_observation_top1_accuracy(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    For each (trip_id, observation_id), pick the candidate road with the highest
    predicted positive probability and check whether its target_label == 1.
    """
    if "observation_id" not in df_meta.columns:
        return float("nan")

    eval_df = pd.DataFrame(
        {
            "trip_id": df_meta["trip_id"].values,
            "observation_id": df_meta["observation_id"].values,
            "y_true": y_true,
            "y_prob": y_prob,
        }
    )
    obs_has_pos = eval_df.groupby(["trip_id", "observation_id"])["y_true"].transform("max") > 0
    eval_df = eval_df.loc[obs_has_pos]
    if len(eval_df) == 0:
        return float("nan")

    top_idx = eval_df.groupby(["trip_id", "observation_id"])["y_prob"].idxmax()
    return float(eval_df.loc[top_idx, "y_true"].mean())


def evaluate_cv(
    builder: Callable[[], Pipeline],
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate a model pipeline with trip-wise GroupKFold CV."""
    X = df[feature_cols]
    y = df["target_label"].astype(int).to_numpy()
    groups = df["trip_id"].to_numpy()

    unique_groups = len(np.unique(groups))
    if unique_groups < 2:
        raise ValueError("Need at least 2 distinct trips in training data for GroupKFold CV.")
    gkf = GroupKFold(n_splits=min(n_splits, unique_groups))

    metrics_list: List[Dict[str, float]] = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = builder()
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_va)
        y_prob = model.predict_proba(X_va)[:, 1]
        pred_time = time.perf_counter() - t1

        top1 = compute_observation_top1_accuracy(df.iloc[val_idx], y_va, y_prob)
        metrics_list.append(
            {
                "accuracy": float(accuracy_score(y_va, y_pred)),
                "precision": float(precision_score(y_va, y_pred, zero_division=0)),
                "recall": float(recall_score(y_va, y_pred, zero_division=0)),
                "f1": float(f1_score(y_va, y_pred, zero_division=0)),
                "log_loss": float(log_loss(y_va, np.clip(y_prob, 1e-7, 1 - 1e-7))),
                "top1_accuracy": top1,
                "training_time_seconds": train_time,
                "prediction_time_seconds": pred_time,
            }
        )

    return {k: float(np.nanmean([m[k] for m in metrics_list])) for k in metrics_list[0]}


def build_rf(params: Dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 20),
                    min_samples_leaf=params.get("min_samples_leaf", 2),
                    min_samples_split=params.get("min_samples_split", 2),
                    max_features=params.get("max_features", "sqrt"),
                    class_weight=params.get("class_weight", "balanced"),
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_hgb(params: Dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=params.get("learning_rate", 0.1),
                    max_iter=params.get("max_iter", 100),
                    max_leaf_nodes=params.get("max_leaf_nodes", 31),
                    max_depth=params.get("max_depth", None),
                    min_samples_leaf=params.get("min_samples_leaf", 20),
                    l2_regularization=params.get("l2_regularization", 0.0),
                    class_weight=params.get("class_weight", None),
                    random_state=42,
                ),
            ),
        ]
    )


def run_bayesian_hpo(
    model_family: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run Bayesian HPO via Optuna (TPE sampler) if installed, otherwise fall back
    to deterministic random search over the same parameter space.
    Optimizes composite score: 0.5 * top1_accuracy + 0.5 * f1 - 0.1 * log_loss.
    """
    trials_log: List[Dict[str, Any]] = []

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        has_optuna = True
    except ImportError:
        has_optuna = False

    def sample_params(trial_or_rng: Any) -> Dict[str, Any]:
        if has_optuna:
            t = trial_or_rng
            if model_family == "rf":
                return {
                    "n_estimators": t.suggest_int("n_estimators", 100, 300, step=50),
                    "max_depth": t.suggest_categorical("max_depth", [10, 15, 20, 30, None]),
                    "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
                    "min_samples_split": t.suggest_int("min_samples_split", 2, 10),
                    "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.7]),
                    "class_weight": t.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
                }
            return {
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "max_iter": t.suggest_int("max_iter", 100, 400, step=50),
                "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 63),
                "max_depth": t.suggest_categorical("max_depth", [4, 6, 8, 12, None]),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 80),
                "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 5.0, log=True),
                "class_weight": t.suggest_categorical("class_weight", ["balanced", None]),
            }
        rng = trial_or_rng
        if model_family == "rf":
            return {
                "n_estimators": int(rng.choice([100, 150, 200, 250, 300])),
                "max_depth": [10, 15, 20, 30, None][int(rng.integers(0, 5))],
                "min_samples_leaf": int(rng.integers(1, 11)),
                "min_samples_split": int(rng.integers(2, 11)),
                "max_features": ["sqrt", "log2", 0.7][int(rng.integers(0, 3))],
                "class_weight": ["balanced", "balanced_subsample"][int(rng.integers(0, 2))],
            }
        return {
            "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
            "max_iter": int(rng.choice([100, 150, 200, 250, 300, 350, 400])),
            "max_leaf_nodes": int(rng.integers(15, 64)),
            "max_depth": [4, 6, 8, 12, None][int(rng.integers(0, 5))],
            "min_samples_leaf": int(rng.integers(10, 81)),
            "l2_regularization": float(np.exp(rng.uniform(np.log(1e-4), np.log(5.0)))),
            "class_weight": ["balanced", None][int(rng.integers(0, 2))],
        }

    def eval_params(params: Dict[str, Any], trial_id: int) -> float:
        builder = (lambda p=params: build_rf(p)) if model_family == "rf" else (lambda p=params: build_hgb(p))
        metrics = evaluate_cv(builder, df, feature_cols, n_splits=n_splits)
        top1 = metrics["top1_accuracy"] if not np.isnan(metrics["top1_accuracy"]) else metrics["f1"]
        score = 0.5 * top1 + 0.5 * metrics["f1"] - 0.1 * metrics["log_loss"]
        trials_log.append({"model_family": model_family, "trial": trial_id, "score": score, **params, **metrics})
        return score

    if has_optuna:
        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda t: eval_params(sample_params(t), t.number), n_trials=n_trials)
        best_params = dict(study.best_params)
    else:
        rng = np.random.default_rng(random_state)
        best_score = -np.inf
        best_params = {}
        for i in range(n_trials):
            p = sample_params(rng)
            s = eval_params(p, i)
            if s > best_score:
                best_score = s
                best_params = p

    return best_params, trials_log


def main(
    training_csv: str = "training_features.csv",
    data_dir: str = "Student folder",
    n_trials: int = 20,
    n_splits: int = 5,
    output_dir: str = "artifacts",
) -> None:
    df, feature_cols = load_training_table(training_csv, data_dir=data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rf_base_params = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2, "class_weight": "balanced"}
    hgb_base_params = {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20}

    results: List[Dict[str, Any]] = []
    for name, builder, params in [
        ("Baseline Random Forest", lambda: build_rf(rf_base_params), rf_base_params),
        ("Baseline HistGradientBoosting", lambda: build_hgb(hgb_base_params), hgb_base_params),
    ]:
        m = evaluate_cv(builder, df, feature_cols, n_splits=n_splits)
        results.append({"model": name, "params": json.dumps(params), **m})

    rf_best_params, rf_trials = run_bayesian_hpo("rf", df, feature_cols, n_trials=n_trials, n_splits=n_splits)
    hgb_best_params, hgb_trials = run_bayesian_hpo("hgb", df, feature_cols, n_trials=n_trials, n_splits=n_splits)

    for name, builder, params in [
        ("Tuned Random Forest", lambda p=rf_best_params: build_rf(p), rf_best_params),
        ("Tuned HistGradientBoosting", lambda p=hgb_best_params: build_hgb(p), hgb_best_params),
    ]:
        m = evaluate_cv(builder, df, feature_cols, n_splits=n_splits)
        results.append({"model": name, "params": json.dumps(params), **m})

    comp_df = pd.DataFrame(results).sort_values(
        by=["top1_accuracy", "f1", "log_loss"], ascending=[False, False, True]
    )
    comp_df.to_csv("model_optimization_comparison.csv", index=False)
    pd.DataFrame(rf_trials + hgb_trials).to_csv(out_path / "hpo_trials.csv", index=False)

    best_row = comp_df.iloc[0]
    best_name = str(best_row["model"])
    best_params = json.loads(str(best_row["params"]))
    final_model = build_rf(best_params) if "Random Forest" in best_name else build_hgb(best_params)
    final_model.fit(df[feature_cols], df["target_label"].astype(int))

    best_metrics = {
        k: float(best_row[k])
        for k in ["accuracy", "precision", "recall", "f1", "log_loss", "top1_accuracy", "training_time_seconds", "prediction_time_seconds"]
    }
    save_model(
        model=final_model,
        feature_columns=feature_cols,
        model_name=best_name,
        metrics=best_metrics,
        path=str(out_path / "best_model.joblib"),
    )

    config_out = {
        "selected_model": best_name,
        "best_params": best_params,
        "feature_columns": feature_cols,
        "cv_strategy": f"GroupKFold(n_splits={n_splits}, group='trip_id')",
        "metrics": best_metrics,
        "artifact_path": str(out_path / "best_model.joblib"),
    }
    with open("tuned_model_config.json", "w", encoding="utf-8") as fp:
        json.dump(config_out, fp, indent=2)

    print("=== Phase 2 Model Optimization Comparison ===")
    print(comp_df[["model", "top1_accuracy", "f1", "log_loss", "accuracy", "precision", "recall", "training_time_seconds"]].to_string(index=False))
    print(f"\nSaved best model '{best_name}' to {out_path / 'best_model.joblib'}")


if __name__ == "__main__":
    main()