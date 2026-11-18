"""
PathSnap Dataset Exploration & Reporting Script (Member 1)

Generates dataset inventory tables, partition-level summary statistics,
road-network connectivity diagnostics, sample cleaned CSV output, and
visualizations for the `train` (80 trips) and `student_test` (10 trips)
partitions. Never accesses evaluator-test data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from data_loader import TripData, load_dataset
from preprocessing import preprocess_data, to_observation_dataframe


def analyze_road_network(trip: TripData) -> Dict[str, int]:
    """Compute node/arc counts, self-loops, isolated nodes, and connected components."""
    n_nodes = len(trip.nodes)
    n_arcs = len(trip.arcs)
    if n_nodes == 0:
        return {
            "num_nodes": 0,
            "num_arcs": 0,
            "self_loops": 0,
            "isolated_nodes": 0,
            "connected_components": 0,
            "largest_component_nodes": 0,
        }

    u = trip.arcs["from_node_id"].to_numpy()
    v = trip.arcs["to_node_id"].to_numpy()
    self_loops = int((u == v).sum())

    # Undirected connectivity for component & isolated-node analysis
    deg = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(deg, u, 1)
    np.add.at(deg, v, 1)
    isolated = int((deg == 0).sum())

    data = np.ones(len(u), dtype=np.int8)
    adj = csr_matrix((data, (u, v)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    _, counts = np.unique(labels, return_counts=True)
    largest_comp = int(counts.max()) if len(counts) > 0 else 0

    return {
        "num_nodes": int(n_nodes),
        "num_arcs": int(n_arcs),
        "self_loops": self_loops,
        "isolated_nodes": isolated,
        "connected_components": int(n_comp),
        "largest_component_nodes": largest_comp,
    }


def analyze_route_continuity(trip: TripData) -> Tuple[int, int]:
    """Return (route_length, num_continuity_breaks) for a trip's reference route."""
    if trip.route is None or len(trip.route) == 0:
        return 0, 0
    arc_ids = trip.route["arc_id"].to_numpy()
    if len(arc_ids) < 2:
        return len(arc_ids), 0

    arc_lookup = trip.arcs.set_index("arc_id")[["from_node_id", "to_node_id"]]
    u_seq = arc_lookup.loc[arc_ids, "from_node_id"].to_numpy()
    v_seq = arc_lookup.loc[arc_ids, "to_node_id"].to_numpy()
    breaks = int((v_seq[:-1] != u_seq[1:]).sum())
    return int(len(arc_ids)), breaks


def build_inventory_table(clean_data: Dict[str, TripData]) -> pd.DataFrame:
    """Build per-trip dataset inventory and quality summary DataFrame."""
    rows: List[Dict[str, object]] = []
    for tid, trip in sorted(clean_data.items()):
        rep = trip.metadata.get("cleaning_report", {})
        net = analyze_road_network(trip)
        route_len, route_breaks = analyze_route_continuity(trip)
        trk = trip.track

        rows.append(
            {
                "trip_id": tid,
                "partition": trip.partition,
                "raw_obs": rep.get("raw_obs", len(trk)),
                "clean_obs": rep.get("clean_obs", len(trk)),
                "dropped_invalid": rep.get("dropped_invalid_coords_or_ts", 0),
                "dropped_dup_ts": rep.get("dropped_duplicate_timestamps", 0),
                "dropped_speed_outliers": rep.get("dropped_speed_outliers", 0),
                "num_gaps_gt_10s": rep.get("num_gaps", 0),
                "num_segments": rep.get("num_segments", 1),
                "duration_s": round(float(rep.get("duration_s", 0.0)), 2),
                "total_distance_km": round(float(rep.get("total_distance_m", 0.0)) / 1000.0, 3),
                "median_speed_mps": round(float(trk["speed_mps"].median()), 2) if len(trk) > 0 else 0.0,
                "max_dt_s": round(float(trk["dt_s"].max()), 2) if len(trk) > 0 else 0.0,
                "lon_min": round(float(trk["lon"].min()), 5) if len(trk) > 0 else np.nan,
                "lon_max": round(float(trk["lon"].max()), 5) if len(trk) > 0 else np.nan,
                "lat_min": round(float(trk["lat"].min()), 5) if len(trk) > 0 else np.nan,
                "lat_max": round(float(trk["lat"].max()), 5) if len(trk) > 0 else np.nan,
                **net,
                "route_arc_count": route_len,
                "route_continuity_breaks": route_breaks,
            }
        )
    return pd.DataFrame(rows)


def generate_eda_plots(
    inventory_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    sample_train: TripData,
    sample_test: TripData,
    output_dir: Path,
) -> None:
    """Generate summary EDA plots and sample trajectory/road-network overlays."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Multi-panel EDA summary
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("PathSnap Dataset Exploration (Train vs. Student Test)", fontsize=14, fontweight="bold")

    # Panel A: Observations per trip
    for part, color in [("train", "#2563eb"), ("student_test", "#ea580c")]:
        sub = inventory_df[inventory_df["partition"] == part]
        axes[0, 0].hist(sub["clean_obs"], bins=20, alpha=0.65, label=f"{part} (n={len(sub)})", color=color)
    axes[0, 0].set_title("Clean GPS Observations per Trip")
    axes[0, 0].set_xlabel("Observations")
    axes[0, 0].set_ylabel("Trip Count")
    axes[0, 0].legend()

    # Panel B: Sampling interval distribution (log-binned dt > 0)
    dt_pos = obs_df.loc[obs_df["dt_s"] > 0, "dt_s"]
    axes[0, 1].hist(np.clip(dt_pos, 1, 60), bins=30, color="#0d9488", edgecolor="white")
    axes[0, 1].set_title("Sampling Interval Distribution (clipped at 60s)")
    axes[0, 1].set_xlabel("dt (seconds)")
    axes[0, 1].set_ylabel("Observation Steps")

    # Panel C: Speed distribution
    spd = obs_df.loc[obs_df["speed_mps"] > 0, "speed_mps"]
    axes[1, 0].hist(spd, bins=40, color="#7c3aed", edgecolor="white")
    axes[1, 0].set_title("Cleaned Step Speed Distribution")
    axes[1, 0].set_xlabel("Speed (m/s)")
    axes[1, 0].set_ylabel("Steps")

    # Panel D: Road-network size (nodes vs arcs)
    for part, color in [("train", "#2563eb"), ("student_test", "#ea580c")]:
        sub = inventory_df[inventory_df["partition"] == part]
        axes[1, 1].scatter(sub["num_nodes"], sub["num_arcs"], alpha=0.75, label=part, color=color)
    axes[1, 1].set_title("Road Network Size: Nodes vs. Arcs")
    axes[1, 1].set_xlabel("Number of Nodes")
    axes[1, 1].set_ylabel("Number of Arcs")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "eda_summary.png", dpi=150)
    plt.close(fig)

    # 2. Sample trajectory + road network overlay
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, trip, title in [
        (axes[0], sample_train, f"Train Trip {sample_train.trip_id}"),
        (axes[1], sample_test, f"Student Test Trip {sample_test.trip_id}"),
    ]:
        # Plot road nodes (subsampled if huge) and GPS track in local meters
        nodes = trip.nodes
        if len(nodes) > 15000:
            nodes_plot = nodes.sample(15000, random_state=42)
        else:
            nodes_plot = nodes
        ax.scatter(nodes_plot["x_m"], nodes_plot["y_m"], s=1, color="#94a3b8", alpha=0.35, label="Road nodes")
        ax.plot(trip.track["x_m"], trip.track["y_m"], color="#2563eb", linewidth=1.4, label="Cleaned GPS track")
        ax.set_title(title)
        ax.set_xlabel("Local X (m)")
        ax.set_ylabel("Local Y (m)")
        ax.axis("equal")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_dir / "sample_trajectories.png", dpi=150)
    plt.close(fig)


def run_exploration(data_dir: str = "Student folder", output_dir: str = "outputs") -> Dict[str, object]:
    """Run full exploration and preprocessing pipeline on train and student_test."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_train = load_dataset(data_dir, partition="train", load_route=True)
    raw_test = load_dataset(data_dir, partition="student_test", load_route=True)

    clean_train = preprocess_data(raw_train)
    clean_test = preprocess_data(raw_test)

    combined = {**clean_train, **clean_test}
    inventory_df = build_inventory_table(combined)
    inventory_df.to_csv(out_path / "dataset_inventory.csv", index=False)

    obs_df = to_observation_dataframe(combined)

    # Save cleaned example CSV for Member 2
    sample_tid = sorted(clean_train.keys())[0]
    clean_train[sample_tid].track.to_csv(out_path / f"cleaned_sample_train_{sample_tid}.csv", index=False)

    # Partition summary
    summary = {}
    for part in ["train", "student_test"]:
        sub_inv = inventory_df[inventory_df["partition"] == part]
        sub_obs = obs_df[obs_df["partition"] == part]
        dt_pos = sub_obs.loc[sub_obs["dt_s"] > 0, "dt_s"]
        summary[part] = {
            "num_trips": int(len(sub_inv)),
            "total_raw_obs": int(sub_inv["raw_obs"].sum()),
            "total_clean_obs": int(sub_inv["clean_obs"].sum()),
            "total_dropped_speed_outliers": int(sub_inv["dropped_speed_outliers"].sum()),
            "trips_with_gaps_gt_10s": int((sub_inv["num_gaps_gt_10s"] > 0).sum()),
            "frac_1s_sampling": round(float((dt_pos == 1.0).mean()), 4) if len(dt_pos) > 0 else 0.0,
            "median_nodes_per_trip": float(sub_inv["num_nodes"].median()),
            "median_arcs_per_trip": float(sub_inv["num_arcs"].median()),
            "trips_with_route_breaks": int((sub_inv["route_continuity_breaks"] > 0).sum()),
        }

    with open(out_path / "partition_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    sample_test_tid = sorted(clean_test.keys())[0]
    generate_eda_plots(
        inventory_df=inventory_df,
        obs_df=obs_df,
        sample_train=clean_train[sample_tid],
        sample_test=clean_test[sample_test_tid],
        output_dir=out_path,
    )
    return summary


if __name__ == "__main__":
    stats = run_exploration()
    print(json.dumps(stats, indent=2))