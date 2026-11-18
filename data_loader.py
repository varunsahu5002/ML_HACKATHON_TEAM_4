"""
PathSnap Data Loader Module (Member 1)

Provides leakage-safe loading utilities for the PathSnap GPS map-matching dataset.
Each trip directory contains four headerless, whitespace-separated files:
    - <trip_id>.track : lon, lat, timestamp (seconds)
    - <trip_id>.nodes : lon, lat (0-based row index = node_id)
    - <trip_id>.arcs  : from_node_id, to_node_id (0-based row index = arc_id)
    - <trip_id>.route : arc_id (chronological reference route; optional for inference)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRACK_COLUMNS = ["lon", "lat", "timestamp"]
NODE_COLUMNS = ["lon", "lat"]
ARC_COLUMNS = ["from_node_id", "to_node_id"]
ROUTE_COLUMNS = ["arc_id"]


@dataclass
class TripData:
    """Container for a single PathSnap trip and its associated road network."""

    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: Optional[pd.DataFrame] = None
    metadata: Dict[str, object] = field(default_factory=dict)


def _read_whitespace_table(
    file_path: Path,
    columns: List[str],
    dtypes: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Read a headerless whitespace-separated file safely into a DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns,
        engine="c",
        on_bad_lines="warn",
    )
    if dtypes:
        for col, dtype in dtypes.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df


def load_trip(
    trip_dir: Union[str, Path],
    trip_id: Optional[str] = None,
    partition: str = "unknown",
    load_route: bool = True,
) -> TripData:
    """
    Load a single trip directory into a TripData object.

    Parameters
    ----------
    trip_dir : str or Path
        Path to the trip folder containing <trip_id>.track, .nodes, .arcs, and optionally .route.
    trip_id : str, optional
        Trip identifier. Defaults to the folder name.
    partition : str, default="unknown"
        Dataset partition label (e.g., "train", "student_test", "inference").
    load_route : bool, default=True
        Whether to load the .route reference file if present. Set False for
        unlabeled inference or to guarantee ground-truth isolation.

    Returns
    -------
    TripData
    """
    trip_path = Path(trip_dir)
    if not trip_path.is_dir():
        raise NotADirectoryError(f"Trip directory does not exist: {trip_path}")

    tid = trip_id or trip_path.name

    track_df = _read_whitespace_table(
        trip_path / f"{tid}.track",
        TRACK_COLUMNS,
        dtypes={"lon": "float64", "lat": "float64", "timestamp": "float64"},
    )
    track_df.insert(0, "raw_obs_idx", np.arange(len(track_df), dtype=np.int64))

    nodes_df = _read_whitespace_table(
        trip_path / f"{tid}.nodes",
        NODE_COLUMNS,
        dtypes={"lon": "float64", "lat": "float64"},
    )
    nodes_df.insert(0, "node_id", np.arange(len(nodes_df), dtype=np.int64))

    arcs_df = _read_whitespace_table(
        trip_path / f"{tid}.arcs",
        ARC_COLUMNS,
    )
    for col in ARC_COLUMNS:
        arcs_df[col] = pd.to_numeric(arcs_df[col], errors="coerce")
    arcs_df.insert(0, "arc_id", np.arange(len(arcs_df), dtype=np.int64))

    route_df: Optional[pd.DataFrame] = None
    route_file = trip_path / f"{tid}.route"
    if load_route and route_file.exists():
        route_df = _read_whitespace_table(route_file, ROUTE_COLUMNS)
        route_df["arc_id"] = pd.to_numeric(route_df["arc_id"], errors="coerce")
        route_df.insert(0, "route_seq_idx", np.arange(len(route_df), dtype=np.int64))

    metadata: Dict[str, object] = {
        "trip_id": tid,
        "partition": partition,
        "trip_dir": str(trip_path),
        "raw_track_count": int(len(track_df)),
        "raw_node_count": int(len(nodes_df)),
        "raw_arc_count": int(len(arcs_df)),
        "raw_route_count": int(len(route_df)) if route_df is not None else 0,
        "has_route": route_df is not None,
    }

    return TripData(
        trip_id=tid,
        partition=partition,
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=route_df,
        metadata=metadata,
    )


def load_dataset(
    data_dir: Union[str, Path],
    partition: str = "train",
    trip_ids: Optional[Iterable[str]] = None,
    load_route: Optional[bool] = None,
) -> Dict[str, TripData]:
    """
    Load all trips (or a selected subset) from a PathSnap dataset partition.

    Parameters
    ----------
    data_dir : str or Path
        Root dataset directory (e.g., "Student folder") or direct partition directory.
    partition : str, default="train"
        Subdirectory name under data_dir ("train" or "student_test").
    trip_ids : Iterable[str], optional
        Specific trip IDs to load. If None, loads all sorted trip subdirectories.
    load_route : bool, optional
        Whether to load .route files. Defaults to True for "train" and "student_test",
        and False for any evaluator/test/inference partition to prevent ground-truth leakage.

    Returns
    -------
    dict[str, TripData]
        Mapping from trip_id to TripData, sorted by trip_id.
    """
    root = Path(data_dir)
    partition_dir = root / partition if (root / partition).is_dir() else root

    if not partition_dir.is_dir():
        raise NotADirectoryError(f"Dataset partition directory not found: {partition_dir}")

    is_evaluator_partition = "evaluator" in partition.lower()
    if load_route is None:
        load_route = not is_evaluator_partition
    elif load_route and is_evaluator_partition:
        warnings.warn(
            "Loading .route ground truth on an evaluator partition is disabled to prevent data leakage.",
            UserWarning,
            stacklevel=2,
        )
        load_route = False

    available_dirs = sorted([p for p in partition_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if trip_ids is not None:
        requested = set(trip_ids)
        selected_dirs = [p for p in available_dirs if p.name in requested]
        missing = requested - {p.name for p in selected_dirs}
        if missing:
            raise ValueError(f"Requested trip_ids not found in {partition_dir}: {sorted(missing)}")
    else:
        selected_dirs = available_dirs

    dataset: Dict[str, TripData] = {}
    for tdir in selected_dirs:
        dataset[tdir.name] = load_trip(
            trip_dir=tdir,
            trip_id=tdir.name,
            partition=partition,
            load_route=load_route,
        )

    logger.info("Loaded %d trips from partition '%s' (%s).", len(dataset), partition, partition_dir)
    return dataset


def split_train_val(
    raw_data: Dict[str, TripData],
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, TripData], Dict[str, TripData]]:
    """
    Split loaded trips into train and validation subsets at the TRIP level.

    Splitting by trip_id (rather than by individual GPS observation) guarantees
    zero observation leakage across train and validation sets.

    Parameters
    ----------
    raw_data : dict[str, TripData]
        Dictionary of trips (typically the 80-trip "train" partition).
    val_ratio : float, default=0.2
        Fraction of trips to allocate to validation (0 < val_ratio < 1).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    (train_trips, val_trips) : tuple of dict[str, TripData]
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    trip_ids = sorted(raw_data.keys())
    if len(trip_ids) < 2:
        raise ValueError("Need at least 2 trips to perform a train/validation split.")

    rng = np.random.default_rng(random_state)
    shuffled = list(trip_ids)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1

    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    train_trips = {tid: raw_data[tid] for tid in train_ids}
    val_trips = {tid: raw_data[tid] for tid in val_ids}
    return train_trips, val_trips