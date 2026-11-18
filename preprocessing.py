"""
PathSnap GPS & Road-Network Preprocessing Module (Member 1)

Cleans, validates, orders, and enriches raw GPS trajectories and road networks
so downstream candidate-generation and feature-engineering modules (Member 2)
receive consistent, leakage-free trip data with local metric coordinates.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader import TripData

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray | float:
    """Compute great-circle distance in meters using the Haversine formula."""
    lon1_r = np.radians(lon1)
    lat1_r = np.radians(lat1)
    lon2_r = np.radians(lon2)
    lat2_r = np.radians(lat2)

    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_bearing_deg(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    """Compute forward azimuth bearing in degrees [0, 360) from point 1 to point 2."""
    lon1_r, lat1_r, lon2_r, lat2_r = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2_r - lon1_r
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360.0) % 360.0


def local_xy_projection(
    lon: np.ndarray,
    lat: np.ndarray,
    origin_lon: float,
    origin_lat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project WGS84 (lon, lat) into a local tangent-plane metric (x_m, y_m) frame
    centered at (origin_lon, origin_lat). Safe across any global region.
    """
    cos_lat0 = np.cos(np.radians(origin_lat))
    x_m = np.radians(lon - origin_lon) * EARTH_RADIUS_M * cos_lat0
    y_m = np.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x_m, y_m


def _is_valid_wgs84(lon: pd.Series, lat: pd.Series) -> pd.Series:
    """Return boolean mask of finite, in-bounds, non-(0,0) GPS coordinates."""
    finite = np.isfinite(lon) & np.isfinite(lat)
    in_bounds = (lon >= -180.0) & (lon <= 180.0) & (lat >= -90.0) & (lat <= 90.0)
    not_null_island = ~((lon == 0.0) & (lat == 0.0))
    return finite & in_bounds & not_null_island


def _filter_speed_outliers(
    df: pd.DataFrame,
    max_speed_mps: float,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove extreme GPS jump outliers whose implied step speed exceeds max_speed_mps.
    Uses iterative single-point spike removal: if point i causes an impossible jump
    from i-1, point i is dropped and kinematics are re-evaluated.
    """
    if len(df) < 2 or max_speed_mps <= 0:
        return df, 0

    keep_mask = np.ones(len(df), dtype=bool)
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    ts = df["timestamp"].to_numpy()

    last_kept = 0
    dropped = 0
    for i in range(1, len(df)):
        dt = ts[i] - ts[last_kept]
        dist = float(haversine_distance(lon[last_kept], lat[last_kept], lon[i], lat[i]))
        if dt > 0 and (dist / dt) > max_speed_mps:
            keep_mask[i] = False
            dropped += 1
        else:
            last_kept = i

    return df.loc[keep_mask].copy(), dropped


def preprocess_trip(
    trip_data: TripData,
    max_speed_mps: float = 65.0,
    gap_threshold_s: float = 10.0,
    drop_duplicate_timestamps: bool = True,
    drop_stationary_duplicates: bool = False,
    filter_speed_outliers: bool = True,
    deep_copy: bool = True,
) -> TripData:
    """
    Clean and enrich a single TripData instance.

    Steps performed:
        1. Drop missing/NaN/infinite values and invalid WGS84 coordinates (including (0, 0)).
        2. Drop invalid/negative timestamps and sort chronologically (stable sort).
        3. Remove duplicate timestamps (and optionally consecutive identical coordinates).
        4. Optionally filter extreme GPS speed outliers (> max_speed_mps).
        5. Compute local metric coordinates (x_m, y_m) shared between track and nodes.
        6. Compute kinematics: dt_s, step_dist_m, speed_mps, bearing_deg, is_gap, segment_id.
        7. Validate road-network nodes/arcs and reference route indices (if present).

    Parameters
    ----------
    trip_data : TripData
        Raw trip loaded by data_loader.load_trip / load_dataset.
    max_speed_mps : float, default=65.0
        Maximum plausible vehicle speed in m/s (~234 km/h) for outlier filtering.
    gap_threshold_s : float, default=10.0
        Time delta threshold (seconds) above which a GPS dropout/gap is flagged.
    drop_duplicate_timestamps : bool, default=True
        Keep first observation when multiple rows share an identical timestamp.
    drop_stationary_duplicates : bool, default=False
        If True, drop consecutive observations with identical (lon, lat).
    filter_speed_outliers : bool, default=True
        If True, remove GPS jump outliers exceeding max_speed_mps.
    deep_copy : bool, default=True
        Avoid mutating the input TripData in place.

    Returns
    -------
    TripData
        Cleaned and enriched TripData with updated metadata['cleaning_report'].
    """
    trip = copy.deepcopy(trip_data) if deep_copy else trip_data
    track = trip.track.copy()
    raw_count = len(track)

    # 1. Missing & invalid coordinates / timestamps
    valid_coords = _is_valid_wgs84(track["lon"], track["lat"])
    valid_ts = np.isfinite(track["timestamp"]) & (track["timestamp"] >= 0.0)
    track = track.loc[valid_coords & valid_ts].copy()
    dropped_invalid = raw_count - len(track)

    # 2. Chronological ordering (stable sort preserves raw_obs_idx on equal timestamps)
    track = track.sort_values(by=["timestamp", "raw_obs_idx"], kind="mergesort").reset_index(drop=True)

    # 3. Duplicate handling
    dropped_dup_ts = 0
    if drop_duplicate_timestamps and len(track) > 0:
        before = len(track)
        track = track.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
        dropped_dup_ts = before - len(track)

    dropped_stationary = 0
    if drop_stationary_duplicates and len(track) > 1:
        before = len(track)
        same_pos = (track["lon"] == track["lon"].shift(1)) & (track["lat"] == track["lat"].shift(1))
        track = track.loc[~same_pos].reset_index(drop=True)
        dropped_stationary = before - len(track)

    # 4. Speed outlier filtering
    dropped_speed = 0
    if filter_speed_outliers and len(track) > 1:
        track, dropped_speed = _filter_speed_outliers(track, max_speed_mps=max_speed_mps)
        track = track.reset_index(drop=True)

    # Assign clean sequential observation ID
    track.insert(0, "obs_id", np.arange(len(track), dtype=np.int64))

    # 5. Shared local metric projection (origin = clean track centroid, fallback to nodes centroid)
    nodes = trip.nodes.copy()
    valid_nodes_mask = _is_valid_wgs84(nodes["lon"], nodes["lat"])
    nodes = nodes.loc[valid_nodes_mask].copy()

    if len(track) > 0:
        origin_lon = float(track["lon"].mean())
        origin_lat = float(track["lat"].mean())
    elif len(nodes) > 0:
        origin_lon = float(nodes["lon"].mean())
        origin_lat = float(nodes["lat"].mean())
    else:
        origin_lon, origin_lat = 0.0, 0.0

    if len(track) > 0:
        tx, ty = local_xy_projection(
            track["lon"].to_numpy(), track["lat"].to_numpy(), origin_lon, origin_lat
        )
        track["x_m"] = tx
        track["y_m"] = ty
    else:
        track["x_m"] = np.array([], dtype=np.float64)
        track["y_m"] = np.array([], dtype=np.float64)

    if len(nodes) > 0:
        nx, ny = local_xy_projection(
            nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), origin_lon, origin_lat
        )
        nodes["x_m"] = nx
        nodes["y_m"] = ny

    # 6. Kinematics & gap segmentation
    n_obs = len(track)
    if n_obs > 0:
        dt_s = np.zeros(n_obs, dtype=np.float64)
        step_dist_m = np.zeros(n_obs, dtype=np.float64)
        speed_mps = np.zeros(n_obs, dtype=np.float64)
        bearing_deg = np.zeros(n_obs, dtype=np.float64)

        if n_obs > 1:
            lon_arr = track["lon"].to_numpy()
            lat_arr = track["lat"].to_numpy()
            ts_arr = track["timestamp"].to_numpy()

            dt_diff = np.diff(ts_arr)
            dist_diff = haversine_distance(lon_arr[:-1], lat_arr[:-1], lon_arr[1:], lat_arr[1:])
            spd_diff = np.where(dt_diff > 0.0, dist_diff / dt_diff, 0.0)
            brg_diff = compute_bearing_deg(lon_arr[:-1], lat_arr[:-1], lon_arr[1:], lat_arr[1:])

            dt_s[1:] = dt_diff
            step_dist_m[1:] = dist_diff
            speed_mps[1:] = spd_diff
            bearing_deg[1:] = brg_diff
            bearing_deg[0] = brg_diff[0]

        is_gap = dt_s > gap_threshold_s
        segment_id = np.cumsum(is_gap.astype(np.int64))

        track["dt_s"] = dt_s
        track["step_dist_m"] = step_dist_m
        track["speed_mps"] = speed_mps
        track["bearing_deg"] = bearing_deg
        track["is_gap"] = is_gap
        track["segment_id"] = segment_id
    else:
        for col in ["dt_s", "step_dist_m", "speed_mps", "bearing_deg"]:
            track[col] = np.array([], dtype=np.float64)
        track["is_gap"] = np.array([], dtype=bool)
        track["segment_id"] = np.array([], dtype=np.int64)

    # 7. Validate arcs & route
    arcs = trip.arcs.copy()
    valid_node_ids = set(nodes["node_id"].tolist())
    arcs = arcs.dropna(subset=["from_node_id", "to_node_id"]).copy()
    arcs["from_node_id"] = arcs["from_node_id"].astype(np.int64)
    arcs["to_node_id"] = arcs["to_node_id"].astype(np.int64)
    arcs = arcs.loc[
        arcs["from_node_id"].isin(valid_node_ids) & arcs["to_node_id"].isin(valid_node_ids)
    ].copy()
    arcs["is_self_loop"] = arcs["from_node_id"] == arcs["to_node_id"]

    route = trip.route
    if route is not None:
        route = route.dropna(subset=["arc_id"]).copy()
        route["arc_id"] = route["arc_id"].astype(np.int64)
        valid_arc_ids = set(arcs["arc_id"].tolist())
        route = route.loc[route["arc_id"].isin(valid_arc_ids)].reset_index(drop=True)

    trip.track = track
    trip.nodes = nodes
    trip.arcs = arcs
    trip.route = route

    trip.metadata["origin_lon"] = origin_lon
    trip.metadata["origin_lat"] = origin_lat
    trip.metadata["cleaning_report"] = {
        "raw_obs": int(raw_count),
        "clean_obs": int(len(track)),
        "dropped_invalid_coords_or_ts": int(dropped_invalid),
        "dropped_duplicate_timestamps": int(dropped_dup_ts),
        "dropped_stationary_duplicates": int(dropped_stationary),
        "dropped_speed_outliers": int(dropped_speed),
        "num_gaps": int(track["is_gap"].sum()) if len(track) > 0 else 0,
        "num_segments": int(track["segment_id"].nunique()) if len(track) > 0 else 0,
        "total_distance_m": float(track["step_dist_m"].sum()) if len(track) > 0 else 0.0,
        "duration_s": float(track["timestamp"].iloc[-1] - track["timestamp"].iloc[0])
        if len(track) > 1
        else 0.0,
    }
    return trip


def preprocess_data(
    raw_data: Dict[str, TripData],
    max_speed_mps: float = 65.0,
    gap_threshold_s: float = 10.0,
    drop_duplicate_timestamps: bool = True,
    drop_stationary_duplicates: bool = False,
    filter_speed_outliers: bool = True,
) -> Dict[str, TripData]:
    """
    Preprocess an entire dictionary of TripData objects.

    Parameters
    ----------
    raw_data : dict[str, TripData]
        Output of data_loader.load_dataset or split_train_val.

    Returns
    -------
    dict[str, TripData]
        Dictionary mapping trip_id to cleaned TripData.
    """
    clean_data: Dict[str, TripData] = {}
    for tid, trip in raw_data.items():
        clean_data[tid] = preprocess_trip(
            trip,
            max_speed_mps=max_speed_mps,
            gap_threshold_s=gap_threshold_s,
            drop_duplicate_timestamps=drop_duplicate_timestamps,
            drop_stationary_duplicates=drop_stationary_duplicates,
            filter_speed_outliers=filter_speed_outliers,
            deep_copy=True,
        )
    return clean_data


def to_observation_dataframe(clean_data: Dict[str, TripData]) -> pd.DataFrame:
    """
    Flatten cleaned TripData objects into a single observation-level DataFrame
    ready for Member 2's candidate generation and feature engineering.
    """
    frames = []
    for tid, trip in sorted(clean_data.items()):
        df = trip.track.copy()
        df.insert(0, "trip_id", tid)
        df.insert(1, "partition", trip.partition)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)