"""
PathSnap — Interactive Folium Map Generator (Deliverables Section 2.2 & 6)

Generates an interactive Folium HTML map visualizing:
  1. Local road-network geometry (filtered to trajectory bounding box)
  2. Raw GPS observations
  3. Reference route (.route), where available
  4. Model-predicted candidate route (when artifacts/best_model.joblib is present)

Includes distinguishable colors, layer controls, automatic bounding-box zoom,
and a floating map legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import folium
import pandas as pd

from data_loader import load_trip
from preprocessing import preprocess_trip


def _add_legend(m: folium.Map, has_pred: bool) -> None:
    pred_row = (
        '<div><span style="background:#dc2626;width:14px;height:4px;display:inline-block;margin-right:6px;"></span>Predicted Route</div>'
        if has_pred
        else ""
    )
    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 9999;
        background: rgba(255, 255, 255, 0.94); padding: 10px 14px;
        border: 1px solid #cbd5e1; border-radius: 6px;
        font-family: Arial, sans-serif; font-size: 12px; line-height: 1.6;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    ">
      <div style="font-weight:700;margin-bottom:4px;">PathSnap Map Legend</div>
      <div><span style="background:#94a3b8;width:14px;height:3px;display:inline-block;margin-right:6px;"></span>Road Network</div>
      <div><span style="background:#2563eb;width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:8px;"></span>Raw GPS Points</div>
      <div><span style="background:#16a34a;width:14px;height:4px;display:inline-block;margin-right:6px;"></span>Reference Route</div>
      {pred_row}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def _predict_trip_arcs(trip, model_path: str = "artifacts/best_model.joblib") -> Optional[List[int]]:
    """Predict top-1 candidate road_id per observation if model artifact exists."""
    mpath = Path(model_path)
    if not mpath.exists():
        return None

    from candidate_generation import build_road_network, generate_candidates
    from feature_engineering import extract_features
    from model_utils import load_model

    artifact = load_model(str(mpath))
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    road_net = build_road_network(trip.nodes, trip.arcs)
    obs_df = trip.track.copy()
    obs_df["trip_id"] = trip.trip_id
    obs_df["observation_id"] = obs_df["obs_id"]

    cands = generate_candidates(
        obs_df,
        road_net,
        radius_m=100.0,
        max_candidates=8,
        observation_id_col="observation_id",
        trip_id_col="trip_id",
    )
    feat_df = extract_features(obs_df, cands, road_net, include_labels=False)
    if isinstance(feat_df, list):
        feat_df = pd.DataFrame(feat_df)
    if len(feat_df) == 0:
        return None

    probs = model.predict_proba(feat_df[feature_cols])[:, 1]
    feat_df = feat_df.copy()
    feat_df["prob"] = probs
    top_idx = feat_df.groupby("observation_id")["prob"].idxmax()
    pred_arcs = feat_df.loc[top_idx].sort_values("observation_id")["candidate_road_id"].astype(int).tolist()
    return pred_arcs


def build_trip_map(
    trip_dir: str,
    partition: str = "student_test",
    output_html: str = "outputs/folium_map.html",
    model_path: str = "artifacts/best_model.joblib",
) -> str:
    raw_trip = load_trip(trip_dir, partition=partition, load_route=True)
    clean_trip = preprocess_trip(raw_trip)

    trk_raw = raw_trip.track
    center_lat = float(trk_raw["lat"].mean())
    center_lon = float(trk_raw["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", attr="Esri")

    # 1. Road-network geometry within padded trajectory bounding box
    pad = 0.004
    lat_min, lat_max = float(trk_raw["lat"].min() - pad), float(trk_raw["lat"].max() + pad)
    lon_min, lon_max = float(trk_raw["lon"].min() - pad), float(trk_raw["lon"].max() + pad)

    nodes = raw_trip.nodes.set_index("node_id")
    node_lon: Dict[int, float] = nodes["lon"].to_dict()
    node_lat: Dict[int, float] = nodes["lat"].to_dict()

    fg_net = folium.FeatureGroup(name="Road Network", show=True)
    for _, row in raw_trip.arcs.iterrows():
        u, v = int(row["from_node_id"]), int(row["to_node_id"])
        if u not in node_lon or v not in node_lon:
            continue
        lon1, lat1 = node_lon[u], node_lat[u]
        lon2, lat2 = node_lon[v], node_lat[v]
        if (lat_min <= lat1 <= lat_max and lon_min <= lon1 <= lon_max) or (
            lat_min <= lat2 <= lat_max and lon_min <= lon2 <= lon_max
        ):
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="#94a3b8",
                weight=1.5,
                opacity=0.65,
            ).add_to(fg_net)
    fg_net.add_to(m)

    # 2. Raw GPS observations
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    step = max(1, len(trk_raw) // 500)  # keep HTML responsive for long trips
    for _, r in trk_raw.iloc[::step].iterrows():
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=3,
            color="#2563eb",
            fill=True,
            fill_opacity=0.85,
            tooltip=f"t={r['timestamp']:.0f}s",
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # 3. Reference route (.route)
    arcs_lookup = raw_trip.arcs.set_index("arc_id")[["from_node_id", "to_node_id"]].to_dict("index")
    if raw_trip.route is not None and len(raw_trip.route) > 0:
        fg_ref = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        for aid in raw_trip.route["arc_id"].astype(int).tolist():
            if aid in arcs_lookup:
                u, v = int(arcs_lookup[aid]["from_node_id"]), int(arcs_lookup[aid]["to_node_id"])
                folium.PolyLine(
                    locations=[[node_lat[u], node_lon[u]], [node_lat[v], node_lon[v]]],
                    color="#16a34a",
                    weight=5,
                    opacity=0.85,
                ).add_to(fg_ref)
        fg_ref.add_to(m)

    # 4. Predicted route (top-1 candidate per observation)
    pred_arcs = _predict_trip_arcs(clean_trip, model_path=model_path)
    if pred_arcs:
        fg_pred = folium.FeatureGroup(name="Predicted Route (Classifier Top-1)", show=True)
        for aid in dict.fromkeys(pred_arcs):  # preserve order, deduplicate consecutive repeats
            if aid in arcs_lookup:
                u, v = int(arcs_lookup[aid]["from_node_id"]), int(arcs_lookup[aid]["to_node_id"])
                folium.PolyLine(
                    locations=[[node_lat[u], node_lon[u]], [node_lat[v], node_lon[v]]],
                    color="#dc2626",
                    weight=3.5,
                    dash_array="6,4",
                    opacity=0.9,
                ).add_to(fg_pred)
        fg_pred.add_to(m)

    m.fit_bounds([[float(trk_raw["lat"].min()), float(trk_raw["lon"].min())], [float(trk_raw["lat"].max()), float(trk_raw["lon"].max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    _add_legend(m, has_pred=bool(pred_arcs))

    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"Saved Folium map to {out}")
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PathSnap Folium visualization")
    parser.add_argument("--trip-dir", default="Student folder/student_test/00000000")
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--output-html", default="outputs/folium_map_00000000.html")
    parser.add_argument("--model-path", default="artifacts/best_model.joblib")
    args = parser.parse_args()
    build_trip_map(
        trip_dir=args.trip_dir,
        partition=args.partition,
        output_html=args.output_html,
        model_path=args.model_path,
    )
