"""Per-GPS-observation, per-road-candidate feature extraction for PathSnap."""
from __future__ import annotations

from feature_utils import records, first, coords, project_to_polyline, haversine_m, bearing_deg, circular_difference
from candidate_generation import LON, LAT, GPS_ID, TRIP_ID, ROAD_ID, GEOMETRY

TIME = ("timestamp", "time", "seconds", "t")
TRUTH = ("ground_truth_road_id", "matched_road_id", "true_road_id", "road_id")
ROAD_TYPE = ("highway", "road_type", "type", "class")
DIRECTION = ("oneway", "one_way", "direction", "directed")


def extract_features(gps_data, candidate_data, road_network, *, observation_id_col=None,
                     trip_id_col=None, road_id_col=None, geometry_col=None,
                     include_labels=True):
    """Create an ML-ready DataFrame (or list[dict] if pandas is unavailable).

    ``candidate_data`` is the output of :func:`generate_candidates`. Ground truth
    becomes ``target_label`` only when an observation explicitly has a road ID;
    it is never copied into a feature. The supplied PathSnap .route sequence is
    not point-aligned and therefore cannot directly provide these labels.
    """
    gps=records(gps_data); cand=records(candidate_data); network=records(road_network)
    rid_names=(road_id_col,) if road_id_col else ROAD_ID
    geom_names=(geometry_col,) if geometry_col else GEOMETRY
    road_by_id={}
    for i,r in enumerate(network):
        rid=first(r,rid_names,i); line=coords(first(r,geom_names))
        if len(line)>=2: road_by_id[rid]=(r,line)
    oid_names=(observation_id_col,) if observation_id_col else GPS_ID
    tid_names=(trip_id_col,) if trip_id_col else TRIP_ID
    point_by_id={}
    point_by_key={}
    for i,p in enumerate(gps):
        oid=first(p,oid_names,i); tid=first(p,tid_names,None)
        point_by_id[oid]=(i,p); point_by_key[(tid,oid)]=(i,p)
    ordered=sorted(point_by_key.items(),key=lambda pair:pair[1][0])
    previous_by_oid={}; last_by_trip={}
    # Previous observation lookup is trip-local and uses original input order.
    for (row_tid,oid),(idx,p) in ordered:
        tid=first(p,tid_names,None); lon=first(p,LON); lat=first(p,LAT)
        if lon is not None and lat is not None:
            previous_by_oid[(tid,oid)]=last_by_trip.get(tid)
            last_by_trip[tid]=((float(lon),float(lat)),p)
    out=[]
    for c in cand:
        oid=c.get("observation_id"); tid=c.get("trip_id")
        item=point_by_key.get((tid,oid),point_by_id.get(oid))
        if item is None: continue
        idx,p=item; tid=c.get("trip_id",first(p,tid_names,None))
        lon=first(p,LON); lat=first(p,LAT)
        if lon is None or lat is None: continue
        point=(float(lon),float(lat)); rid=c.get("candidate_road_id")
        road_entry=road_by_id.get(rid)
        if road_entry:
            road,line=road_entry; dist,bearing,length=project_to_polyline(point,line)
        else:
            road={}; dist=c.get("candidate_distance_m"); bearing=c.get("candidate_bearing_deg"); length=c.get("road_length_m")
        item_out={"trip_id":tid,"observation_id":oid,"observation_index":idx,
                  "candidate_road_id":rid,
                  "distance_to_road_m":float(dist) if dist is not None else None,
                  "perpendicular_distance_m":float(dist) if dist is not None else None,
                  "road_length_m":float(length) if length is not None else None}
        # Movement features use only the immediately preceding point in this trip.
        prev=previous_by_oid.get((tid,oid))
        if prev:
            ppoint,prow=prev; movement=haversine_m(ppoint,point)
            gps_bearing=bearing_deg(ppoint,point) if movement>0 else None
            item_out["gps_movement_distance_m"]=movement
            item_out["gps_movement_bearing_deg"]=gps_bearing
            ts=first(p,TIME); pts=first(prow,TIME)
            try:
                dt=float(ts)-float(pts)
                item_out["gps_speed_mps"]=movement/dt if dt>0 else None
            except (TypeError,ValueError):
                pass
            if bearing is not None and gps_bearing is not None:
                diff=circular_difference(float(gps_bearing),float(bearing))
                direction=first(road,DIRECTION)
                # An explicit bidirectional flag permits either orientation. The
                # dataset's directed arcs are one-way edges, so use raw difference.
                bidirectional=isinstance(direction,str) and direction.lower() in ("both","bidirectional","two-way","0","false")
                if direction is False or direction == 0: bidirectional=True
                if bidirectional: diff=min(diff,180-diff)
                item_out["bearing_difference_deg"]=diff
        rtype=first(road,ROAD_TYPE)
        if rtype is not None: item_out["road_type"]=str(rtype)
        # Do not infer labels from the route sequence: it has no GPS alignment.
        if include_labels:
            true_id=first(p,TRUTH)
            if true_id is not None: item_out["target_label"]=int(rid==true_id)
        out.append(item_out)
    try:
        import pandas as pd
        return pd.DataFrame(out)
    except ImportError:
        return out


def get_feature_columns(feature_dataframe):
    """Return model input columns, excluding IDs and the optional target."""
    cols=list(feature_dataframe.columns) if hasattr(feature_dataframe,"columns") else list(feature_dataframe[0]) if feature_dataframe else []
    excluded={"trip_id","observation_id","observation_index","candidate_road_id","target_label"}
    return [c for c in cols if c not in excluded]


def build_feature_preprocessor(feature_dataframe, *, scale_numeric=True):
    """Return an sklearn transformer for leak-safe numeric/categorical inputs.

    Fit this transformer on training feature rows only, then reuse its ``transform``
    method for validation/student/evaluator rows. IDs and ``target_label`` are
    excluded. Numeric missing values are median-imputed; categorical values
    (for example ``road_type`` when a dataset supplies it) are imputed and
    one-hot encoded with unseen categories ignored. Scaling is optional.
    scikit-learn is imported lazily so feature extraction remains independently
    importable without the ML dependency.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    columns=get_feature_columns(feature_dataframe)
    if hasattr(feature_dataframe,"select_dtypes"):
        numeric=[c for c in columns if c in feature_dataframe.select_dtypes(include="number").columns]
    else:
        rows=records(feature_dataframe)
        numeric=[c for c in columns if any(isinstance(r.get(c),(int,float)) for r in rows)]
    categorical=[c for c in columns if c not in numeric]
    numeric_steps=[("impute",SimpleImputer(strategy="median"))]
    if scale_numeric: numeric_steps.append(("scale",StandardScaler()))
    return ColumnTransformer([
        ("numeric",Pipeline(numeric_steps),numeric),
        ("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                                  ("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical),
    ],remainder="drop")
