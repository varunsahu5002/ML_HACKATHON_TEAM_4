"""Spatial road candidate generation for cleaned PathSnap GPS observations."""
from __future__ import annotations

import math
from feature_utils import records, first, coords, project_to_polyline

GPS_ID = ("observation_id", "gps_id", "point_id", "id")
TRIP_ID = ("trip_id", "trajectory_id")
ROAD_ID = ("road_id", "edge_id", "segment_id", "id")
GEOMETRY = ("geometry", "geom", "coordinates")
LON = ("longitude", "lon", "lng", "x")
LAT = ("latitude", "lat", "y")


def _spatial_index(roads, radius_m):
    """Build optional STRtree; return None when Shapely/index support is absent."""
    try:
        from shapely.geometry import LineString
        from shapely.strtree import STRtree
        geoms = []
        # Bounding boxes are expanded per-road by the maximum latitude-independent
        # search radius. Exact distance filtering follows the index query.
        for road in roads:
            line = road["_coords"]
            geoms.append(LineString(line) if len(line) >= 2 else LineString(line * 2))
        return ("strtree", STRtree(geoms), geoms, {id(g): i for i, g in enumerate(geoms)})
    except (ImportError, ValueError, TypeError):
        # A coarse lon/lat grid keeps the no-dependency path practical for the
        # provided tens-of-thousands-of-arcs networks. Exact distance checks follow.
        if not roads:
            return ("grid", 0.002, 0.002, {})
        mean_lat=sum(sum(pt[1] for pt in road["_coords"])/len(road["_coords"]) for road in roads)/len(roads)
        cell_lat=max(0.0005, max(radius_m,100.0)/110540)
        cell_lon=max(0.0005, max(radius_m,100.0)/(111320*max(0.01,math.cos(math.radians(mean_lat)))))
        grid={}
        for i,road in enumerate(roads):
            line=road["_coords"]
            x0=math.floor(min(p[0] for p in line)/cell_lon); x1=math.floor(max(p[0] for p in line)/cell_lon)
            y0=math.floor(min(p[1] for p in line)/cell_lat); y1=math.floor(max(p[1] for p in line)/cell_lat)
            for x in range(x0,x1+1):
                for y in range(y0,y1+1): grid.setdefault((x,y),[]).append(i)
        return ("grid",cell_lon,cell_lat,grid)


def build_road_network(nodes, arcs):
    """Build directed road records from PathSnap ``.nodes`` and ``.arcs`` rows.

    Node rows are ``(longitude, latitude)``; arc rows are ``(from_node, to_node)``.
    Arc row order is retained as ``road_id`` because PathSnap route files contain
    those arc row IDs. Reverse travel is represented only by a separate reverse arc.
    """
    node_rows=records(nodes); arc_rows=records(arcs)
    points=[]
    for row in node_rows:
        if isinstance(row, dict):
            points.append((float(first(row,LON)),float(first(row,LAT))))
        else:
            points.append((float(row[0]),float(row[1])))
    roads=[]
    for arc_id,row in enumerate(arc_rows):
        if isinstance(row, dict):
            src=first(row,("from_node_id","from_node","source")); dst=first(row,("to_node_id","to_node","target"))
        else:
            src,dst=row[0],row[1]
        src=int(src); dst=int(dst)
        if 0 <= src < len(points) and 0 <= dst < len(points):
            roads.append({"road_id":arc_id,"from_node_id":src,"to_node_id":dst,
                          "geometry":[points[src],points[dst]],"directed":True})
    return roads


def generate_candidates(gps_data, road_network, radius_m=100.0, max_candidates=8,
                        observation_id_col=None, trip_id_col=None,
                        road_id_col=None, geometry_col=None):
    """Return one row per GPS observation/nearby road.

    Inputs may be pandas/GeoPandas frames or iterable dictionaries. Roads need a
    stable ID and line geometry; GPS rows need lon/lat and optionally trip/point
    identifiers. Shapely STRtree is used when available; otherwise a coarse
    standard-library spatial grid is used. Both paths apply exact distance filtering.
    """
    gps = records(gps_data); roads_in = records(road_network)
    id_names = (road_id_col,) if road_id_col else ROAD_ID
    geom_names = (geometry_col,) if geometry_col else GEOMETRY
    roads=[]
    for pos, road in enumerate(roads_in):
        rid = first(road, id_names, pos)
        line = coords(first(road, geom_names))
        if len(line) < 2:
            continue
        item = dict(road); item.update(_road_id=rid, _coords=line)
        roads.append(item)

    index_data = _spatial_index(roads, radius_m)
    out=[]
    for ordinal, point in enumerate(gps):
        lon = first(point, LON); lat = first(point, LAT)
        if lon is None or lat is None:
            raise ValueError("GPS rows must include longitude/latitude (or lon/lat)")
        p=(float(lon),float(lat)); possible=range(len(roads))
        if index_data and roads:
            # Degree envelope approximates the meter radius; exact geodesic distance
            # rejects false positives. Handle Shapely 1.x geometry and 2.x indices.
            try:
                if index_data[0] == "grid":
                    _,cell_lon,cell_lat,grid=index_data
                    dx=radius_m/(111320*max(0.01,math.cos(math.radians(p[1])))); dy=radius_m/110540
                    x0=math.floor((p[0]-dx)/cell_lon); x1=math.floor((p[0]+dx)/cell_lon)
                    y0=math.floor((p[1]-dy)/cell_lat); y1=math.floor((p[1]+dy)/cell_lat)
                    possible=set()
                    for x in range(x0,x1+1):
                        for y in range(y0,y1+1): possible.update(grid.get((x,y),()))
                else:
                    _,tree,geoms,geom_indices=index_data
                    from shapely.geometry import box
                    dx=radius_m/(111320*max(0.01,math.cos(math.radians(p[1])))); dy=radius_m/110540
                    hits=tree.query(box(p[0]-dx,p[1]-dy,p[0]+dx,p[1]+dy))
                    possible=[]
                    for hit in hits:
                        if isinstance(hit, int) or getattr(hit,"dtype",None) is not None:
                            possible.append(int(hit))
                        else:
                            i=geom_indices.get(id(hit))
                            if i is not None: possible.append(i)
            except Exception:
                possible=range(len(roads))
        ranked=[]
        for ri in possible:
            road=roads[int(ri)]; dist, direction, length=project_to_polyline(p,road["_coords"])
            if dist <= radius_m:
                ranked.append((dist,road,direction,length))
        ranked.sort(key=lambda row: row[0])
        oid=first(point,(observation_id_col,) if observation_id_col else GPS_ID,ordinal)
        tid=first(point,(trip_id_col,) if trip_id_col else TRIP_ID,None)
        if not ranked:
            out.append({"trip_id":tid,"observation_id":oid,"observation_index":ordinal,
                        "candidate_road_id":None,"candidate_rank":-1,
                        "candidate_distance_m":None,"candidate_bearing_deg":None,
                        "road_length_m":None})
        for rank,(dist,road,direction,length) in enumerate(ranked[:max_candidates]):
            out.append({"trip_id":tid,"observation_id":oid,"observation_index":ordinal,
                        "candidate_road_id":road["_road_id"],"candidate_rank":rank,
                        "candidate_distance_m":dist,"candidate_bearing_deg":direction,
                        "road_length_m":length})
    return out
