"""Shared input and geometry helpers for PathSnap feature generation.

Coordinates are interpreted as WGS84 longitude/latitude unless stated otherwise.
Shapely and pandas are optional; plain Python records are supported.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def records(data):
    """Convert a DataFrame, GeoDataFrame, or iterable of mappings to records."""
    if hasattr(data, "to_dict"):
        try:
            return data.to_dict(orient="records")
        except TypeError:
            pass
    return [dict(row) if isinstance(row, Mapping) else row for row in data]


def first(row, names, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def coords(geometry):
    """Return a polyline as [(lon, lat), ...] from common geometry forms."""
    if geometry is None:
        return []
    if hasattr(geometry, "geom_type"):
        if geometry.geom_type == "LineString":
            return [(float(x), float(y)) for x, y, *rest in geometry.coords]
        if geometry.geom_type == "MultiLineString":
            return [pt for line in geometry.geoms for pt in coords(line)]
        if geometry.geom_type == "Point":
            return [(float(geometry.x), float(geometry.y))]
    if isinstance(geometry, Mapping):
        typ = geometry.get("type")
        values = geometry.get("coordinates", [])
        if typ == "LineString":
            return [(float(p[0]), float(p[1])) for p in values]
        if typ == "MultiLineString":
            return [(float(p[0]), float(p[1])) for line in values for p in line]
        if typ == "Point":
            return [(float(values[0]), float(values[1]))]
    if isinstance(geometry, (list, tuple)):
        if len(geometry) >= 2 and all(isinstance(v, (int, float)) for v in geometry[:2]):
            return [(float(geometry[0]), float(geometry[1]))]
        return [(float(p[0]), float(p[1])) for p in geometry]
    raise TypeError(f"Unsupported geometry format: {type(geometry).__name__}")


def haversine_m(a, b):
    lon1, lat1 = map(math.radians, a); lon2, lat2 = map(math.radians, b)
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371008.8 * 2 * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(a, b):
    lon1, lat1 = map(math.radians, a); lon2, lat2 = map(math.radians, b)
    dlon = lon2-lon1
    y = math.sin(dlon)*math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def project_to_polyline(point, line):
    """Return nearest distance (m), along-line bearing, and line length (m)."""
    if not line:
        return float("inf"), None, 0.0
    if len(line) == 1:
        return haversine_m(point, line[0]), None, 0.0
    lat0 = math.radians(point[1]); scale_x = 6371008.8*math.cos(lat0)*math.pi/180
    scale_y = 6371008.8*math.pi/180
    px, py = point[0]*scale_x, point[1]*scale_y
    best, best_bearing, total = float("inf"), None, 0.0
    for a,b in zip(line, line[1:]):
        ax, ay = a[0]*scale_x, a[1]*scale_y; bx, by = b[0]*scale_x, b[1]*scale_y
        dx, dy = bx-ax, by-ay; seg2 = dx*dx+dy*dy
        seglen = math.sqrt(seg2); total += seglen
        t = 0 if seg2 == 0 else max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/seg2))
        dist = math.hypot(px-(ax+t*dx), py-(ay+t*dy))
        if dist < best:
            best, best_bearing = dist, bearing_deg(a,b) if seglen else None
    return best, best_bearing, total


def circular_difference(a, b):
    return abs((a-b+180) % 360 - 180)
