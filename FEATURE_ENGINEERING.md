# PathSnap candidate and feature interface

The supplied archives were inspected. In the student dataset, each trip directory contains:

| File | Representation | Interpretation |
| --- | --- | --- |
| `.track` | Headerless whitespace rows: longitude, latitude, timestamp in seconds | GPS observations, nominally 1 Hz |
| `.nodes` | Headerless whitespace rows: longitude, latitude | Node ID is the zero-based row index |
| `.arcs` | Headerless integer pairs: from-node ID, to-node ID | Directed edge; road/arc ID is the zero-based row index |
| `.route` | One integer arc ID per row | Reference route sequence; not aligned to GPS observations |

Thus the road network is a directed graph. Arc orientation is the direction from its first node to its second; reverse travel only exists if a reverse arc is present. This dataset does not provide per-road name, class/type, speed limit, or other semantic road attributes. It also does not provide a direct GPS-observation-to-route label. The `.route` sequence must not be joined to track rows by row number. Candidate labels are created only if an input GPS record explicitly contains a `ground_truth_road_id`, `matched_road_id`, `true_road_id`, or `road_id` field; the label is returned as `target_label`, never as an input feature.

## Input and usage

The functions accept pandas DataFrames/GeoDataFrames or iterables of dictionaries. Coordinates are WGS84 `(longitude, latitude)`. GPS records use `longitude`/`latitude` (also accepts `lon`/`lat`) and may include `trip_id`, `observation_id`, and `timestamp`. Road records use `road_id` and `geometry` (a Shapely LineString, GeoJSON geometry, or coordinate sequence).

```python
from candidate_generation import build_road_network, generate_candidates
from feature_engineering import extract_features, get_feature_columns, build_feature_preprocessor

# Parse whitespace tables in the caller's data loader; preserve their row order.
roads = build_road_network(nodes, arcs)
candidates = generate_candidates(cleaned_gps, roads, radius_m=100, max_candidates=8)
matrix = extract_features(cleaned_gps, candidates, roads)
model_columns = get_feature_columns(matrix)
X = matrix[model_columns]  # Excludes identifiers and target_label.
y = matrix["target_label"] if "target_label" in matrix else None
preprocessor = build_feature_preprocessor(matrix)
X_train = preprocessor.fit_transform(X)  # Fit only on training trip rows.
```

`build_road_network(nodes, arcs)` accepts tuples or two-column dictionaries and assigns each arc its original row index as `road_id`. `generate_candidates(gps_data, road_network, radius_m=100.0, max_candidates=8, ...)` returns one row per eligible observation-road pair, sorted nearest first. It uses Shapely STRtree bounding-box queries when installed; otherwise it builds a coarse standard-library lon/lat grid. Both paths apply an exact distance check. A GPS observation with no nearby candidate gets one sentinel row with `candidate_road_id=None` and `candidate_rank=-1`.

`extract_features(gps_data, candidate_data, road_network, ...)` returns a pandas DataFrame when pandas is installed, otherwise a list of dictionaries. Feature extraction accepts Member 1's cleaned GPS output and does not implement or mutate a preprocessing pipeline.

## Output columns

| Column | Meaning |
| --- | --- |
| `trip_id` | Input trip identifier, or null if unavailable |
| `observation_id` | Input point identifier, or input row index |
| `observation_index` | Zero-based input order |
| `candidate_road_id` | Road arc row ID; null for the no-candidate sentinel |
| `distance_to_road_m` | Minimum point-to-finite-segment distance in metres |
| `perpendicular_distance_m` | Same nearest-segment distance, exposed under the perpendicular-distance name |
| `road_length_m` | Candidate arc length, calculated geodesically segment by segment |
| `gps_movement_distance_m` | Haversine distance from the immediately preceding GPS observation in the same trip |
| `gps_movement_bearing_deg` | Initial bearing of that previous-to-current GPS displacement, clockwise from north |
| `gps_speed_mps` | Movement distance divided by timestamp difference when both timestamps exist and the difference is positive |
| `bearing_difference_deg` | Circular absolute difference between GPS movement bearing and directed arc bearing (0–180 degrees) |
| `road_type` | Only emitted if an input road record has a recognized road-type field; absent for the supplied dataset |
| `target_label` | Optional 1/0 candidate match label from an explicit per-observation road ID; excluded by `get_feature_columns` |

Distance and length computations use a local metre-scale approximation for nearest projection and haversine distance for GPS displacement. The nearest projection is clamped to the finite arc endpoints. The candidate bearing is the bearing of the nearest arc segment in its stored direction. Candidate direction is meaningful for these directed arcs, so the bearing difference is not made direction-invariant. IDs and labels are excluded from `get_feature_columns`. `build_feature_preprocessor(frame, scale_numeric=True)` returns a scikit-learn `ColumnTransformer` that median-imputes and optionally scales numeric columns and imputes/one-hot encodes categorical columns. Fit it on training trip rows only, then reuse it unchanged for validation/student/evaluator rows.

## Label and partition handling

Keep evaluator/test truth out of this module's inputs. The supplied training `.route` is useful as a sequence reference but does not provide point-level alignment; no labels are inferred from it. For supervised training, point-level labels need to be derived by the owning data-preprocessing/labeling workflow using training trips only, and passed in an explicit ground-truth column. For student/evaluator prediction, omit that column. `target_label` is a target only and must never be included in `X`.

## Example artifact

`feature_matrix_sample.csv` is generated from training trip `00000001` in the supplied archive, using its track, nodes, and arcs. It demonstrates the returned columns and is an input interface example rather than a point-labeled training set. The archive itself is not duplicated in this repository.
