# ML_HACKATHON_TEAM_4
# PathSnap – GPS Trajectory Map Matching

**ML_HACKATHON_TEAM_4 | Machine Learning Hackathon Project**

PathSnap is a GPS trajectory map-matching system designed to identify the most likely road segments corresponding to noisy GPS observations. It combines GPS preprocessing, road candidate generation, feature engineering, machine learning and sequential map-matching techniques to reconstruct probable vehicle routes.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [System Architecture](#4-system-architecture)
5. [Dataset Organization](#5-dataset-organization)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Road Network and Candidate Generation](#7-road-network-and-candidate-generation)
8. [Feature Engineering](#8-feature-engineering)
9. [Ground-Truth Label Alignment](#9-ground-truth-label-alignment)
10. [Machine Learning Implementation](#10-machine-learning-implementation)
11. [Model Evaluation and Comparison](#11-model-evaluation-and-comparison)
12. [Model Saving and Probability Prediction](#12-model-saving-and-probability-prediction)
13. [Repository Structure](#13-repository-structure)
14. [Technology Stack](#14-technology-stack)
15. [Installation and Execution](#15-installation-and-execution)
16. [Current Development Status](#16-current-development-status)
17. [Future Improvements](#17-future-improvements)

---

## 1. Project Overview

GPS-based navigation systems frequently encounter inaccurate location measurements due to signal interference, urban infrastructure, environmental conditions and GPS device limitations.

Consequently, raw GPS coordinates may not accurately represent the actual roads travelled by a vehicle.

PathSnap addresses this challenge by identifying possible road segments for each GPS observation and using machine learning to estimate the likelihood of each candidate being the correct road.

The project follows a modular approach consisting of:

- GPS trajectory preprocessing
- Road network construction
- Candidate road generation
- Feature extraction and engineering
- Ground-truth route alignment
- Machine learning-based candidate classification
- Probability generation for downstream sequential inference
- HMM/Viterbi-based route selection and evaluation

## 2. Problem Statement

Given a sequence of noisy GPS observations and an associated road network, determine the most likely sequence of road segments travelled by a vehicle.

The system must account for GPS measurement errors, spatial proximity, movement direction, vehicle speed and road network structure.

The objective is to improve the reliability of GPS-to-road assignments by combining machine learning with sequential map-matching techniques.

## 3. Project Objectives

1. Load and inspect GPS trajectory and road network data.
2. Clean and preprocess raw GPS observations.
3. Generate possible road candidates for each GPS observation.
4. Extract meaningful spatial and movement-related features.
5. Generate supervised training labels using reference route information.
6. Train and compare multiple machine learning classifiers.
7. Select an appropriate baseline model using group-aware validation.
8. Generate candidate-road probabilities for downstream inference.
9. Integrate machine learning predictions with sequential map matching.
10. Evaluate the complete system using designated evaluation data.

## 4. System Architecture

PathSnap follows a modular processing pipeline.

```text
                 Raw GPS Trajectories
                          |
                          v
                   Data Loading
                          |
                          v
                  GPS Preprocessing
                          |
                          v
                Road Network Construction
                          |
                          v
                  Candidate Generation
                          |
                          v
                  Feature Engineering
                          |
                          v
               Ground-Truth Label Alignment
                          |
                          v
                ML Model Training
                          |
                          v
             Candidate-Road Probabilities
                          |
                          v
                    HMM / Viterbi
                          |
                          v
               Predicted Road Sequence
                          |
                          v
                 Final Evaluation
```

Each stage is implemented as a separate Python module to support modular development and team collaboration.

## 5. Dataset Organization

The development data is organized into separate partitions.

```text
data/
|
|-- train/
|
|-- student_test/
```

### Training partition

The training partition contains 80 trips, with information including:

- GPS track observations
- Road network nodes
- Road network arcs
- Reference route information

The training partition is used for preprocessing, candidate generation, feature engineering, supervised label generation, model training and cross-validation.

### Student test partition

The `student_test/` partition contains 10 trips and is reserved for development-stage evaluation after model choices have been made using training-trip cross-validation. Do not use its labels for fitting, cross-validation, feature selection, hyperparameter or threshold optimization, or model selection. Report its results separately and do not use them to revise decisions.

### Evaluator-test partition

The evaluator-test partition contains 10 trips and is final-evaluation-only. It must remain untouched during development: do not inspect its ground truth or use it for fitting, validation, feature selection, tuning, model/threshold selection, or development comparisons. The prediction path must not load its route file.

## 6. Data Preprocessing

GPS preprocessing converts raw trajectory observations into structured data suitable for subsequent processing.

The preprocessing operations include:

- Cleaning GPS observations
- Sorting observations chronologically using timestamps
- Handling duplicate timestamps
- Filtering unrealistic speed outliers
- Generating observation identifiers
- Converting geographic coordinates into local projected coordinates
- Calculating time differences
- Calculating distances between consecutive observations
- Estimating movement speed
- Calculating movement bearing
- Identifying GPS gaps
- Creating trajectory segment identifiers

The resulting processed GPS observations are passed to candidate generation and feature engineering.

## 7. Road Network and Candidate Generation

The road network is constructed from the available node and arc information.

Candidate generation identifies road segments located near individual GPS observations.

The current training-data builder uses the following candidate-generation configuration:

| Parameter | Value |
|---|---:|
| Search radius | 100 metres |
| Maximum candidates | 8 |

Each GPS observation can therefore be associated with multiple possible road segments.

These candidate roads are subsequently evaluated using spatial and movement-related features.

## 8. Feature Engineering

Feature engineering converts GPS observations and candidate road information into numerical representations for machine learning.

The current ML pipeline uses seven predictive features.

| Feature | Description |
|---|---|
| `distance_to_road_m` | Distance between the GPS observation and candidate road |
| `perpendicular_distance_m` | Perpendicular distance to the candidate road |
| `road_length_m` | Length of the candidate road |
| `gps_movement_distance_m` | Distance travelled between consecutive GPS observations |
| `gps_movement_bearing_deg` | Direction of GPS movement |
| `gps_speed_mps` | Estimated GPS movement speed in metres per second |
| `bearing_difference_deg` | Difference between GPS movement direction and candidate road direction |

The target variable is:

`target_label`

- `1` – Positive candidate-road match
- `0` – Negative candidate-road match

Identifiers and label-related columns are excluded from the predictive feature matrix.

## 9. Ground-Truth Label Alignment

The PathSnap `.route` file is a chronological sequence of arc IDs, not a per-GPS-observation label table. The repository's `optimize_model.py` training builder labels each observation's nearest generated candidate whose arc appears anywhere in that trip's route. This is a route-membership heuristic; it does not align route sequence position to GPS time, and its labels and derived metrics should be treated as approximate. It does not pair route and track rows by index. The source training archive is supplied separately; the generated `training_features.csv` is not checked in.

## 10. Machine Learning Implementation

The ML module trains and compares multiple classifiers to identify the most suitable baseline model for candidate-road classification.

### Training dataset statistics

| Metric | Value |
|---|---:|
| Training trips | 80 |
| Candidate rows (previously reported) | 1,234,949 |
| Positive labels (previously reported) | 181,257 |
| Negative labels (previously reported) | 1,053,692 |
| Predictive features | 7 |
| Number of classes | 2 |

These training-row counts were recorded in the earlier README; the feature matrix is missing from this repository, so the counts have not been independently checked.

The task is formulated as a binary classification problem.

### Implemented models

Five classifiers were implemented:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification baseline |
| Random Forest | Ensemble classifier based on decision trees |
| Extra Trees | Randomized tree ensemble |
| Hist Gradient Boosting | Gradient boosting classifier |
| SGD Classifier | Linear classifier trained using stochastic gradient descent |

Each classifier is implemented separately under the `models/` directory and exposes a reusable `build_model()` function.

### ML pipeline

The implemented model pipelines use:

- Median imputation for missing feature values.
- Standard scaling for linear models.
- Class balancing where supported by the selected classifier configuration.
- Reusable scikit-learn pipelines.

## 11. Model Evaluation and Comparison

### Validation methodology

The dataset contains multiple candidate observations belonging to the same GPS trip.

Randomly splitting individual observations could allow information from the same trip to appear in both training and validation data.

To reduce this leakage risk, the model comparison process uses:

**5-Fold GroupKFold Cross-Validation**, grouping by the `trip_id` column.

The trip identifier is used as the grouping variable.

This ensures that observations belonging to the same trip remain within one fold. Corrected evaluation calculates pooled out-of-fold metrics over all validation rows; it also reports fold Macro-F1 mean and standard deviation to show fold variability. Split order is deterministic (`shuffle=False`).

### Evaluation metrics

The corrected comparison reports:

- Accuracy.
- Positive-class precision, recall, and binary F1.
- Macro precision, macro recall, and Macro F1 across both classes.
- Log loss when all evaluated models provide class probabilities.
- Observation-level top-1 accuracy when candidate observation identifiers are present.
- Candidate miss rate only when reference roads and the complete candidate sets are available; it cannot be recovered from a filtered candidate-row file.
- Route continuity only when decoded route sequences and directed road adjacency are available.
- Total cross-validation training and prediction times, plus prediction time per row.

Macro F1 is the corrected baseline selection metric. Binary F1 remains separately named for positive-candidate performance; it is not a substitute for Macro F1. Metrics unavailable from the supplied evaluation inputs are reported as N/A, not inferred.

### Phase 1 recorded results (not reproduced)

The committed `model_comparison.csv` contains the following Phase 1 values, with a binary `f1` column. The original code averaged per-fold metrics without weighting by validation-row count; the corresponding `training_features.csv`, label builder, fold-level predictions, and experiment log are absent from this checkout, so these figures have not been independently reproduced or verified. `model_comparison_phase1_preserved.csv` preserves this original table. Do not interpret it as the corrected pooled Macro-F1 comparison.

| Model | Accuracy | Positive precision | Positive recall | Binary F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.873575 | 0.545386 | 0.827797 | **0.657445** |
| Hist Gradient Boosting | **0.907599** | **0.726176** | 0.593441 | 0.652932 |
| Extra Trees | 0.850437 | 0.493898 | **0.874966** | 0.631302 |
| SGD Classifier | 0.825922 | 0.451440 | 0.874317 | 0.595324 |
| Logistic Regression | 0.825395 | 0.450558 | 0.874006 | 0.594488 |

### Phase 1 provisional baseline: Random Forest

Random Forest has been recorded as the Phase 1 provisional choice using binary F1. It has not yet been selected under the corrected pooled out-of-fold Macro-F1 criterion because the Phase 1 training data and feature-generation artifacts are unavailable for rerunning the comparison.

Its cross-validation results were:

| Metric | Result |
|---|---:|
| Accuracy | 87.36% |
| Precision | 54.54% |
| Recall | 82.78% |
| Binary F1 | 0.657445 |
| Log Loss | 0.273482 |

In the unverified Phase 1 table, Hist Gradient Boosting has higher reported accuracy and lower reported log loss, while Random Forest has the highest reported binary F1.

The corrected training script selects by pooled out-of-fold Macro F1 when the missing training feature matrix is provided. Until that run completes, Random Forest is a historical provisional baseline only.

### Phase 1 single-fold artifact (not independent validation)

The checked-in confusion-matrix artifacts and values below are historical Phase 1 results from one fold of the same five-fold training-trip GroupKFold procedure, not an independent holdout. Their input matrix is absent, so they have not been reproduced in this audit. The corrected generator now creates a pooled out-of-fold confusion matrix so its metrics align with the five-fold comparison.

| Metric | Value |
|---|---:|
| Training trips | 64 |
| Validation trips | 16 |
| Training rows | 986,189 |
| Validation rows | 248,760 |
| Training time | 132.83 seconds |
| Accuracy | 85.9998% |
| Precision | 51.2252% |
| Recall | 78.0109% |
| Binary F1 | 0.618422 |

Confusion matrix:

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 185,711 | 26,872 |
| Positive | 7,955 | 28,222 |

These metrics belong to one historical fold and are not pooled five-fold results.

The generated outputs are:

- `confusion_matrix.csv`
- `confusion_matrix.png`

## 12. Model Saving and Probability Prediction

When `train_models.py` completes with the required training-only feature matrix, it serializes the model selected by pooled out-of-fold Macro F1 using Joblib. The checked-in Phase 1 configuration is historical and does not establish the corrected selection.

The model utility module supports:

- Saving trained models
- Loading saved models
- Retrieving the selected feature list
- Generating class predictions
- Generating class probabilities

The model supports `predict_proba()`, allowing downstream modules to access candidate-road probabilities.

### Example: Loading the model

```python
from model_utils import load_model

model, feature_columns, model_name = load_model(
    "artifacts/best_model.joblib"
)

print("Selected model:", model_name)
print("Features:", feature_columns)
```

### Example: Generating predictions

```python
import pandas as pd

X = pd.read_csv(
    "training_features.csv",
    usecols=feature_columns,
    nrows=10
)

predictions = model.predict(X)

probabilities = model.predict_proba(X)[:, 1]

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

The model-loading example uses the checked-in provisional Phase 2 artifact at `artifacts/best_model.joblib`. It was fit on all rows from the 80-trip `train` partition using the route-membership labeling heuristic above. The selected baseline HistGradientBoosting pipeline performs median imputation and exposes the seven features above. The comparison used two-fold GroupKFold and one trial per family; this is an API-ready development artifact, not a substitute for the full five-model comparison and Bayesian optimization described in the hackathon guidance. Treat its metrics as preliminary.

## 13. Repository Structure

The project uses a root-level modular Python structure.

The current checkout contains the data loader and preprocessing modules, candidate and feature modules, five model definitions, baseline training/selection scripts, and the `evaluation/` package with metric helpers and tests. The training archive is supplied separately; the generated `training_features.csv` is not checked in. `artifacts/best_model.joblib` is the Phase 2 fitted inference artifact. `model_comparison.csv`, `model_config.json`, and confusion-matrix files are historical outputs. `baseline_model_configs.json` records the current five model pipelines as defined in source; it contains no experiment results.

### Module responsibilities

| File | Responsibility |
|---|---|
| `data_loader.py` | Loads GPS tracks, road nodes, arcs and reference routes |
| `data_exploration.py` | Explores dataset characteristics |
| `preprocessing.py` | Cleans and prepares GPS observations |
| `candidate_generation.py` | Generates nearby candidate road segments |
| `feature_engineering.py` | Extracts candidate-road features |
| `feature_utils.py` | Provides supporting feature calculations |
| `model_training.py` | Provides model construction utilities |
| `model_selection.py` | Performs cross-validation and model comparison |
| `model_utils.py` | Handles model saving, loading and inference |
| `train_models.py` | Runs baseline model training and selection |
| `generate_confusion_matrix.py` | Generates a pooled out-of-fold training-trip confusion matrix |

## 14. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Main implementation language |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| SciPy | Scientific computation |
| Scikit-learn | Machine learning and evaluation |
| Joblib | Model serialization |
| Matplotlib | Confusion matrix visualization |
| Seaborn | Data visualization |
| Optuna | Optional future hyperparameter optimization |

## 15. Installation and Execution

### Phase 2 FastAPI backend

The backend scores one or more road candidates for a single GPS observation. It loads Member 1's finalized Joblib artifact once during application startup. By default, the artifact is expected at `artifacts/best_model.joblib` and must use the seven features in the order recorded by `model_utils.save_model`. The trained pipeline performs its own training-matched median imputation; the API does not fit or apply a second preprocessor.

Install and start from the repository root:

```bash
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

If the artifact is stored elsewhere, set `PATHSNAP_MODEL_PATH` to its path before starting Uvicorn. Relative paths are resolved from the repository root. `/health` remains available when the artifact is missing or invalid and reports `model_loaded: false`; `/predict` then returns HTTP 503 with a diagnostic.

#### `GET /health`

Returns HTTP 200 and whether the model loaded:

```json
{"status":"ok","model_loaded":true,"model_name":"Tuned Random Forest"}
```

When startup cannot load the artifact, `status` is `degraded`, `model_loaded` is `false`, and `model_error` explains why.

#### `POST /predict`

Provide a non-empty `candidates` array. Each candidate contains the seven numeric model features; `candidate_road_id` is optional metadata returned with its score. Features can be `null` when unavailable, matching the model pipeline's median imputation. Unknown fields, invalid ranges, or missing required feature keys receive HTTP 422.

Request example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"candidates":[{"candidate_road_id":123,"distance_to_road_m":8.4,"perpendicular_distance_m":8.4,"road_length_m":120.0,"gps_movement_distance_m":15.2,"gps_movement_bearing_deg":87.0,"gps_speed_mps":7.6,"bearing_difference_deg":12.0},{"candidate_road_id":456,"distance_to_road_m":25.0,"perpendicular_distance_m":25.0,"road_length_m":80.0,"gps_movement_distance_m":15.2,"gps_movement_bearing_deg":87.0,"gps_speed_mps":7.6,"bearing_difference_deg":74.0}]}'
```

Response example:

```json
{"predictions":[{"candidate_road_id":123,"prediction":1,"match_probability":0.91},{"candidate_road_id":456,"prediction":0,"match_probability":0.08}],"model_name":"Tuned Random Forest"}
```

`prediction` is the model's class (`1` for a positive candidate match, `0` otherwise); `match_probability` is the probability assigned to class `1`. Candidates are returned in request order. Interactive API documentation is available at `/docs`.

### Prerequisites

- Python 3.11
- Git
- Visual Studio Code or another Python IDE

### Step 1: Clone the repository

```bash
git clone https://github.com/varunsahu5002/ML_HACKATHON_TEAM_4.git

cd ML_HACKATHON_TEAM_4
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install numpy pandas scipy scikit-learn joblib matplotlib seaborn optuna pytest
```

### Step 4: Provide the training feature matrix

This checkout currently lacks both `build_training_data.py` and `training_features.csv`. The training command below cannot run until the team's training-only label-alignment/feature-generation workflow is restored and the resulting training matrix is supplied. The `.route` sequence must not be matched to `.track` rows by index.

### Step 5: Train and compare baseline models

```bash
python train_models.py
```

This performs five-fold trip-grouped cross-validation, selects by pooled out-of-fold Macro F1, records the corrected metrics/configuration, preserves the original comparison table on the first corrected run, and saves the selected model. It reads only `training_features.csv`.

### Step 6: Generate the confusion matrix

```bash
python generate_confusion_matrix.py
```

This reruns Random Forest across the same five training-trip folds and writes pooled out-of-fold confusion-matrix diagnostics. These diagnostics are not a separate test score. Existing Phase 1 artifacts are preserved with a `_phase1_preserved` suffix before first replacement.

**Note:** The training dataset and generated feature table are kept outside the repository. The compact fitted inference artifact is checked in at `artifacts/best_model.joblib`.

### Run the evaluation tests

```bash
python -m pytest -q
```

The tests cover metric semantics, candidate misses, trip-disjoint/repeatable GroupKFold splits, leakage-safe feature selection, and execution of all five model pipelines on a small synthetic fixture. That fixture verifies code paths only; it is not a PathSnap performance result.

## 16. Current Development Status

| Component | Status |
|---|---|
| Data loading | Implemented |
| GPS preprocessing | Implemented |
| Road network construction | Implemented |
| Candidate generation | Implemented |
| Feature engineering | Implemented |
| Ground-truth label alignment | Not present in this checkout; must be restored before training |
| Training feature generation | Not reproducible from this checkout; builder and feature matrix are absent |
| Five baseline ML models | Implemented |
| Phase 1 GroupKFold result | Historical CSV exists; training inputs and run log are absent, so it is unverified |
| Corrected baseline comparison | Code updated; pending a run with the training feature matrix |
| Baseline model selection | Pending corrected pooled Macro-F1 run |
| Student-test evaluation | Pending; use only after training-trip decisions and do not use labels to revise them |
| Evaluator-test data | Not accessed; reserved for final evaluation |
| Model saving and loading | Code implemented; no artifact checked in for clean-checkout verification |
| Probability prediction | Supported by model code; no saved artifact checked in |
| Confusion matrix | Historical Phase 1 artifact; corrected pooled OOF generator is pending the missing training matrix |
| Model configuration | Five current estimator/pipeline configurations recorded in `baseline_model_configs.json`; run-specific config pending a training run |
| HMM/Viterbi integration | Pending integration verification |
| Final evaluator testing | Pending |

## 17. Future Improvements

Potential future improvements include:

- Hyperparameter optimization and model tuning.
- Feature correlation and feature importance analysis.
- Probability calibration.
- Improved candidate-generation strategies.
- More robust handling of noisy GPS observations and large trajectory gaps.
- HMM/Viterbi integration and route continuity analysis.
- Comprehensive evaluation using the designated evaluator dataset.

---

## Repository

[ML_HACKATHON_TEAM_4 – GitHub](https://github.com/varunsahu5002/ML_HACKATHON_TEAM_4)

**Project:** PathSnap – GPS Trajectory Map Matching

**Development stage:** Phase 1 baseline is recorded but not reproducible from this checkout. Corrected metric/CV code is in preparation; training-data generation and a verified baseline run remain pending. Hyperparameter optimization has not started.
