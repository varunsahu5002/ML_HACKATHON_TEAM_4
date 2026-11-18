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

The `student_test/` partition is reserved for the designated final evaluation stage.

It is not used for model training or model selection.

The evaluator-provided test data is also kept separate until the team is ready to perform the final evaluation according to the hackathon instructions.

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

Supervised labels are generated using the available reference route information.

The label-alignment process performs sequence-aware alignment between GPS observations and reference route arcs.

The aligned reference road IDs are compared with generated candidate road IDs to determine candidate labels.

The current training-data builder applies the following filtering conditions:

- Retains observations having exactly one positive candidate.
- Uses a provisional maximum alignment error threshold of 30 metres.
- Excludes observations that do not satisfy these conditions.

This produces the labelled candidate-road dataset used for supervised machine learning.

## 10. Machine Learning Implementation

The ML module trains and compares multiple classifiers to identify the most suitable baseline model for candidate-road classification.

### Training dataset statistics

| Metric | Value |
|---|---:|
| Training trips | 80 |
| Candidate rows | 1,234,949 |
| Positive labels | 181,257 |
| Negative labels | 1,053,692 |
| Predictive features | 7 |
| Number of classes | 2 |

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

**5-Fold GroupKFold Cross-Validation**

The trip identifier is used as the grouping variable.

This ensures that observations belonging to the same trip remain within the same fold.

### Evaluation metrics

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1-score
- Log Loss
- Training time
- Prediction time

F1-score is the primary model-selection metric because the positive and negative classes are imbalanced.

### Actual model comparison results

The following results were obtained from the completed five-model GroupKFold experiment.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.873575 | 0.545386 | 0.827797 | **0.657445** |
| Hist Gradient Boosting | **0.907599** | **0.726176** | 0.593441 | 0.652932 |
| Extra Trees | 0.850437 | 0.493898 | **0.874966** | 0.631302 |
| SGD Classifier | 0.825922 | 0.451440 | 0.874317 | 0.595324 |
| Logistic Regression | 0.825395 | 0.450558 | 0.874006 | 0.594488 |

### Selected baseline: Random Forest

Random Forest achieved the highest mean F1-score among the five evaluated models.

Its cross-validation results were:

| Metric | Result |
|---|---:|
| Accuracy | 87.36% |
| Precision | 54.54% |
| Recall | 82.78% |
| F1-score | 0.657445 |
| Log Loss | 0.273482 |

Hist Gradient Boosting achieved higher accuracy and lower log loss, while Random Forest achieved the highest F1-score.

Therefore, Random Forest was selected as the baseline classifier using mean F1-score as the primary selection criterion.

### Separate held-out validation

An additional evaluation was performed using one held-out fold from a five-fold GroupKFold split.

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
| F1-score | 0.618422 |

Confusion matrix:

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 185,711 | 26,872 |
| Positive | 7,955 | 28,222 |

These metrics belong to a separate held-out fold and are not the five-fold average reported in the model comparison table.

The generated outputs are:

- `confusion_matrix.csv`
- `confusion_matrix.png`

## 12. Model Saving and Probability Prediction

The selected baseline model is serialized using Joblib.

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

Model loading and probability prediction were successfully verified using a small sample from the development training dataset.

This verification confirms that the saved model can be loaded and used for inference. It does not represent final unseen-data evaluation.

## 13. Repository Structure

The project uses a root-level modular Python structure.

```text
ML_HACKATHON_TEAM_4/
|
|-- data/
|   |-- train/
|   |-- student_test/
|
|-- models/
|   |-- __init__.py
|   |-- logistic_regression_model.py
|   |-- random_forest_model.py
|   |-- extra_trees_model.py
|   |-- gradient_boosting_model.py
|   |-- sgd_model.py
|
|-- artifacts/
|   |-- best_model.joblib       # Local model artifact; not pushed
|
|-- data_loader.py
|-- data_exploration.py
|-- preprocessing.py
|-- candidate_generation.py
|-- feature_engineering.py
|-- feature_utils.py
|-- label_alignment.py
|-- build_training_data.py
|
|-- model_training.py
|-- model_selection.py
|-- model_utils.py
|-- train_models.py
|-- generate_confusion_matrix.py
|
|-- FEATURE_ENGINEERING.md
|-- feature_matrix_sample.csv
|-- alignment_test.csv
|-- model_comparison.csv
|-- model_config.json
|-- confusion_matrix.csv
|-- confusion_matrix.png
|
|-- .gitignore
|-- README.md
```

**Note:** The tree documents the project files and local working artifacts. Some files, including generated datasets and the saved model, are not included in the GitHub commit. Check the repository branch for the exact committed file list.

### Module responsibilities

| File | Responsibility |
|---|---|
| `data_loader.py` | Loads GPS tracks, road nodes, arcs and reference routes |
| `data_exploration.py` | Explores dataset characteristics |
| `preprocessing.py` | Cleans and prepares GPS observations |
| `candidate_generation.py` | Generates nearby candidate road segments |
| `feature_engineering.py` | Extracts candidate-road features |
| `feature_utils.py` | Provides supporting feature calculations |
| `label_alignment.py` | Aligns GPS observations with reference routes |
| `build_training_data.py` | Builds the labelled training feature matrix |
| `model_training.py` | Provides model construction utilities |
| `model_selection.py` | Performs cross-validation and model comparison |
| `model_utils.py` | Handles model saving, loading and inference |
| `train_models.py` | Runs baseline model training and selection |
| `generate_confusion_matrix.py` | Generates a held-out validation confusion matrix |

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
pip install numpy pandas scipy scikit-learn joblib matplotlib seaborn optuna
```

### Step 4: Generate the training feature matrix

Ensure that the development training data is available under `data/train/`.

```bash
python build_training_data.py
```

This generates the labelled `training_features.csv` dataset.

### Step 5: Train and compare baseline models

```bash
python train_models.py
```

This performs the five-fold GroupKFold comparison, selects the best baseline model according to mean F1-score and saves the trained model.

### Step 6: Generate the confusion matrix

```bash
python generate_confusion_matrix.py
```

This performs a separate held-out GroupKFold evaluation and generates the confusion matrix and model configuration outputs.

**Important:** The generated training dataset and saved model artifact may need to be obtained separately because they are not included in the normal GitHub commit.

## 16. Current Development Status

| Component | Status |
|---|---|
| Data loading | Implemented |
| GPS preprocessing | Implemented |
| Road network construction | Implemented |
| Candidate generation | Implemented |
| Feature engineering | Implemented |
| Ground-truth label alignment | Implemented |
| Training feature generation | Completed for 80 trips |
| Five baseline ML models | Implemented |
| GroupKFold model comparison | Completed |
| Baseline model selection | Completed |
| Model saving and loading | Verified locally |
| Probability prediction | Verified locally |
| Confusion matrix | Generated |
| Model configuration | Generated |
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

**Development stage:** Baseline ML implementation and validation completed; full-system integration and final evaluation pending.
