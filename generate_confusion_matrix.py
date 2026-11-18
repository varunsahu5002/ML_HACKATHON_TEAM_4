
import json
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from models.random_forest_model import build_model

DATA_PATH = "training_features.csv"

FEATURE_COLUMNS = [
    "distance_to_road_m",
    "perpendicular_distance_m",
    "road_length_m",
    "gps_movement_distance_m",
    "gps_movement_bearing_deg",
    "gps_speed_mps",
    "bearing_difference_deg",
]

TARGET_COLUMN = "target_label"
GROUP_COLUMN = "trip_id"


def main():
    print("Loading development training features...")
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=[TARGET_COLUMN, GROUP_COLUMN]).copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    groups = df[GROUP_COLUMN]

    print("Rows:", len(df))
    print("Trips:", groups.nunique())
    print("Classes:", sorted(y.unique()))

    # One trip-group holdout fold; student_test is never accessed.
    splitter = GroupKFold(n_splits=5)
    train_idx, validation_idx = next(splitter.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[validation_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[validation_idx]

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Training trips:", groups.iloc[train_idx].nunique())
    print("Validation trips:", groups.iloc[validation_idx].nunique())

    model = build_model()

    print("\nTraining Random Forest...")
    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start:.2f} seconds")

    predictions = model.predict(X_val)

    cm = confusion_matrix(y_val, predictions, labels=[0, 1])

    print("\nConfusion Matrix")
    print(cm)
    print("\nValidation Metrics")
    print("Accuracy :", accuracy_score(y_val, predictions))
    print("Precision:", precision_score(y_val, predictions, zero_division=0))
    print("Recall   :", recall_score(y_val, predictions, zero_division=0))
    print("F1-score :", f1_score(y_val, predictions, zero_division=0))

    # Save numeric confusion matrix.
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    cm_df.to_csv("confusion_matrix.csv")

    # Save visual confusion matrix.
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest - GroupKFold Holdout Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()

    # Save reproducibility/configuration details.
    classifier = model.named_steps["classifier"]

    config = {
        "selected_model": "Random Forest",
        "model_type": type(classifier).__name__,
        "model_parameters": classifier.get_params(),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "validation_method": "One held-out fold from 5-fold GroupKFold",
        "training_partition": "data/train only, through training_features.csv",
        "student_test_accessed": False,
        "confusion_matrix_labels": {
            "0": "Negative candidate",
            "1": "Positive candidate"
        }
    }

    with open("model_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, default=str)

    print("\nCreated:")
    print("- confusion_matrix.csv")
    print("- confusion_matrix.png")
    print("- model_config.json")


if __name__ == "__main__":
    main()
