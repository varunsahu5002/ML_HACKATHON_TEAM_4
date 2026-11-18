
"""Save and load trained map-matching models."""

import os
import joblib


ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.joblib")


def save_model(model, feature_columns, model_name, metrics=None):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": list(feature_columns),
        "model_name": model_name,
        "metrics": metrics or {}
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


def load_model(path=MODEL_PATH):
    artifact = joblib.load(path)

    return (
        artifact["model"],
        artifact["feature_columns"],
        artifact["model_name"]
    )


def predict_probabilities(model, X):
    """Return probability that each candidate is the correct road."""

    probabilities = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)

    return probabilities[:, positive_index]
