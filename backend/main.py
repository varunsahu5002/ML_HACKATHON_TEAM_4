"""FastAPI application for scoring PathSnap road candidates."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
FEATURE_COLUMNS = [
    "distance_to_road_m",
    "perpendicular_distance_m",
    "road_length_m",
    "gps_movement_distance_m",
    "gps_movement_bearing_deg",
    "gps_speed_mps",
    "bearing_difference_deg",
]


class Candidate(BaseModel):
    """Features for one GPS observation and one candidate road."""

    model_config = ConfigDict(extra="forbid")

    distance_to_road_m: float | None = Field(ge=0)
    perpendicular_distance_m: float | None = Field(ge=0)
    road_length_m: float | None = Field(ge=0)
    gps_movement_distance_m: float | None = Field(ge=0)
    gps_movement_bearing_deg: float | None = Field(ge=0, le=360)
    gps_speed_mps: float | None = Field(ge=0)
    bearing_difference_deg: float | None = Field(ge=0, le=180)
    candidate_road_id: str | int | None = None


class PredictionRequest(BaseModel):
    """A candidate set to score for a single GPS observation."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(min_length=1)


def _load_artifact(path: Path) -> tuple[Any, list[str], str | None]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Place Member 1's best_model.joblib "
            "there or set PATHSNAP_MODEL_PATH."
        )
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        columns = artifact.get("feature_columns")
        name = artifact.get("model_name")
    else:
        raise ValueError("Model artifact must be a dictionary produced by model_utils.save_model.")
    if model is None or not isinstance(columns, list):
        raise ValueError("Model artifact must contain 'model' and 'feature_columns'.")
    if columns != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature_columns do not match the PathSnap API contract. "
            f"Expected {FEATURE_COLUMNS}; got {columns}."
        )
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Loaded model does not implement predict().")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Loaded model does not implement predict_proba().")
    return model, columns, name


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("PATHSNAP_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    app.state.model = None
    app.state.feature_columns = FEATURE_COLUMNS
    app.state.model_name = None
    app.state.model_error = None
    try:
        app.state.model, app.state.feature_columns, app.state.model_name = _load_artifact(model_path)
    except Exception as exc:
        # Keep health available so a deployment can diagnose a missing/bad artifact.
        app.state.model_error = str(exc)
    yield


app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "detail": exc.errors()},
    )


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    result = {"status": "ok" if loaded else "degraded", "model_loaded": loaded}
    if loaded:
        result["model_name"] = request.app.state.model_name
    else:
        result["model_error"] = request.app.state.model_error
    return result


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request):
    model = request.app.state.model
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "model_unavailable",
                "detail": request.app.state.model_error or "Model is not loaded.",
            },
        )

    rows = []
    for candidate in payload.candidates:
        values = candidate.model_dump(exclude={"candidate_road_id"})
        rows.append(values)
    frame = pd.DataFrame(rows, columns=request.app.state.feature_columns, dtype=float)
    try:
        labels = model.predict(frame)
        probabilities = model.predict_proba(frame)
        classes = list(model.classes_)
        positive_index = classes.index(1)
        scores = np.asarray(probabilities)[:, positive_index]
    except (ValueError, TypeError, IndexError, AttributeError) as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "prediction_failed", "detail": str(exc)},
        )

    results = []
    for candidate, label, score in zip(payload.candidates, labels, scores):
        results.append(
            {
                "candidate_road_id": candidate.candidate_road_id,
                "prediction": int(label),
                "match_probability": float(score),
            }
        )
    return {"predictions": results, "model_name": request.app.state.model_name}
