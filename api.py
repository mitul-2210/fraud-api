from __future__ import annotations

import json
from pathlib import Path
import os
import logging
from typing import List
import boto3

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Prefer SageMaker model directory if present; fallback to app directory
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("SM_MODEL_DIR", str(APP_DIR)))
MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


class PredictRequest(BaseModel):
    # 30 features: Time, V1..V28, Amount (order must match training columns)
    features: List[float] = Field(min_length=30, max_length=30)


class PredictResponse(BaseModel):
    is_fraud: bool
    probability_fraud: float


def _maybe_download_from_s3() -> None:
    bucket = os.getenv("MODEL_S3_BUCKET")
    prefix = os.getenv("MODEL_S3_PREFIX")
    if not bucket or not prefix:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    targets = {
        "model.joblib": MODEL_PATH,
        "feature_names.json": FEATURE_NAMES_PATH,
    }
    for key, dest in targets.items():
        if not dest.exists():
            s3_key = f"{prefix.rstrip('/')}/{key}"
            logging.info(f"Downloading s3://{bucket}/{s3_key} -> {dest}")
            s3.download_file(bucket, s3_key, str(dest))


def load_artifacts():
    if not MODEL_PATH.exists() or not FEATURE_NAMES_PATH.exists():
        _maybe_download_from_s3()
    if not MODEL_PATH.exists() or not FEATURE_NAMES_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Provide MODEL_S3_BUCKET/PREFIX or run 'python train_model.py'."
        )
    model = joblib.load(MODEL_PATH)
    feature_names: List[str] = json.loads(FEATURE_NAMES_PATH.read_text())
    return model, feature_names


model, feature_names = load_artifacts()
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        x = np.array(req.features, dtype=float).reshape(1, -1)
        proba = getattr(model, "predict_proba", None)
        if proba is None:
            # Fallback if model lacks predict_proba
            pred_label = int(model.predict(x)[0])
            return PredictResponse(is_fraud=bool(pred_label), probability_fraud=float(pred_label))
        prob = float(proba(x)[0][1])
        pred_label = prob >= 0.5
        return PredictResponse(is_fraud=bool(pred_label), probability_fraud=prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# SageMaker compatibility endpoints
@app.get("/ping")
def ping():
    # Health check endpoint expected by SageMaker
    return {"status": "ok"}


@app.post("/invocations")
def invocations(raw_body: bytes = None, content_type: str | None = None):
    # Accept JSON or CSV payloads for SageMaker runtime
    try:
        ct = content_type or "application/json"
        if "json" in ct:
            body = raw_body.decode("utf-8")
            payload = json.loads(body)
            if isinstance(payload, dict) and "features" in payload:
                arr = np.array(payload["features"], dtype=float).reshape(1, -1)
            elif isinstance(payload, list):
                arr = np.array(payload, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
            else:
                raise ValueError("Invalid JSON shape. Provide {'features':[...]} or [[...],[...]].")
        elif "csv" in ct or "text/plain" in ct:
            text = raw_body.decode("utf-8").strip()
            rows = [[float(x) for x in line.split(',')] for line in text.splitlines() if line]
            arr = np.array(rows, dtype=float)
        else:
            raise ValueError(f"Unsupported content type: {ct}")

        proba = getattr(model, "predict_proba", None)
        if proba is None:
            preds = model.predict(arr).astype(int).tolist()
            probs = preds
        else:
            probs = proba(arr)[:, 1].tolist()
            preds = [int(p >= 0.5) for p in probs]
        return {
            "predictions": preds,
            "probabilities": probs,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


