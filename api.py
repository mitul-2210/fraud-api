from __future__ import annotations

import json
from pathlib import Path
import os
import logging
from typing import List, Optional
import hashlib
from datetime import datetime
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

def _generate_high_risk_sample(trained_model, num: int = 4000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    # Random candidates covering broad ranges
    times = rng.integers(0, 300000, size=num).astype(float)
    v = rng.uniform(-6.0, 6.0, size=(num, 28)).astype(float)
    # Log-uniform amounts between ~10 and 5000
    amounts = np.exp(rng.uniform(np.log(10.0), np.log(5000.0), size=num)).astype(float)
    X = np.column_stack([times, v, amounts])
    proba_fn = getattr(trained_model, "predict_proba", None)
    if proba_fn is None:
        preds = trained_model.predict(X).astype(int)
        idx = int(np.argmax(preds))
    else:
        probs = proba_fn(X)[:, 1]
        idx = int(np.argmax(probs))
    return X[idx].tolist()

FRAUD_SAMPLE_FEATURES = _generate_high_risk_sample(model)
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


class RawTxnRequest(BaseModel):
    amount: float
    timestamp: Optional[float] = None
    card_brand: Optional[str] = None
    card_bin: Optional[str] = None
    last4: Optional[str] = None
    merchant_id: Optional[str] = None
    country: Optional[str] = None
    channel: Optional[str] = None
    device_os: Optional[str] = None
    device_browser: Optional[str] = None


def _hash_to_float(s: str, salt: str, scale: float = 2.0) -> float:
    h = hashlib.md5((salt + "::" + (s or "")).encode("utf-8")).hexdigest()
    val = int(h[:8], 16) / 0xFFFFFFFF
    return (val * 2.0 - 1.0) * scale


def _engineer_demo_features(raw: RawTxnRequest) -> np.ndarray:
    ts = raw.timestamp if raw.timestamp is not None else datetime.utcnow().timestamp()
    time_feat = float(int(ts) % 300000)

    v = [0.0] * 28
    fields = {
        "brand": (raw.card_brand or "unknown").lower(),
        "bin": raw.card_bin or "",
        "last4": raw.last4 or "",
        "merchant": raw.merchant_id or "",
        "country": (raw.country or "").upper(),
        "channel": (raw.channel or "ecom").lower(),
        "os": (raw.device_os or "").lower(),
        "browser": (raw.device_browser or "").lower(),
    }
    salts = [
        "brand", "bin", "last4", "merchant", "country", "channel", "os", "browser",
        "brand+merchant", "country+merchant", "bin+country", "os+browser",
        "brand+country", "channel+merchant", "bin+os", "browser+merchant",
    ]
    combos = [
        fields["brand"], fields["bin"], fields["last4"], fields["merchant"],
        fields["country"], fields["channel"], fields["os"], fields["browser"],
        fields["brand"] + ":" + fields["merchant"],
        fields["country"] + ":" + fields["merchant"],
        fields["bin"] + ":" + fields["country"],
        fields["os"] + ":" + fields["browser"],
        fields["brand"] + ":" + fields["country"],
        fields["channel"] + ":" + fields["merchant"],
        fields["bin"] + ":" + fields["os"],
        fields["browser"] + ":" + fields["merchant"],
    ]
    for i in range(min(16, len(v))):
        v[i] = _hash_to_float(combos[i], salts[i])

    amt = float(raw.amount)
    v[16] = np.log1p(max(amt, 0.0))
    v[17] = (amt % 1000) / 1000.0 * 2 - 1
    v[18] = _hash_to_float(fields["merchant"], "amt-bucket-" + str(int(amt // 50)))
    v[19] = _hash_to_float(fields["country"], "amt-sign-" + ("H" if amt > 500 else "L"))
    v[20] = _hash_to_float(fields["channel"], "ch")
    v[21] = _hash_to_float(fields["brand"], "br")
    v[22] = _hash_to_float(fields["bin"], "bn")
    v[23] = _hash_to_float(fields["last4"], "l4")
    v[24] = _hash_to_float(fields["os"], "os")
    v[25] = _hash_to_float(fields["browser"], "bw")
    v[26] = _hash_to_float(fields["merchant"], "mx")
    v[27] = _hash_to_float(fields["country"], "cx")

    features = [time_feat] + v + [amt]
    return np.array(features, dtype=float).reshape(1, -1)


@app.post("/score", response_model=PredictResponse)
def score_from_raw(req: RawTxnRequest):
    try:
        x = _engineer_demo_features(req)
        proba = getattr(model, "predict_proba", None)
        if proba is None:
            pred_label = int(model.predict(x)[0])
            return PredictResponse(is_fraud=bool(pred_label), probability_fraud=float(pred_label))
        prob = float(proba(x)[0][1])
        pred_label = prob >= 0.5
        return PredictResponse(is_fraud=bool(pred_label), probability_fraud=prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sample_fraud")
def sample_fraud():
    # Return a model-tailored high-risk feature vector (Time, V1..V28, Amount)
    return {"features": FRAUD_SAMPLE_FEATURES}
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


