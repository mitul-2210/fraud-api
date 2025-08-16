from __future__ import annotations

import json
from pathlib import Path
import os
import logging
from typing import List, Optional
import hashlib
from datetime import datetime
import random
import boto3
import pandas as pd

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
FEATURE_PIPELINE_PATH = MODEL_DIR / "feature_pipeline.joblib"
RAW_PIPELINE_PATH = MODEL_DIR / "raw_pipeline.joblib"
RAW_SCHEMA_PATH = MODEL_DIR / "raw_schema.json"

# Configurable prediction threshold and CORS origins
THRESHOLD: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))


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
        "raw_pipeline.joblib": RAW_PIPELINE_PATH,
        "raw_schema.json": RAW_SCHEMA_PATH,
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
    feature_pipeline = None
    raw_pipeline = None
    try:
        if FEATURE_PIPELINE_PATH.exists():
            feature_pipeline = joblib.load(FEATURE_PIPELINE_PATH)
        if RAW_PIPELINE_PATH.exists():
            raw_pipeline = joblib.load(RAW_PIPELINE_PATH)
    except Exception:
        feature_pipeline = feature_pipeline or None
        raw_pipeline = raw_pipeline or None
    return model, feature_names, feature_pipeline, raw_pipeline


model, feature_names, feature_pipeline, raw_pipeline = load_artifacts()

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

FRAUD_SAMPLES_PATH = APP_DIR / "fraud_samples.json"
try:
    FRAUD_SAMPLES = json.loads(FRAUD_SAMPLES_PATH.read_text()) if FRAUD_SAMPLES_PATH.exists() else None
except Exception:
    FRAUD_SAMPLES = None
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")
cors_origins = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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
        pred_label = prob >= THRESHOLD
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
            preds = [int(p >= THRESHOLD) for p in probs]
        return {
            "predictions": preds,
            "probabilities": probs,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metadata")
def metadata():
    return {
        "model_path": str(MODEL_PATH),
        "feature_names_path": str(FEATURE_NAMES_PATH),
        "feature_pipeline_present": FEATURE_PIPELINE_PATH.exists(),
        "raw_pipeline_present": RAW_PIPELINE_PATH.exists(),
        "raw_schema_present": RAW_SCHEMA_PATH.exists(),
        "threshold": THRESHOLD,
        "cors_origins": [o for o in (os.getenv("CORS_ORIGINS", "*").split(",")) if o],
        "version": "1.0.0",
    }


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
        # Preferred path: trained raw pipeline (production-like)
        if raw_pipeline is not None:
            payload = {
                "amount": float(req.amount),
                "time_minute": float((req.timestamp or datetime.utcnow().timestamp()) % 3600) / 60.0,
                "country": (req.country or "").upper(),
                "channel": (req.channel or "ecom").lower(),
                "merchant_id": req.merchant_id or "m_demo",
                "card_brand": (req.card_brand or "unknown").lower(),
                "card_bin": req.card_bin or "",
                "last4": req.last4 or "",
                "device_os": (req.device_os or "").lower(),
                "device_browser": (req.device_browser or "").lower(),
            }
            df = pd.DataFrame([payload])
            proba = getattr(raw_pipeline, "predict_proba", None)
            if proba is None:
                pred_label = int(raw_pipeline.predict(df)[0])
                return PredictResponse(is_fraud=bool(pred_label), probability_fraud=float(pred_label))
            prob = float(proba(df)[0][1])
            pred_label = prob >= THRESHOLD
            return PredictResponse(is_fraud=bool(pred_label), probability_fraud=prob)
        # Fallback: demo transformer to 30 features + model
        x = _engineer_demo_features(req)
        proba = getattr(model, "predict_proba", None)
        if proba is None:
            pred_label = int(model.predict(x)[0])
            return PredictResponse(is_fraud=bool(pred_label), probability_fraud=float(pred_label))
        prob = float(proba(x)[0][1])
        pred_label = prob >= THRESHOLD
        return PredictResponse(is_fraud=bool(pred_label), probability_fraud=prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sample_fraud")
def sample_fraud():
    # Return a random fraud-like vector. Prefer curated samples file if present; else generate.
    if FRAUD_SAMPLES and isinstance(FRAUD_SAMPLES, list):
        return {"features": random.choice(FRAUD_SAMPLES)}
    # Fallback: generate a high-risk sample on the fly with a random seed
    features = _generate_high_risk_sample(model, num=5000, seed=random.randint(1, 10_000_000))
    return {"features": features}
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


