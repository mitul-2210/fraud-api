import io
import json
import os
from typing import Any, Dict, List

import joblib
import numpy as np


def _load_model(model_dir: str):
    # Try common filenames
    candidates = [
        os.path.join(model_dir, "model.joblib"),
        os.path.join(model_dir, "model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"Model file not found in {model_dir} (expected one of: {candidates})")


def model_fn(model_dir: str):
    return _load_model(model_dir)


def input_fn(request_body: bytes, content_type: str = "application/json"):
    body = request_body if isinstance(request_body, (bytes, bytearray)) else str(request_body).encode()
    ct = (content_type or "application/json").lower()
    if "json" in ct:
        payload = json.loads(body.decode("utf-8"))
        if isinstance(payload, dict) and "features" in payload:
            arr = np.array(payload["features"], dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        elif isinstance(payload, list):
            arr = np.array(payload, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        else:
            raise ValueError("Invalid JSON: expected {'features':[...]} or [[...],[...]]")
    if "csv" in ct or "text/plain" in ct:
        text = body.decode("utf-8").strip()
        rows = [[float(x) for x in line.split(',')] for line in text.splitlines() if line]
        return np.array(rows, dtype=float)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    proba_fn = getattr(model, "predict_proba", None)
    if proba_fn is None:
        preds = model.predict(input_data).astype(int)
        probs = preds.astype(float)
    else:
        probs = proba_fn(input_data)[:, 1]
        preds = (probs >= 0.5).astype(int)
    return {"predictions": preds.tolist(), "probabilities": [float(p) for p in np.atleast_1d(probs)]}


def output_fn(prediction, accept: str = "application/json"):
    if "json" in (accept or "application/json").lower():
        return json.dumps(prediction), "application/json"
    # Fallback to CSV
    preds = prediction.get("predictions", [])
    probs = prediction.get("probabilities", [])
    lines = [f"{p},{q}" for p, q in zip(preds, probs)]
    return "\n".join(lines), "text/csv"


