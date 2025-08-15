import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split


def get_feature_columns(dataframe: pd.DataFrame) -> List[str]:
    expected = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    missing = [c for c in expected if c not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. Columns present: {list(dataframe.columns)}"
        )
    return expected


def train_and_save(
    csv_path: Path,
    model_out: Path,
    feature_names_out: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    class_weight: str = "balanced_subsample",
):
    df = pd.read_csv(csv_path)

    feature_columns = get_feature_columns(df)
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in dataset")

    X = df[feature_columns].astype(float)
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=random_state, stratify=y.values
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    feature_names_out.write_text(json.dumps(feature_columns))

    print("Model saved to:", model_out)
    print("Feature names saved to:", feature_names_out)
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest fraud model and save artifacts")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--model-out", type=Path, default=Path("model.joblib"))
    parser.add_argument(
        "--feature-names-out", type=Path, default=Path("feature_names.json")
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced_subsample",
        choices=["balanced", "balanced_subsample"],
    )
    args = parser.parse_args()

    train_and_save(
        csv_path=args.data_path,
        model_out=args.model_out,
        feature_names_out=args.feature_names_out,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=args.class_weight,
    )


if __name__ == "__main__":
    main()
