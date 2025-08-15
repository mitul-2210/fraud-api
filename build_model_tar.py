import tarfile
from pathlib import Path


def build_model_tar(model_path: Path, out_path: Path, inference_script: Path | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        # Place model at the tar root as model.joblib (expected by SageMaker sklearn container)
        tar.add(model_path, arcname="model.joblib")
        # Also include a copy as model.pkl for maximum compatibility
        tar.add(model_path, arcname="model.pkl")
        # If a custom inference script is provided, place it under code/
        if inference_script and inference_script.exists():
            tar.add(inference_script, arcname="code/inference.py")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    model = here / "model.joblib"
    out = here / "model.tar.gz"
    inf = here / "inference.py"
    if not model.exists():
        raise SystemExit(f"Missing model file: {model}")
    build_model_tar(model, out, inference_script=inf if inf.exists() else None)
    print(f"Wrote {out}")


