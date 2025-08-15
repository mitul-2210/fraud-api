import argparse
import boto3
import datetime as dt
from pathlib import Path


def get_sklearn_inference_image(region: str, sklearn_version: str) -> str:
    # Use SageMaker SDK to retrieve the correct, permitted image URI
    try:
        from sagemaker import image_uris
        # Let SDK resolve a compatible version for inference
        return image_uris.retrieve(
            framework="sklearn",
            region=region,
            version=sklearn_version,
            image_scope="inference",
            instance_type="ml.t2.medium",
        )
    except Exception:
        # Fallback to common account mappings if SDK not present or version missing
        account_map = {
            "ap-south-1": "991648021394",
            "us-east-1": "683313688378",
            "us-east-2": "257758044811",
            "us-west-2": "334831119152",
            "eu-west-1": "985815980388",
        }
        account = account_map.get(region)
        if not account:
            raise RuntimeError(f"Unsupported region for hardcoded map: {region}")
        return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:{sklearn_version}-cpu-py3"


def main():
    parser = argparse.ArgumentParser(description="Deploy using SageMaker built-in scikit-learn container")
    parser.add_argument("--region", default=None)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="fraud-model")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--endpoint-name", default="fraud-api-endpoint")
    parser.add_argument("--sklearn-version", default="1.4-0", help="SageMaker sklearn container version (e.g., 1.4-0, 1.2-2, 1.0-1)")
    args = parser.parse_args()

    session = boto3.session.Session()
    region = args.region or session.region_name or "ap-south-1"
    s3 = boto3.client("s3", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)

    model_tar = Path("model.tar.gz")
    if not model_tar.exists():
        raise SystemExit("model.tar.gz not found. Run build_model_tar.py first.")

    key = f"{args.prefix}/model.tar.gz"
    s3.upload_file(str(model_tar), args.bucket, key)
    model_data_url = f"s3://{args.bucket}/{key}"
    print("Uploaded model archive to:", model_data_url)

    image_uri = get_sklearn_inference_image(region, args.sklearn_version)
    print("Using sklearn inference image:", image_uri)

    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_name = f"fraud-sklearn-model-{ts}"
    config_name = f"fraud-sklearn-config-{ts}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "Mode": "SingleModel",
            "ModelDataUrl": model_data_url,
            "Environment": {
                # Ensure the built-in sklearn container loads our custom handlers from model.tar.gz
                # We packaged the script at code/inference.py
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        },
        ExecutionRoleArn=args.role_arn,
    )
    print("Created model:", model_name)

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.t2.medium",
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    print("Created endpoint config:", config_name)

    sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
    print("Creating endpoint:", args.endpoint_name)
    print("Note: provisioning can take several minutes.")


if __name__ == "__main__":
    main()


