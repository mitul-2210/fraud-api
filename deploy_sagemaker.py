import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional

import boto3


def upload_artifacts(s3_client, bucket: str, prefix: str, model_path: Path, features_path: Path) -> str:
    s3_client.upload_file(str(model_path), bucket, f"{prefix}/model.joblib")
    s3_client.upload_file(str(features_path), bucket, f"{prefix}/feature_names.json")
    return f"s3://{bucket}/{prefix}/"


def create_model(sm_client, name: str, image_uri: str, model_data_url: str, role_arn: str):
    return sm_client.create_model(
        ModelName=name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            # Pass through any env if needed
            "Environment": {},
        },
        ExecutionRoleArn=role_arn,
    )


def create_endpoint_config(sm_client, name: str, model_name: str, instance_type: str = "ml.t2.medium", initial_instances: int = 1):
    return sm_client.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": initial_instances,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
        DataCaptureConfig={
            "EnableCapture": False
        },
    )


def create_endpoint(sm_client, name: str, config_name: str):
    return sm_client.create_endpoint(EndpointName=name, EndpointConfigName=config_name)


def main():
    parser = argparse.ArgumentParser(description="Deploy the fraud API container to SageMaker real-time endpoint")
    parser.add_argument("--ecr-image", required=True, help="ECR image URI, e.g. 123.dkr.ecr.us-east-1.amazonaws.com/fraud-api:latest")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket for model artifacts")
    parser.add_argument("--s3-prefix", default="fraud-model", help="S3 key prefix for artifacts")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN for SageMaker execution")
    parser.add_argument("--model-path", type=Path, default=Path("model.joblib"))
    parser.add_argument("--features-path", type=Path, default=Path("feature_names.json"))
    parser.add_argument("--instance-type", default="ml.t2.medium")
    parser.add_argument("--endpoint-name", default=None)
    args = parser.parse_args()

    s3 = boto3.client("s3")
    sm = boto3.client("sagemaker")

    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_name = f"fraud-api-model-{ts}"
    endpoint_config_name = f"fraud-api-config-{ts}"
    endpoint_name = args.endpoint_name or f"fraud-api-endpoint"

    model_data_url = upload_artifacts(s3, args.s3_bucket, args.s3_prefix, args.model_path, args.features_path)
    print("Uploaded artifacts to:", model_data_url)

    create_model(sm, model_name, args.ecr_image, model_data_url, args.role_arn)
    print("Created model:", model_name)

    create_endpoint_config(sm, endpoint_config_name, model_name, args.instance_type)
    print("Created endpoint config:", endpoint_config_name)

    create_endpoint(sm, endpoint_name, endpoint_config_name)
    print("Creating endpoint:", endpoint_name)
    print("Note: endpoint creation can take several minutes. Monitor status in the SageMaker console.")


if __name__ == "__main__":
    main()


