# Credit Card Fraud Detection – Cloud API

This project trains a Random Forest model on the classic credit-card fraud dataset and serves predictions via a FastAPI service suitable for AWS deployment.

## 1) Train the model locally

Prereqs: Python 3.10+ and the dataset CSV (typically named `creditcard.csv`).

```bash
pip install -r requirements.txt
python train_model.py --data-path path/to/creditcard.csv
```

Artifacts produced in the current folder:
- `model.joblib`
- `feature_names.json`

## 2) Run the API locally

```bash
uvicorn api:app --reload --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

Prediction request (list of 30 floats in the order `Time, V1..V28, Amount`):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, 0.217134702226017, 149.62]}'
```

## 3) Containerize

```bash
docker build -t fraud-api:latest .
docker run --rm -p 8000:8000 fraud-api:latest
```

## 4) Deploy on AWS (quick options)

- App Runner (simplest): Push image to ECR, create App Runner service from ECR. Set port `8000`, health check `/health`.
- ECS Fargate: Create ECR repo, push image, deploy a Fargate service behind an ALB. Health check `/health`.
- Lambda + API Gateway: Package as a container or add an ASGI adapter (e.g., Mangum) and deploy. Use `/invocations` for SageMaker-style or `/predict` for JSON.

### SageMaker (bring-your-own-container)

This repo is compatible with SageMaker real-time endpoints. The container now listens on port `8080` and exposes `/ping` and `/invocations` as expected by SageMaker.

Steps:
- Build and push the image to ECR (see below).
- Upload model artifacts (`model.joblib`, `feature_names.json`) to S3, e.g. `s3://<bucket>/fraud-model/`.
- Create a SageMaker Model that references the ECR image and sets `ModelDataUrl` to the S3 prefix.
- Create an Endpoint Config (choose instance type, e.g. `ml.t2.medium`).
- Create the Endpoint.

At runtime, SageMaker mounts the artifacts at `/opt/ml/model`. The API auto-loads from there.

### ECR push (example)

```bash
aws ecr create-repository --repository-name fraud-api
aws ecr get-login-password | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
docker tag fraud-api:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/fraud-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/fraud-api:latest
```

Then create an App Runner service or ECS Fargate task from that image.

For SageMaker, use the same ECR image when creating the SageMaker Model.

## Notes

- The API expects the model artifacts (`model.joblib`, `feature_names.json`) in the same directory as `api.py` at runtime.
- Column order must be `Time, V1..V28, Amount` and match the training set.
- Use `/ping` and `/invocations` if integrating with SageMaker runtime.


