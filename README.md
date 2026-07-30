# Used Car Price Prediction - End-to-End MLOps Pipeline
This repository contains a full end-to-end Machine Learning Operations (MLOps) pipeline for predicting used car prices. It demonstrates a production-ready architecture, moving from raw data processing and automated model training to model registry and live API serving.

## Architecture Stack
- Orchestration: Apache Airflow
- Experiment Tracking & Model Registry: MLflow
- Object Storage: MinIO (S3-compatible)
    - ml-data: Dataset raw and processed
    - models bucket: Custom preprocessing artifacts (schema & category mappings via boto3)
    - mlflow-artifacts bucket: Serialized model objects (mlflow.sklearn.log_model)
- Metadata Database: PostgreSQL
- Model Serving: FastAPI (Dockerized, Stateless)

## Data
Primary dataset: Vehicle Sales Cleaned. This dataset is sourced from Kaggle and contains a collection of information regarding the sales transactions of various vehicles.
- Link: https://www.kaggle.com/datasets/krishanukalita/vehicle-sales-cleaned

Columns:
- COMPANY
- MODEL
- TYPE
- SIZE
- transmission
- state
- condition
- odometer
- color
- interior_color
- seller
- mmr
- sellingprice
- sale_day
- sale_month
- sale_year.

## How to Run the Project
This project is fully containerized with Docker. You do not need to set up local Python environments.
### Step 1: Clone the repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### Step 2: Build and start the infrastructure
Run the following command to build the Docker images and start all services:
```bash
docker-compose up --build -d
```
*`Note:` Allow 1–2 minutes for PostgreSQL and Airflow webserver initialization).*

### Step 3: Trigger the ML Pipeline
1. Open the Airflow UI (link below).
2. Locate the used_car_ml_pipeline DAG.
3. Click "Trigger DAG" to execute the pipeline.
4. Once completed, the top-performing model will automatically be promoted and served via the API.

## Service Endpoints
- Apache Airflow (Orchestration): http://localhost:8080/
    - Default login: admin / admin

- MLflow (Tracking Server & Registry): http://localhost:5000/
    - View training metrics, parameters, run artifacts, and registered models.

- MinIO (Object Storage UI): http://localhost:9001/
    - Default login: admin / password123

- FastAPI (Live Model Serving): http://localhost:8000/docs#/default
    - Interactive Swagger UI to test the /predict endpoint.

## Pipeline Breakdown & Execution Steps
The automated workflow managed by Airflow (`used_car_ml_pipeline`) executes the following steps:
1. Data Processing: Cleans the raw used car dataset, handles numerical imputations, applies categorical encoding mappings, and outputs prepared parquet datasets.
2. Model Training (Baseline & Challengers):
    - Trains Baseline and Challenger models (LightGBM & XGBoost).
    - Custom Artifact Storage: Uploads preprocessing rules (expected columns and category mappings as .pkl files) directly to the MinIO models S3 bucket using boto3.
    - MLflow Tracking & Logging: Logs hyperparameters, evaluation metrics (RMSE, R²), metadata artifacts, and model binaries (mlflow.sklearn.log_model) directly to MLflow (s3://mlflow-artifacts/).
3. Evaluate & Promote (evaluate_and_promote.py):
    - Queries the MLflow Tracking Server for finished runs matching the active PIPELINE_RUN_ID.
    - Evaluates runs on test set metrics, ranking models by lowest RMSE.
    - Registers the top-performing model in the MLflow Model Registry under Used_Car_Pricing_Model.
    - Promotes the newly registered version to the @Champion alias.

## Model Serving & Inference Layer
The inference service (used_car_api) is completely decoupled from model training:
- Stateless Execution: Runs a lightweight FastAPI application using python:3.14-slim.
- Dynamic Model Resolution: On startup, the API initializes CarPricePredictor, which connects to the MLflow Model Registry at http://mlflow:5000 to fetch the model assigned the @Champion alias (models:/Used_Car_Pricing_Model@Champion).
- Feature Matrix Alignment: Pulls matching expected column schemas and category mapping dictionaries to ensure prediction inputs match the exact structure expected by the active Champion.
