import os
import numpy as np
import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
from utils import load_dataset, evaluate_model, print_metrics
import pandas as pd
import boto3

def train_challenger():
    print("=== Starting Challenger Models Training ===")
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Used_Car_Pricing")
    
    data_path = "s3://ml-data/processed/full_dataset.parquet"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    columns_path = os.path.join(script_dir, "../models/challenger_expected_columns.pkl")
    mappings_path = os.path.join(script_dir, "../models/category_mappings.pkl")
    
    X_train, X_test, y_train, y_test = load_dataset(data_path)
    
    X_train = X_train.sample(frac=0.2, random_state=42)
    y_train = y_train.loc[X_train.index]
    
    # 1. Save the expected columns schema BEFORE integer conversion
    expected_columns = list(X_train.columns)
    joblib.dump(expected_columns, columns_path)
    
    # 2. Extract mappings and apply integer encoding safely
    category_mappings = {}
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        # Convert and save the ordered categories
        X_train[col] = X_train[col].astype('category')
        category_mappings[col] = list(X_train[col].cat.categories)
        
        # Apply integer codes to train
        X_train[col] = X_train[col].cat.codes
        
        # Apply the exact same integer codes to test
        test_cat = pd.Categorical(X_test[col], categories=category_mappings[col])
        X_test[col] = test_cat.codes
        
    # Save the mappings for the API to use during inference
    joblib.dump(category_mappings, mappings_path)

    joblib.dump(expected_columns, columns_path)

    # --- NEW: UPLOAD TO MINIO MODELS BUCKET ---
    print("\nUploading challenger expected columns and mappings to MinIO 'models' bucket...")
    s3_client = boto3.client(
        's3',
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "admin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "password123")
    )
    s3_client.upload_file(
        Filename=columns_path, 
        Bucket='models', 
        Key='challenger_expected_columns.pkl'
    )
    s3_client.upload_file(
        Filename=mappings_path, 
        Bucket='models', 
        Key='category_mappings.pkl'
    )
    
    # Model parameters are tuned with optuna, all tuning in models-01.ipynb
    models_to_train = {
        "Challenger_LGBM": {
            "model_class": LGBMRegressor,
            "params": {"n_estimators": 4600, "max_depth": 11, "learning_rate": 0.05, "num_leaves": 133,
                       "min_child_samples" :7, "subsample": 0.7, "colsample_bytree": 0.75,
                           # "reg_alpha": 5.866342158074236e-05, "reg_lambda": 1.8106931104542093e-07,
                           "random_state": 42
                       },
            "name": "LightGBM"

        },
        "Challenger_XGBM": {
            "model_class": XGBRegressor,
            "params": {"n_estimators": 4600, "max_depth": 11, "learning_rate": 0.05, "subsample": 0.7, 
                       "colsample_bytree": 0.75, "max_leaves": 129, "min_child_weight" :4, "enable_categorical": True,
                           # "reg_alpha": 5.866342158074236e-05, "reg_lambda": 1.8106931104542093e-07,
                           "random_state": 42
                           },
            "name": "XGBoost"
        }
    }

    for run_name, config in models_to_train.items():
        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Training {config['name']} ---")

            pipeline_run_id = os.environ.get("PIPELINE_RUN_ID", "manual_local_run")
            mlflow.set_tag("pipeline_run_id", pipeline_run_id)
            
            mlflow.log_param("model_type", config['name'])
            mlflow.log_param("target_transformation", "log1p")
            mlflow.log_params(config['params'])
            
            model = config['model_class'](**config['params'])
            
            print(f"Fitting {config['name']} to the training data...")
            model.fit(X_train, np.log1p(y_train))
            
            print("Evaluating...")
            metrics = evaluate_model(model, X_test, y_test, log_target=True)
            print_metrics(metrics)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                
            print(f"Uploading artifacts for {config['name']}...")
            mlflow.sklearn.log_model(
                model, 
                f"{run_name.lower()}_model",
                serialization_format="pickle"
            )
            
            # Log BOTH artifacts directly into this specific MLflow run
            mlflow.log_artifact(columns_path, "metadata")
            mlflow.log_artifact(mappings_path, "metadata")
            mlflow.set_tag("Stage", "Challenger")
            
    print("\n=== Challenger Training Pipeline Finished ===")

if __name__ == "__main__":
    train_challenger()