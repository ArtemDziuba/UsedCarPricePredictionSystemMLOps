import os
import numpy as np
import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
from utils import load_dataset, evaluate_model, print_metrics
import pandas as pd

def train_challenger():
    print("=== Starting Challenger Models Training ===")
    
    # 1. Connect to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Used_Car_Pricing")
    
    # 2. Point directly to the FULL dataset in MinIO
    data_path = "s3://ml-data/processed/full_dataset.parquet"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    columns_path = os.path.join(script_dir, "../models/challenger_expected_columns.pkl")
    
    # 3. Load the data
    X_train, X_test, y_train, y_test = load_dataset(data_path)
    
    # Shrink training data to 20% for local Docker memory safety
    X_train = X_train.sample(frac=0.2, random_state=42)
    y_train = y_train.loc[X_train.index]
    
    # Crucial for tree models: Convert object/string columns to 'category' dtype
    # Both LightGBM and XGBoost natively optimize category types!
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        # Convert to category first, then extract the underlying integer codes
        X_train[col] = X_train[col].astype('category').cat.codes
        
        # We must apply the EXACT same categories from train to test to prevent mismatches
        test_cat = pd.Categorical(X_test[col], categories=X_train[col].astype('category').cat.categories)
        X_test[col] = test_cat.codes
        
    # Save the expected columns for the API
    expected_columns = list(X_train.columns)
    joblib.dump(expected_columns, columns_path)
    
    # 4. Define the models in a dictionary
    models_to_train = {
        "Challenger_LGBM": {
            "model_class": LGBMRegressor,
            "params": {"n_estimators": 450, "max_depth": 12, "learning_rate": 0.08, "num_leaves": 129,
                       "subsample": 0.8, "colsample_bytree": 0.75,"random_state": 42},
            "name": "LightGBM"
        },
        "Challenger_XGBM": {
            "model_class": XGBRegressor,
            "params": {"n_estimators": 450, "max_depth": 10, "learning_rate": 0.08, "subsample": 0.75, 
                       "colsample_bytree": 0.9, "min_child_weight" :4, "random_state": 42, "enable_categorical": True},
            "name": "XGBoost"
        }
    }

    # 5. Train and log each model dynamically
    for run_name, config in models_to_train.items():
        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Training {config['name']} ---")

            pipeline_run_id = os.environ.get("PIPELINE_RUN_ID", "manual_local_run")
            mlflow.set_tag("pipeline_run_id", pipeline_run_id)
            
            mlflow.log_param("model_type", config['name'])
            mlflow.log_param("target_transformation", "log1p")
            mlflow.log_params(config['params'])
            
            # Initialize model with its specific parameters
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
            mlflow.log_artifact(columns_path, "metadata")
            mlflow.set_tag("Stage", "Challenger")
            
    print("\n=== Challenger Training Pipeline Finished ===")

if __name__ == "__main__":
    train_challenger()