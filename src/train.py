import os
from sklearn.linear_model import LinearRegression
from utils import load_dataset, save_model, evaluate_model, print_metrics
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

def train_baseline():
    """
    Executes the training pipeline for the Baseline Linear Regression model.
    """
    print("=== Starting Baseline Model Training ===")
    
    # 1. Point to the MLflow container and set the experiment
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Used_Car_Pricing")
    
    # 2. Define Bulletproof Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/processed/num_dataset.parquet")
    model_save_path = os.path.join(script_dir, "../models/baseline.pkl")
    columns_path = os.path.join(script_dir, "../models/expected_columns.pkl")
    
    # 3. Load and split the dataset
    X_train, X_test, y_train, y_test = load_dataset(data_path)

    X_train = X_train.sample(frac=0.1, random_state=42)
    y_train = y_train.loc[X_train.index]
    
    # MLFLOW RUN
    with mlflow.start_run(run_name="Baseline_LR"):
        
        # Log your model configuration
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("target_transformation", "log1p")
        
        # 4. Initialize the Model
        model = LinearRegression()
        
        # 5. Train (Fit) the Model
        print("\nFitting Linear Regression model to the training data...")
        model.fit(X_train, np.log1p(y_train))
        print("Training complete!")
        
        # 6. Evaluate the Model
        print("\nEvaluating model on the 20% test holdout...")
        metrics = evaluate_model(model, X_test, y_test, log_target=True)
        print_metrics(metrics)
        
        # Log every metric your evaluate_model function returns (R2, RMSE, etc.)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # 7. Save local artifacts (Keeps your current setup working)
        save_model(model, model_save_path)
        
        expected_columns = list(X_train.columns)
        joblib.dump(expected_columns, columns_path)
        
        # 8. Magically push everything to MLflow & MinIO!
        print("\nUploading artifacts to MLflow/MinIO...")
        
        # This saves the sklearn model automatically
        mlflow.sklearn.log_model(model, "baseline_model")
        
        # This saves your expected columns list into the MLflow UI as a text artifact
        mlflow.log_artifact(columns_path, "metadata")
        
        # Tag this run so it's easy to find
        mlflow.set_tag("Stage", "Baseline")
    
    print("=== Baseline Training Pipeline Finished ===")
    

if __name__ == "__main__":
    train_baseline()

