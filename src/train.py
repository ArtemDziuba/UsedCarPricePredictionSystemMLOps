import os
from sklearn.linear_model import LinearRegression
from utils import load_dataset, save_model, evaluate_model, print_metrics
import joblib
import numpy as np

def train_baseline():
    """
    Executes the training pipeline for the Baseline Linear Regression model.
    """
    print("=== Starting Baseline Model Training ===")
    
    # 1. Define Bulletproof Paths
    # This finds the exact folder where train.py is located (the 'src' folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Now we build the absolute paths relative to the src folder
    data_path = os.path.join(script_dir, "../data/processed/num_dataset.parquet")
    model_save_path = os.path.join(script_dir, "../models/baseline.pkl")
    
    # 2. Load and split the dataset
    X_train, X_test, y_train, y_test = load_dataset(data_path)
    
    # 3. Initialize the Model
    model = LinearRegression()
    
    # 4. Train (Fit) the Model
    print("\nFitting Linear Regression model to the training data...")
    model.fit(X_train, np.log1p(y_train))
    print("Training complete!")
    
    # 5. Evaluate the Model
    print("\nEvaluating model on the 20% test holdout...")
    metrics = evaluate_model(model, X_test, y_test, log_target=True)
    print_metrics(metrics)
    
    # 6. Save the Model Artifact
    save_model(model, model_save_path)
    
    print("=== Baseline Training Pipeline Finished ===")

    # Save the expected column names so the API knows exactly how to format incoming data
    expected_columns = list(X_train.columns)
    columns_path = os.path.join(script_dir, "../models/expected_columns.pkl")
    joblib.dump(expected_columns, columns_path)
    

if __name__ == "__main__":
    train_baseline()

