import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_dataset(file_path: str, target_col: str = 'sellingprice', test_size: float = 0.2, random_state: int = 42):
    """
    Loads a dataset (Parquet or CSV) and splits it 80/20 into train and test sets.
    
    Args:
        file_path (str): Path to the processed dataset.
        target_col (str): The name of the target variable to predict.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducible shuffles.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"Loading dataset from {file_path}...")
    
    # Support both formats just in case you ever switch back
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
        
    # Separate Features (X) and Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Perform the 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Split complete: {len(X_train)} training rows, {len(X_test)} testing rows.")
    return X_train, X_test, y_train, y_test

def save_model(model, filepath: str):
    """
    Saves the trained model to disk using joblib.
    """
    # Ensure the target directory exists (e.g., 'models/')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"Model successfully saved to: {filepath}")

def load_model(filepath: str):
    """
    Loads a trained model from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    
    print(f"Loading model from {filepath}...")
    return joblib.load(filepath)

def evaluate_model(model, X_test, y_test, log_target=False):
    """
    Generates predictions and calculates RMSE, MAE and R².

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Ground truth in the original scale.
        log_target: Whether the model predicts log1p(target).

    Returns:
        dict: RMSE, MAE and R².
    """
    y_pred = model.predict(X_test)

    if log_target:
        y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 0, None)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def print_metrics(metrics: dict):
    """
    Prints the evaluation metrics in a clean, readable format.
    """
    print("\n" + "="*32)
    print("   MODEL EVALUATION METRICS")
    print("="*32)
    for metric_name, value in metrics.items():
        # Formats the numbers with commas for thousands and 2 decimal places
        print(f"{metric_name:<6}: {value:,.2f}")
    print("="*32 + "\n")