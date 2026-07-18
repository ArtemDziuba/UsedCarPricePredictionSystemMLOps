import os
import sys
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from src.preprocessing import preprocess_shared, preprocess_num_dataset

class CarPricePredictor:
    def __init__(self, tracking_uri: str = "http://mlflow:5000", experiment_name: str = "Used_Car_Pricing"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.model = None
        self.artifacts = None
        
        # Configure MLflow client connection
        mlflow.set_tracking_uri(self.tracking_uri)

    def load_assets(self):
        """
        Dynamically fetches the 'Champion' model and its matching tracking artifacts 
        straight from the MLflow Registry.
        """
        print(f"Connecting to MLflow Tracking Server at {self.tracking_uri}...")
        
        try:
            # 1. Download the Champion model dynamically using the MLflow Model Registry URI
            # Syntax: 'models:/<model_name>@<alias>' or 'models:/<model_name>/<version>'
            model_uri = f"models:/Used_Car_Pricing_Model@Champion"
            print(f"Fetching active Champion model from: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # 2. Extract the run ID associated with this Champion model to get its matching metadata
            # This ensures your preprocessing columns always align perfectly with the loaded model version
            client = mlflow.tracking.MlflowClient()
            model_name = "Used_Car_Pricing_Model"
            model_version_details = client.get_model_version_by_alias(model_name, "Champion")
            run_id = model_version_details.run_id
            
            print(f"Downloading matching preprocessing rules from run: {run_id}")
            # Downloads the specific expected_columns artifact tied to this run
            local_artifact_dir = client.download_artifacts(run_id=run_id, path="metadata")
            
            # 3. Load the expected columns tracking sheet
            columns_path = os.path.join(local_artifact_dir, "expected_columns.pkl")
            self.expected_columns = joblib.load(columns_path)
            
            # 4. Fallback to local preprocessing artifacts for standard caps/imputations
            # (Alternatively, you can log the global preprocessing_artifacts.pkl to MLflow as well!)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            global_artifacts_path = os.path.join(script_dir, "../models/preprocessing_artifacts.pkl")
            self.artifacts = joblib.load(global_artifacts_path)
            
            print("Successfully loaded Champion model and matching features matrix rules.")
            
        except Exception as e:
            print(f"Failed to fetch assets dynamically from registry: {e}")
            print("Attempting fallback to local baseline model...")
            self._load_local_fallback()

    def _load_local_fallback(self):
        """Fallback mechanism if MLflow server is unreachable during startup."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fallback_model_path = os.path.join(script_dir, "../models/baseline.pkl")
        fallback_artifacts_path = os.path.join(script_dir, "../models/preprocessing_artifacts.pkl")
        
        if os.path.exists(fallback_model_path) and os.path.exists(fallback_artifacts_path):
            self.model = joblib.load(fallback_model_path)
            self.artifacts = joblib.load(fallback_artifacts_path)
            self.expected_columns = joblib.load(os.path.join(script_dir, "../models/expected_columns.pkl"))
            print("Loaded local fallback baseline assets successfully.")
        else:
            raise RuntimeError("Critical Error: No remote registry assets or local fallbacks available.")

    def predict(self, input_data: dict) -> float:
        """Processes raw incoming dictionary, aligns features, and executes inference."""
        if not self.model or not self.expected_columns:
            raise RuntimeError("Inference assets are uninitialized.")

        # 1. Convert incoming JSON request payload to DataFrame
        raw_df = pd.DataFrame([input_data])
        
        # 2. Standard Shared Preprocessing
        df_shared = preprocess_shared(raw_df)
        
        # 3. Numerical Imputation & Feature Caps
        df_num = preprocess_num_dataset(df_shared, self.artifacts)
        
        # 4. Model Agnostic Column Alignment
        # Dynamically matches whatever structural features the current Champion expects
        input_aligned = df_num.reindex(columns=self.expected_columns, fill_value=0)
        
        # 5. Inference execution (Assumes log1p target optimization)
        prediction = np.expm1(self.model.predict(input_aligned)[0])
        
        return max(-1.0, float(prediction))