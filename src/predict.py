import os
import io
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import boto3
import threading
import time
from src.preprocessing import preprocess_shared, preprocess_num_dataset, preprocess_full_dataset

class CarPricePredictor:
    def __init__(self, tracking_uri: str = "http://mlflow:5000", experiment_name: str = "Used_Car_Pricing"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Core assets
        self.model = None
        self.artifacts = None
        self.expected_columns = None
        self.category_mappings = None
        
        # State tracking
        self.model_used = "uninitialized"
        self.model_type = "LinearRegression"
        self.current_run_id = None
        
        # Concurrency control to prevent API requests from failing during a reload
        self._lock = threading.Lock()
        
        mlflow.set_tracking_uri(self.tracking_uri)

    def load_assets(self):
        """Loads or reloads the Champion model from MLflow and MinIO."""
        print(f"Connecting to MLflow Tracking Server at {self.tracking_uri}...")
        
        try:
            client = mlflow.tracking.MlflowClient()
            model_name = "Used_Car_Pricing_Model"
            model_version_details = client.get_model_version_by_alias(model_name, "Champion")
            run_id = model_version_details.run_id
            
            # If we are already running this run_id, do nothing
            if run_id == self.current_run_id:
                return

            print(f"Loading new Champion assets for Run ID: {run_id}...")
            
            # 1. Download Model to a temporary variable first (to not break current predictions)
            model_uri = f"models:/{model_name}@Champion"
            new_model = mlflow.pyfunc.load_model(model_uri)
            
            # Fetch metadata
            run = client.get_run(run_id)
            new_model_type = run.data.params.get("model_type", "LinearRegression")

            # 2. Connect to MinIO
            s3_client = boto3.client(
                's3',
                endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "admin"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "password123")
            )

            # 3. Download Global Artifacts
            art_obj = s3_client.get_object(Bucket='models', Key='preprocessing_artifacts.pkl')
            new_artifacts = joblib.load(io.BytesIO(art_obj['Body'].read()))

            # 4. Download Specific Routing Files
            new_category_mappings = None
            if new_model_type in ["LightGBM", "XGBoost"]:
                columns_key = "challenger_expected_columns.pkl"
                map_obj = s3_client.get_object(Bucket='models', Key='category_mappings.pkl')
                new_category_mappings = joblib.load(io.BytesIO(map_obj['Body'].read()))
            else:
                columns_key = "baseline_expected_columns.pkl"
                
            col_obj = s3_client.get_object(Bucket='models', Key=columns_key)
            new_expected_columns = joblib.load(io.BytesIO(col_obj['Body'].read()))

            # --- ATOMIC SWAP ---
            # We use a lock so that if a prediction request comes in right at this millisecond, 
            # it waits safely rather than crashing on half-updated variables.
            with self._lock:
                self.model = new_model
                self.artifacts = new_artifacts
                self.expected_columns = new_expected_columns
                self.category_mappings = new_category_mappings
                
                self.model_type = new_model_type
                self.current_run_id = run_id
                self.model_used = f"{model_name}@Champion (Run ID: {run_id})"
                
            print(f"Successfully loaded new Champion: {self.model_type}")
            
        except Exception as e:
            if self.model is None:
                print(f"Initial load failed: {e}. Attempting fallback...")
                self._load_local_fallback()
            else:
                print(f"Background reload failed: {e}. Keeping current model active.")

    def _load_local_fallback(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fallback_model_path = os.path.join(script_dir, "../models/baseline.pkl")
        fallback_artifacts_path = os.path.join(script_dir, "../models/preprocessing_artifacts.pkl")
        fallback_columns_path = os.path.join(script_dir, "../models/baseline_expected_columns.pkl")

        if os.path.exists(fallback_model_path) and os.path.exists(fallback_artifacts_path):
            with self._lock:
                self.model = joblib.load(fallback_model_path)
                self.artifacts = joblib.load(fallback_artifacts_path)
                self.expected_columns = joblib.load(fallback_columns_path)
                
                self.model_used = "local_baseline.pkl"
                self.model_type = "LinearRegression"
                self.current_run_id = "local_fallback"
                
            print("Loaded local fallback baseline assets successfully.")
        else:
            raise RuntimeError("Critical Error: No remote registry assets or local fallbacks available.")

    def _poll_registry(self, interval_seconds=60):
        """Background thread worker that checks MLflow every 60 seconds."""
        while True:
            time.sleep(interval_seconds)
            try:
                # Lightweight check to see if the Champion alias moved
                client = mlflow.tracking.MlflowClient()
                model_version_details = client.get_model_version_by_alias("Used_Car_Pricing_Model", "Champion")
                latest_run_id = model_version_details.run_id
                
                if latest_run_id != self.current_run_id:
                    print(f"🔄 Background Thread: New Champion detected! (Run ID: {latest_run_id})")
                    self.load_assets()
            except Exception as e:
                # Fail silently so the thread doesn't die if MLflow drops for a minute
                pass

    def start_background_polling(self, interval_seconds=60):
        """Starts the daemon thread to check for model updates."""
        print(f"Starting MLflow polling thread (every {interval_seconds}s)...")
        polling_thread = threading.Thread(target=self._poll_registry, args=(interval_seconds,), daemon=True)
        polling_thread.start()

    def predict(self, input_data: dict) -> float:
        # Use the lock to ensure we don't try to predict EXACTLY while a swap is occurring
        with self._lock:
            if not self.model or not self.expected_columns:
                raise RuntimeError("Inference assets are uninitialized.")

            raw_df = pd.DataFrame([input_data])
            df_shared = preprocess_shared(raw_df)
            
            if self.model_type in ["LightGBM", "XGBoost"]:
                df_processed = preprocess_full_dataset(df_shared)
                if self.category_mappings:
                    for col, cats in self.category_mappings.items():
                        if col in df_processed.columns:
                            df_processed[col] = pd.Categorical(df_processed[col], categories=cats).codes
            else:
                df_processed = preprocess_num_dataset(df_shared, self.artifacts)
                
            input_aligned = df_processed.reindex(columns=self.expected_columns, fill_value=0)
            prediction = np.expm1(self.model.predict(input_aligned)[0])
            
            return max(-1.0, float(prediction))