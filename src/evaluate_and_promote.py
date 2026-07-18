import os
import mlflow
from mlflow.tracking import MlflowClient

def evaluate_and_promote():
    print("=== Starting Model Evaluation & Promotion ===")
    
    # 0. Inject MinIO credentials
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "admin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "password123")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    
    # 1. Connect to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    experiment_name = "Used_Car_Pricing"
    model_registry_name = "Used_Car_Pricing_Model"
    
    # 2. Get the Experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
    
    # 3. Filter by Airflow Pipeline Run ID
    pipeline_run_id = os.environ.get("PIPELINE_RUN_ID", "manual_local_run")
    print(f"Evaluating models from pipeline run: {pipeline_run_id}")
    
    search_filter = f"attributes.status = 'FINISHED' and tags.pipeline_run_id = '{pipeline_run_id}'"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=search_filter,
        order_by=["metrics.RMSE ASC"],
        max_results=10
    )
    
    if not runs:
        raise RuntimeError("No finished runs found for this pipeline execution.")
        
    # Variables to hold the winner
    winner_run = None
    artifact_path = None
    
    # 4. Loop to find and register the model
    print("\nScanning leaderboard for a valid champion...")
    for run in runs:
        run_id = run.info.run_id
        model_type = run.data.params.get("model_type", "Unknown")
        
        # Determine artifact name
        if model_type == "LightGBM":
            artifact_name = "challenger_lgbm_model"
        elif model_type == "XGBoost":
            artifact_name = "challenger_xgbm_model"
        else:
            artifact_name = "baseline_model"
            
        model_uri = f"runs:/{run_id}/{artifact_name}"
        
        try:
            # Register the model
            print(f"Attempting to register run {run_id} ({model_type})...")
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
            
            # Save winner info for promotion step
            winner_run = run
            artifact_path = artifact_name
            print(f"✨ Successfully registered {model_type}!")
            break 
        except Exception as e:
            print(f"⚠️ Could not register {artifact_name} from run {run_id}: {e}")

    # Final check: did we actually find a winner?
    if not winner_run:
        raise RuntimeError("Critical Error: No models could be registered.")
        
    # 5. Extract metrics and promote
    best_rmse = winner_run.data.metrics.get("RMSE", 0)
    best_r2 = winner_run.data.metrics.get("R2", 0)
    winner_type = winner_run.data.params.get("model_type", "Unknown")
    
    print(f"\n🏆 Final Confirmed Winner: {winner_type}")
    print(f"RMSE: {best_rmse:,.2f} | R²: {best_r2:.4f}")
    
    # Get the latest version of the model we just registered
    latest_version = client.get_latest_versions(model_registry_name, stages=["None"])[-1].version
    
    # 6. Assign the 'Champion' alias
    print(f"Promoting Version {latest_version} to 'Champion' alias!")
    client.set_registered_model_alias(
        name=model_registry_name, 
        alias="Champion", 
        version=latest_version
    )
    
    print("=== Evaluation & Promotion Complete ===")

if __name__ == "__main__":
    evaluate_and_promote()