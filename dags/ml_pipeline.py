from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default settings for all tasks in this DAG
default_args = {
    'owner': 'Artem',
    'depends_on_past': False,
    'start_date': datetime(2026, 7, 11),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
with DAG(
    dag_id='used_car_ml_pipeline',
    default_args=default_args,
    description='Executes the data preprocessing and model training pipeline',
    schedule_interval=None, # Set to None so we trigger it manually for now
    catchup=False,
) as dag:
    
    # NEW TASK: Upload to Object Storage
    upload_task = BashOperator(
        task_id='upload_to_storage',
        bash_command='cd /opt/airflow && python src/upload_to_storage.py'
    )

    # Task 1: Run the Preprocessing Script
    preprocess_task = BashOperator(
        task_id='run_preprocessing',
        # Notice the absolute path matching our docker-compose volume mounts
        bash_command='cd /opt/airflow && python src/preprocessing.py' 
    )

    # Task 2: Run the Baseline Training Script
    train_baseline_task = BashOperator(
        task_id='run_baseline_training',
        bash_command='cd /opt/airflow && python src/train.py'
    )

    # Define the dependency: Preprocessing MUST finish successfully before Training starts
    upload_task >> preprocess_task >> train_baseline_task