import pandas as pd
import numpy as np
import os
import joblib
import boto3
import io

# ==========================================
# 1. UTILITIES & NA REPORTING
# ==========================================

FIX_STRATEGY = {
    'sellingprice': 'row dropped if missing (target)',
    'mmr': 'row dropped if missing',
    'sale_day': 'column dropped entirely',
    'COMPANY': "row dropped only if MODEL also missing, else filled 'unknown'",
    'MODEL': "row dropped only if COMPANY also missing, else filled 'unknown'",
    'TYPE': "filled 'unknown'",
    'SIZE': "hierarchical mode fill (COMP+MOD+TYPE -> COMP+MOD -> 'unknown')",
    'sale_month': "mapped to string abbreviation, missing='missing'",
    'sale_year': "filled with mode, flag added",
    'odometer': "row dropped if missing (too risky to impute)",
    'condition': "filled with median, flag added, binned into categories for num_dataset",
    'transmission': "filled 'unknown'",
    'state': "filled 'unknown'",
    'color': "filled 'unknown'",
    'interior_color': "filled 'unknown'",
    'seller': "filled 'unknown'",
}

def print_na_report(df, label):
    """Helper to log missing value percentages and actions."""
    counts = df.isna().sum()
    counts = counts[counts > 0]
    print(f"\n--- NA report: {label} ---")
    if counts.empty:
        print("No missing values.")
        return
    for col, n in counts.items():
        strategy = FIX_STRATEGY.get(col, "no explicit strategy defined")
        print(f"  {col}: {n} missing ({n/len(df):.1%})  -> {strategy}")


# ==========================================
# 2. TRAINING-ONLY PREPROCESSING
# ==========================================

def load_and_clean_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans targets, drops obvious outliers to stabilize models (Training only)."""
    df = df.dropna(subset=['sellingprice', 'mmr'])
    df = df[df['sellingprice'] >= 50]
    df = df.drop(columns=['sale_day'], errors='ignore')
    df = df.dropna(subset=['odometer'])
    return df


# ==========================================
# 3. SHARED PREPROCESSING (Train & Inference)
# ==========================================

def handle_core_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for core categorical columns."""
    df = df.dropna(subset=['COMPANY', 'MODEL'], how='all').copy()
    df['TYPE'] = df['TYPE'].fillna('unknown')

    # Hierarchical imputation for SIZE
    size_map_1 = df.dropna(subset=['SIZE']).groupby(['COMPANY', 'MODEL', 'TYPE'])['SIZE'].agg(
        lambda x: pd.Series.mode(x)[0] if not x.mode().empty else np.nan
    )
    size_map_2 = df.dropna(subset=['SIZE']).groupby(['COMPANY', 'MODEL'])['SIZE'].agg(
        lambda x: pd.Series.mode(x)[0] if not x.mode().empty else np.nan
    )

    df = df.set_index(['COMPANY', 'MODEL', 'TYPE'])
    df['SIZE'] = df['SIZE'].fillna(size_map_1)
    df = df.reset_index()

    df = df.set_index(['COMPANY', 'MODEL'])
    df['SIZE'] = df['SIZE'].fillna(size_map_2)
    df = df.reset_index()

    df['SIZE'] = df['SIZE'].fillna('unknown')
    df['COMPANY'] = df['COMPANY'].fillna('unknown')
    df['MODEL'] = df['MODEL'].fillna('unknown')

    return df

def handle_dates_and_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers dates and tracks numerical missingness."""
    df['sale_month_isMissing'] = df['sale_month'].isnull().astype(int)
    df['sale_year_isMissing'] = df['sale_year'].isnull().astype(int)
    df['condition_was_missing'] = df['condition'].isnull().astype(int)

    # Keep sale_month as a categorical string abbreviation (e.g., 'Jan')
    df['sale_month'] = df['sale_month'].fillna('missing').astype(str).str[:3]
    df['sale_year'] = df['sale_year'].fillna(df['sale_year'].mode().iloc[0])
    df['condition'] = df['condition'].fillna(df['condition'].median())

    categorical_columns = ['transmission', 'state', 'color', 'interior_color', 'seller']
    for col in categorical_columns:
        df[col] = df[col].fillna('unknown')

    return df

def preprocess_shared(df: pd.DataFrame) -> pd.DataFrame:
    """Master shared function. Both API and Training use this."""
    df = handle_core_categories(df)
    df = handle_dates_and_missing_flags(df)
    return df


# ==========================================
# 4. ARTIFACT GENERATION (Stateful Tracking)
# ==========================================

def generate_preprocessing_artifacts(df: pd.DataFrame) -> dict:
    """
    Learns the top categories from the training data and saves them into a dictionary.
    This ensures the API enforces the exact same caps during inference.
    """
    artifacts = {}
    df_temp = df.copy() # Temporary copy so we can mirror the hierarchical grouping

    # 1. COMPANY (Top 20)
    top_companies = df_temp['COMPANY'].value_counts().nlargest(20).index.tolist()
    artifacts['top_companies'] = top_companies
    df_temp['COMPANY'] = df_temp['COMPANY'].where(df_temp['COMPANY'].isin(top_companies), 'Other')

    # 2. MODEL (Top 10 per previously-capped COMPANY)
    top_models = set()
    for comp, group in df_temp.groupby('COMPANY'):
        for mod in group['MODEL'].value_counts().nlargest(10).index:
            top_models.add((comp, mod))
    artifacts['top_models'] = top_models
    
    # Cap Models dynamically for the TYPE calculation
    valid_model_idx = pd.MultiIndex.from_tuples(list(top_models), names=['COMPANY', 'MODEL'])
    curr_model_idx = pd.MultiIndex.from_arrays([df_temp['COMPANY'], df_temp['MODEL']])
    df_temp['MODEL'] = df_temp['MODEL'].where(curr_model_idx.isin(valid_model_idx), 'Other')

    # 3. TYPE (Top 5 per previously-capped COMPANY+MODEL)
    top_types = set()
    for (comp, mod), group in df_temp.groupby(['COMPANY', 'MODEL']):
        for typ in group['TYPE'].value_counts().nlargest(5).index:
            top_types.add((comp, mod, typ))
    artifacts['top_types'] = top_types

    # 4. Standard Top 20s
    artifacts['top_sizes'] = df_temp['SIZE'].value_counts().nlargest(20).index.tolist()
    artifacts['top_states'] = df_temp['state'].value_counts().nlargest(20).index.tolist()
    artifacts['top_colors'] = df_temp['color'].value_counts().nlargest(20).index.tolist()
    artifacts['top_interior_colors'] = df_temp['interior_color'].value_counts().nlargest(20).index.tolist()
    artifacts['top_sellers'] = df_temp['seller'].value_counts().nlargest(20).index.tolist()

    return artifacts


# ==========================================
# 5. DATASET-SPECIFIC PREPROCESSING
# ==========================================

def preprocess_full_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drops tracking flags so Tree models just get the raw values."""
    cols_to_remove = ['sale_month_isMissing', 'sale_year_isMissing']
    return df.drop(columns=cols_to_remove, errors="ignore")

def preprocess_num_dataset(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Stateless transformer. Applies the saved artifacts perfectly.
    The API uses this exactly as-is.
    """
    df_num = df.copy()

    # 1. COMPANY
    df_num['COMPANY'] = df_num['COMPANY'].where(df_num['COMPANY'].isin(artifacts['top_companies']), 'Other')

    # 2. MODEL (Vectorized hierarchical capping)
    valid_model_idx = pd.MultiIndex.from_tuples(list(artifacts['top_models']), names=['COMPANY', 'MODEL'])
    curr_model_idx = pd.MultiIndex.from_arrays([df_num['COMPANY'], df_num['MODEL']])
    df_num['MODEL'] = df_num['MODEL'].where(curr_model_idx.isin(valid_model_idx), 'Other')

    # 3. TYPE (Vectorized hierarchical capping)
    valid_type_idx = pd.MultiIndex.from_tuples(list(artifacts['top_types']), names=['COMPANY', 'MODEL', 'TYPE'])
    curr_type_idx = pd.MultiIndex.from_arrays([df_num['COMPANY'], df_num['MODEL'], df_num['TYPE']])
    df_num['TYPE'] = df_num['TYPE'].where(curr_type_idx.isin(valid_type_idx), 'Other')

    # 4. Other Standard Caps
    df_num['SIZE'] = df_num['SIZE'].where(df_num['SIZE'].isin(artifacts['top_sizes']), 'Other')
    df_num['state'] = df_num['state'].where(df_num['state'].isin(artifacts['top_states']), 'Other')
    df_num['color'] = df_num['color'].where(df_num['color'].isin(artifacts['top_colors']), 'Other')
    df_num['interior_color'] = df_num['interior_color'].where(df_num['interior_color'].isin(artifacts['top_interior_colors']), 'Other')
    df_num['seller'] = df_num['seller'].where(df_num['seller'].isin(artifacts['top_sellers']), 'Other')

    # 5. Bin Condition Score
    bins = [0, 10, 20, 30, 40, 50]
    labels = ['very_bad', 'bad', 'fair', 'good', 'excellent']
    df_num['condition_binned'] = pd.cut(df_num['condition'], bins=bins, labels=labels).astype(str)
    df_num.loc[df_num['condition_was_missing'] == 1, 'condition_binned'] = 'missing'

    # 6. Drop superseded columns
    cols_to_drop = ['condition', 'condition_was_missing', 'mmr']
    df_num = df_num.drop(columns=cols_to_drop, errors='ignore')

    # 7. Dummify
    df_num = pd.get_dummies(df_num, drop_first=True)

    return df_num


# ==========================================
# 6. ORCHESTRATION
# ==========================================

# 1. ADD 'storage_options=None' TO THE FUNCTION INPUTS
def create_datasets(input_path: str, output_dir: str, artifacts_dir: str):
    print("Loading data...")
    # Connect to MinIO using boto3
    s3_client = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='admin',
        aws_secret_access_key='password123'
    )
    
    # Download the file directly into pandas memory
    obj = s3_client.get_object(Bucket='ml-data', Key='raw/car_prices.csv')
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    print("Raw shape:", df.shape)
    print_na_report(df, "raw input")

    # Clean targets (Training Only)
    df = load_and_clean_targets(df)
    
    # Run Shared Preprocessing
    df_master = preprocess_shared(df)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # ------------------------------------
    # TRACK 1: FULL DATASET
    # ------------------------------------
    df_full = preprocess_full_dataset(df_master)
    df_full = df_full.drop(columns=['mmr'])
    full_path = os.path.join(output_dir, 'full_dataset.parquet')
    df_full.to_parquet(full_path, index=False)
    print(f"\nSaved {full_path} (Shape: {df_full.shape})")

    # ------------------------------------
    # TRACK 2: NUMERICAL DATASET & ARTIFACTS
    # ------------------------------------
    print("\nGenerating tracking artifacts and numerical dataset...")
    
    # A. Learn the rules from the training set
    artifacts = generate_preprocessing_artifacts(df_master)
    
    # B. Apply the rules to create the Baseline dataset
    df_num = preprocess_num_dataset(df_master, artifacts)
    
    # C. Save the exact dummy columns expected by the model into the artifacts
    artifacts['expected_columns'] = [c for c in df_num.columns if c != 'sellingprice']
    
    # D. Save Artifacts for the API
    artifacts_path = os.path.join(artifacts_dir, 'preprocessing_artifacts.pkl')
    joblib.dump(artifacts, artifacts_path)
    print(f"Saved preprocessing artifacts to {artifacts_path}")

    # E. Save final dataset
    num_path = os.path.join(output_dir, 'num_dataset.parquet')
    df_num.to_parquet(num_path, index=False)
    print(f"Saved {num_path} (Shape: {df_num.shape})")

    print("\nConnecting to MinIO to upload datasets...")
    
    # 1. Grab the credentials Airflow already injected into the environment
    minio_options = {
        "key": os.environ.get("AWS_ACCESS_KEY_ID", "admin"),
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", "password123"),
        "client_kwargs": {
            "endpoint_url": os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        }
    }

    # 2. Stream the files directly into the s3 bucket
    s3_full_dataset_path = "s3://ml-data/processed/full_dataset.parquet"
    print(f"Uploading full dataset to {s3_full_dataset_path}...")
    df_full.to_parquet(s3_full_dataset_path, storage_options=minio_options)

    s3_num_dataset_path = "s3://ml-data/processed/num_dataset.parquet"
    print(f"Uploading numerical dataset to {s3_num_dataset_path}...")
    df_num.to_parquet(s3_num_dataset_path, storage_options=minio_options)

    print("Uploading preprocessing artifacts to models/preprocessing_artifacts.pkl...")
    s3_client.upload_file(
        Filename=artifacts_path,                 # The local file you just saved 
        Bucket='models',                         # The destination bucket
        Key='preprocessing_artifacts.pkl'        # The file name inside the bucket
    )


    print("\nPipeline execution complete!")

if __name__ == "__main__":
    create_datasets(
        input_path="s3://ml-data/raw/car_prices.csv", # Used just as a tracking reference now
        output_dir="data/processed", 
        artifacts_dir="models"
    )