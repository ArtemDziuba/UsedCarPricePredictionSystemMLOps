import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from api.schemas import CarPredictionRequest, CarPredictionResponse

# Add the src/ folder to Python's path so we can import your preprocessing functions!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing import preprocess_shared, preprocess_num_dataset

app = FastAPI(title="Used Car Price Prediction API", version="1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/baseline.pkl")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "../models/preprocessing_artifacts.pkl")

model = None
artifacts = None

@app.on_event("startup")
def load_assets():
    global model, artifacts
    try:
        model = joblib.load(MODEL_PATH)
        artifacts = joblib.load(ARTIFACTS_PATH)
        print("Model and preprocessing artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading assets: {e}")

@app.post("/predict", response_model=CarPredictionResponse)
def predict_price(request: CarPredictionRequest):
    if not model or not artifacts:
        raise HTTPException(status_code=500, detail="Model or artifacts not loaded.")

    # 1. Convert raw JSON request to a Pandas DataFrame
    raw_df = pd.DataFrame([request.model_dump()])  # .dict() is deprecated in newer Pydantic versions
    
    # 2. Run Shared Preprocessing (Fix dates, core categories, missing flags)
    df_shared = preprocess_shared(raw_df)
    
    # 3. Run Numerical Preprocessing (Caps categories strictly using our artifacts!)
    df_num = preprocess_num_dataset(df_shared, artifacts)
    
    # 4. Align the dataframe to exactly match the model's expected dummy columns
    input_aligned = df_num.reindex(columns=artifacts['expected_columns'], fill_value=0)
    
    # 5. Predict!
    prediction = np.expm1(model.predict(input_aligned)[0])
    
    # Prevent extreme negative baseline explosions
    prediction = max(-1.0, float(prediction))
    
    return CarPredictionResponse(predicted_price=round(prediction, 2))