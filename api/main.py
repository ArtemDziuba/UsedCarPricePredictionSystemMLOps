import os
import sys
from fastapi import FastAPI, HTTPException

# Add the root directory to path so we can import from api and src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.schemas import CarPredictionRequest, CarPredictionResponse
from src.predict import CarPricePredictor

app = FastAPI(title="Used Car Price Prediction API", version="1.0")

# Initialize our new predictor class
predictor = CarPricePredictor()

@app.on_event("startup")
def load_assets():
    try:
        predictor.load_assets()
    except Exception as e:
        print(f"CRITICAL ERROR loading prediction assets: {e}")

@app.post("/predict", response_model=CarPredictionResponse)
def predict_price(request: CarPredictionRequest):
    try:
        # Pydantic v2 uses model_dump() instead of dict()
        prediction = predictor.predict(request.model_dump())
        return CarPredictionResponse(predicted_price=round(prediction, 2))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")