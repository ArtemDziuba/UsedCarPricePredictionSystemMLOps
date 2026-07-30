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
        # 1. Load the initial model so the API is ready immediately
        predictor.load_assets()
        
        # 2. Start the daemon thread to watch for MLflow updates in the background
        predictor.start_background_polling(interval_seconds=60)
        
    except Exception as e:
        print(f"CRITICAL ERROR loading prediction assets: {e}")

@app.post("/predict", response_model=CarPredictionResponse)
def predict_price(request: CarPredictionRequest):
    try:
        # Pydantic v2 uses model_dump() instead of dict()
        prediction = predictor.predict(request.model_dump())

        return CarPredictionResponse(
            predicted_price=round(prediction, 2),
            model_used=predictor.model_used
        )
        return CarPredictionResponse(predicted_price=round(prediction, 2))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")