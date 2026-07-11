from pydantic import BaseModel, Field
from typing import Optional

class CarPredictionRequest(BaseModel):
    # === STRICTLY REQUIRED FIELDS ===
    COMPANY: str = Field(..., example="Ford", description="Car manufacturer is required")
    MODEL: str = Field(..., example="F-150", description="Car model is required")
    TYPE: Optional[str] = "unknown"
    SIZE: Optional[str] = "unknown"
    transmission: Optional[str] = "unknown"
    color: Optional[str] = "unknown"
    interior_color: Optional[str] = "unknown"
    odometer: float = Field(..., example=55000.0, description="Odometer reading is required")
    condition: float = Field(..., example=35.0, description="Condition score (e.g., 1-50) is required")
    state: Optional[str] = "unknown"
    seller: Optional[str] = "unknown"
    sale_day: Optional[str] = "unknown"
    sale_month: Optional[str] = "unknown"
    sale_year: int = Field(..., example=2015, description="Year of sale is required")

class CarPredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"