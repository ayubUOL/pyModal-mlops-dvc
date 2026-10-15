# app.py - a small web API that answers house-price questions using champion.pkl
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="House Price API")

# Load the model ONCE when the API starts (not on every request)
model = joblib.load("champion.pkl")

# The shape of a request. FastAPI rejects anything that doesn't match.
class House(BaseModel):
    rooms: int = Field(gt=0, description="Number of rooms")
    size_sqft: float = Field(gt=0, description="Size in square feet")

# Door 1: "are you alive?"
@app.get("/health")
def health():
    return {"status": "ok"}

# Door 2: the real job
@app.post("/predict")
def predict(house: House):
    data = pd.DataFrame([{"rooms": house.rooms, "size_sqft": house.size_sqft}])
    price = model.predict(data)[0]
    return {"price_lakh": round(float(price), 1)}