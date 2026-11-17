from pydantic import BaseModel, Field
from fastapi import FastAPI
from datetime import datetime
from src.model import load_model
from contextlib import asynccontextmanager
from src.preprocessing import convert_hour_weekday
import pandas as pd
from src.preprocessing import FEATURES


class TripRequest(BaseModel):
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    trip_distance: float = Field(gt=0, le=100)
    tpep_pickup_datetime: datetime


models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["predict_duration"] = load_model()
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health_check():
    return ("Welcome to the server!")


@app.post("/predict")
def predict_duration(triprequest: TripRequest):
    triprequest_dict = triprequest.model_dump()
    triprequest_df = pd.DataFrame([triprequest_dict])
    triprequest_df = convert_hour_weekday(triprequest_df)
    features = triprequest_df[FEATURES]
    prediction = models["predict_duration"].predict(features)
    return {"predicted_duration_minutes": float(prediction[0])}
