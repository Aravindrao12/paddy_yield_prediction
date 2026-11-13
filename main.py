from typing import List,Dict,Any
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="paddy Yield prediction API")
model_dir = r"C:\Users\Infinix\Desktop\python.py\paddy.py\models"

pipeline = joblib.load(os.path.join(model_dir,"paddy_pipeline.pkl"))
input_columns:List[str]=joblib.load(os.path.join(model_dir, "input_columns.pkl"))

class PredictRequest(BaseModel):
    feature: Dict[str,Any]


@app.get("/")
def home():
    return {"message":"paddy Yield API is running","required_features":input_columns}

@app.post("/predict")
def predict(req:PredictRequest):
    missing=[c for c in input_columns if c not in req.feature]
    if missing :
        raise HTTPException(status_code=400,detail=f"missing feature :{missing}")
    df=pd.DataFrame([req.feature])[input_columns]
    try:
        pred=pipeline.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))
    return {"predicted_yield_kg":round(float(pred),2)}     