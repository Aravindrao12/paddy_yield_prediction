from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
import pandas as pd 
import os 
import joblib
from typing import List,Dict,Any

model_dir = r"C:\Users\Infinix\Desktop\python.py\paddy.py\models"

pipeline = joblib.load(os.path.join(model_dir,"paddy_pipeline.pkl"))
input_columns:List[str]=joblib.load(os.path.join(model_dir, "input_columns.pkl"))

app=FastAPI(
    title="paddy yield prediction API",
    description="predict paddy yield using randomforestregression",
    version="1.0",
)
class PaddyInput(BaseModel):
    Hectares: int
    Agriblock: str
    Variety: str
    Soil_Types: str = Field(..., alias="Soil Types")
    Nursery: str
    Rain_30D: float = Field(..., alias="30DRain( in mm)")
    Rain_30_50D: float = Field(..., alias="30_50DRain( in mm)")
    Min_temp_D1_D30: float = Field(..., alias="Min temp_D1_D30")
    Max_temp_D1_D30: float = Field(..., alias="Max temp_D1_D30")
    Min_temp_D31_D60: float = Field(..., alias="Min temp_D31_D60")
    Max_temp_D31_D60: float = Field(..., alias="Max temp_D31_D60")
    Max_temp_D61_D90: float = Field(..., alias="Max temp_D61_D90")
    Min_temp_D91_D120: float = Field(..., alias="Min temp_D91_D120")
    Wind_Speed_D1_D30: float = Field(..., alias="Inst Wind Speed_D1_D30(in Knots)")
    Wind_Speed_D31_D60: float = Field(..., alias="Inst Wind Speed_D31_D60(in Knots)")
    Wind_Speed_D61_D90: float = Field(..., alias="Inst Wind Speed_D61_D90(in Knots)")
    Wind_Speed_D91_D120: float = Field(..., alias="Inst Wind Speed_D91_D120(in Knots)")
    Wind_Dir_D1_D30: str = Field(..., alias="Wind Direction_D1_D30")
    Wind_Dir_D31_D60: str = Field(..., alias="Wind Direction_D31_D60")
    Wind_Dir_D61_D90: str = Field(..., alias="Wind Direction_D61_D90")
    Wind_Dir_D91_D120: str = Field(..., alias="Wind Direction_D91_D120")
    Humidity_D1_D30: float = Field(..., alias="Relative Humidity_D1_D30")
    Humidity_D31_D60: float = Field(..., alias="Relative Humidity_D31_D60")
    Humidity_D61_D90: float = Field(..., alias="Relative Humidity_D61_D90")
    Humidity_D91_D120: float = Field(..., alias="Relative Humidity_D91_D120")


@app.get("/")
def home():
    return {"message": "Paddy Yield API is running"}

# @app.get("/options")
# def options():
#     prep = pipeline.named_steps["preprocessor"]
#     result = {}
#     for name, trans, cols in prep.transformers_:
#         if name == "cat":
#             for col, cats in zip(cols, trans.categories_):
#                 result[col] = [str(c) for c in cats]
#     return result


@app.post("/predict")
def predict(data: PaddyInput):
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])[input_columns]
    try:
        pred = pipeline.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predicted_yield_kg": round(float(pred), 2)}