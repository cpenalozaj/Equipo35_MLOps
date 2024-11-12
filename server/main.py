# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
import joblib
import pandas as pd
from fastapi.encoders import jsonable_encoder

from models.student import StudentData

model = joblib.load('logistic_regression.pkl')
    
# Define the input data format for prediction
class PredictionInput(BaseModel):
    features: StudentData


# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(student_data: PredictionInput):
    """
    Predict the performance of a student based on the input data

    Args:
        student_data (PredictionInput): The input data for prediction
    
    Returns:
        dict: The prediction result
    """


    label_to_text = ["Best","Vg","Good","Pass","Fail"]
    

    features_df = pd.DataFrame(jsonable_encoder(student_data.features), index=[0])
    features_df.rename(columns={"as_": "as"}, inplace=True) # used as_ to avoid reserved keyword, so renaming back to as

    predictions = model.predict(features_df)

    return {"prediction": label_to_text[predictions[0]]}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Inference API running"}