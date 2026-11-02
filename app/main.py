
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer will churn",
    version="1.0"
)

MODEL_PATH = "models/churn_model.joblib"

model = None

class CustomerData(BaseModel):
    data: dict[str, Any]

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Trained model file not found")
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded"
            )

        input_df = pd.DataFrame([customer.data])

        expected_columns = model.feature_names_in_
        missing_columns = [
            col for col in expected_columns
            if col not in input_df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing fields: {missing_columns}"
            )

        input_df = input_df[expected_columns]

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        churn_probability = float(
            probabilities[list(classes).index("Churn")]
        )

        return {
            "prediction": str(prediction),
            "churn_probability": round(churn_probability, 4)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
