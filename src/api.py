from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
from typing import List
import json
import os

app = FastAPI(title="Churn Prediction API")

# Load model and preprocessor
try:
    with open('models/churn_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
    
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor_data = pickle.load(f)
        scaler = preprocessor_data['scaler']
        label_encoders = preprocessor_data['label_encoders']
    
    print("✅ Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str

# Store predictions for drift detection
predictions_log = []

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running!", "status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: Customer):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Encode categorical features SAFELY
        for col in df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except ValueError:
                    # If unseen label, use the most frequent value
                    df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Log prediction for drift detection
        prediction_record = customer_dict.copy()
        prediction_record['prediction'] = int(prediction)
        prediction_record['probability'] = float(probability)
        predictions_log.append(prediction_record)
        
        # Save predictions log
        os.makedirs('data/predictions', exist_ok=True)
        with open('data/predictions/predictions_log.json', 'w') as f:
            json.dump(predictions_log, f, indent=2)
        
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/stats")
def get_stats():
    if len(predictions_log) == 0:
        return {"message": "No predictions yet"}
    
    total_predictions = len(predictions_log)
    churn_count = sum(1 for p in predictions_log if p['prediction'] == 1)
    avg_probability = np.mean([p['probability'] for p in predictions_log])
    
    return {
        "total_predictions": total_predictions,
        "churn_predictions": churn_count,
        "churn_rate": churn_count / total_predictions,
        "average_churn_probability": float(avg_probability)
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "total_predictions": len(predictions_log)
    }
