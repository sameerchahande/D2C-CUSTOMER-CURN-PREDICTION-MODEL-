from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional, Dict, Any
import uvicorn

MODEL_PATH = "churn_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
FEATURES_PATH = "model_features.pkl"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using a Random Forest model based on RFM and behavioral features.",
    version="1.0.0"
)
model = None
label_encoders = None
feature_cols = None
cat_cols = None

def load_artifacts():
    """Load trained model and preprocessing artifacts."""
    global model, label_encoders, feature_cols, cat_cols
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        meta = joblib.load(FEATURES_PATH)
        feature_cols = meta['feature_cols']
        cat_cols = meta['cat_cols']
        return True
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        print("Please run Train_Model.py first to train and save the model.")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup."""
    if not load_artifacts():
        print("WARNING: Model artifacts not found. API will not function correctly.")


class CustomerFeatures(BaseModel):
    city_tier: str = Field(..., description="City tier (Tier 1, Tier 2, Tier 3)")
    age_group: str = Field(..., description="Age group (18-24, 25-34, 35-44, 45+)")
    acquisition_channel: str = Field(..., description="Acquisition channel (Instagram, Marketplace, Influencer, Google Search, Organic, Referral)")
    loyalty_tier: str = Field(..., description="Loyalty tier (None, Silver, Gold, Platinum)")
    preferred_category: str = Field(..., description="Preferred product category")
    marketing_consent: str = Field(..., description="Marketing consent (Yes/No)")
    
    # Behavioral features
    recency_days: int = Field(0, ge=0, description="Days since last purchase")
    frequency_180d: int = Field(0, ge=0, description="Purchase frequency in last 180 days")
    monetary_180d: float = Field(0.0, ge=0, description="Total spend in last 180 days")
    return_rate_180d: float = Field(0.0, ge=0, le=1, description="Return rate in last 180 days")
    avg_discount_pct_180d: float = Field(0.0, ge=0, le=1, description="Average discount percentage")
    avg_rating_180d: float = Field(3.5, ge=1, le=5, description="Average rating given")
    category_diversity_180d: int = Field(0, ge=0, description="Number of unique categories purchased")
    ticket_count_90d: int = Field(0, ge=0, description="Support ticket count in last 90 days")
    negative_ticket_rate_90d: float = Field(0.0, ge=0, le=1, description="Rate of negative tickets")
    avg_resolution_hours_90d: float = Field(0.0, ge=0, description="Average resolution time in hours")
    days_since_signup: int = Field(0, ge=0, description="Days since customer signed up")
    sessions_30d: int = Field(0, ge=0, description="Number of sessions in last 30 days")
    product_views_30d: int = Field(0, ge=0, description="Product views in last 30 days")
    cart_adds_30d: int = Field(0, ge=0, description="Cart additions in last 30 days")
    wishlist_adds_30d: int = Field(0, ge=0, description="Wishlist additions in last 30 days")
    abandoned_carts_30d: int = Field(0, ge=0, description="Abandoned cart count in last 30 days")
    email_opens_30d: int = Field(0, ge=0, description="Email opens in last 30 days")
    campaign_clicks_30d: int = Field(0, ge=0, description="Campaign clicks in last 30 days")
    last_visit_days_ago: int = Field(0, ge=0, description="Days since last website/app visit")

class PredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

class SinglePredictionRequest(BaseModel):
    customer: CustomerFeatures

class PredictionResult(BaseModel):
    customer_index: int
    churn_probability: float
    churn_prediction: int 
    churn_label: str 

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    high_risk_count: int
    total_customers: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_features: Optional[int] = None


def preprocess_customer(customer: CustomerFeatures) -> np.ndarray:
    """Convert a single customer's features into the model input format."""
    if model is None or label_encoders is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    
    raw = {
        'city_tier': customer.city_tier,
        'age_group': customer.age_group,
        'acquisition_channel': customer.acquisition_channel,
        'loyalty_tier': customer.loyalty_tier,
        'preferred_category': customer.preferred_category,
        'marketing_consent': customer.marketing_consent,
        'recency_days': customer.recency_days,
        'frequency_180d': customer.frequency_180d,
        'monetary_180d': customer.monetary_180d,
        'return_rate_180d': customer.return_rate_180d,
        'avg_discount_pct_180d': customer.avg_discount_pct_180d,
        'avg_rating_180d': customer.avg_rating_180d,
        'category_diversity_180d': customer.category_diversity_180d,
        'ticket_count_90d': customer.ticket_count_90d,
        'negative_ticket_rate_90d': customer.negative_ticket_rate_90d,
        'avg_resolution_hours_90d': customer.avg_resolution_hours_90d,
        'days_since_signup': customer.days_since_signup,
        'sessions_30d': customer.sessions_30d,
        'product_views_30d': customer.product_views_30d,
        'cart_adds_30d': customer.cart_adds_30d,
        'wishlist_adds_30d': customer.wishlist_adds_30d,
        'abandoned_carts_30d': customer.abandoned_carts_30d,
        'email_opens_30d': customer.email_opens_30d,
        'campaign_clicks_30d': customer.campaign_clicks_30d,
        'last_visit_days_ago': customer.last_visit_days_ago
    }
    
    for col in cat_cols:
        try:
            raw[col] = label_encoders[col].transform([str(raw[col])])[0]
        except ValueError:
            raw[col] = 0
    
    features = np.array([raw[col] for col in feature_cols], dtype=np.float64).reshape(1, -1)
    return features


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API and model are ready."""
    model_loaded = model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        total_features=len(feature_cols) if feature_cols else None
    )

@app.post("/predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: PredictionRequest):
    """
    Predict churn probability for a batch of customers.
    
    Returns churn probability (0-1), binary prediction (0/1), and label for each customer.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    
    if not request.customers:
        raise HTTPException(status_code=400, detail="No customers provided.")
    
    predictions = []
    for i, customer in enumerate(request.customers):
        try:
            features = preprocess_customer(customer)
            probability = float(model.predict_proba(features)[0, 1])
            prediction = int(model.predict(features)[0])
            
            predictions.append(PredictionResult(
                customer_index=i,
                churn_probability=round(probability, 4),
                churn_prediction=prediction,
                churn_label="Churn" if prediction == 1 else "No Churn"
            ))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing customer {i}: {str(e)}")
    
    high_risk_count = sum(1 for p in predictions if p.churn_prediction == 1)
    
    return BatchPredictionResponse(
        predictions=predictions,
        high_risk_count=high_risk_count,
        total_customers=len(predictions)
    )

@app.post("/predict/single", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(request: SinglePredictionRequest):
    """
    Predict churn probability for a single customer.
    
    Returns churn probability (0-1), binary prediction (0/1), and label.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    
    features = preprocess_customer(request.customer)
    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(model.predict(features)[0])
    
    return PredictionResult(
        customer_index=0,
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        churn_label="Churn" if prediction == 1 else "No Churn"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)