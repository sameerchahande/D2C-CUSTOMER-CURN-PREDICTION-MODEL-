# Customer Churn Prediction API

A FastAPI-based REST API that predicts customer churn using a Random Forest model trained on RFM (Recency, Frequency, Monetary) and behavioral features.

## Project Structure

```
├── main.py          # FastAPI application with prediction endpoints
├── Train_Model.py   # Script to train and save the Random Forest model
├── Test_Api         # Test script for API endpoints
├── Requirement.txt  # Python dependencies
├── Readme.md        # This file
```

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -r Requirement.txt
```

### 2. Train the Model

```bash
python Train_Model.py
```

This will:
- Load the customer dataset from `rfm_modeling_snapshot.csv`
- Train a Random Forest classifier
- Save the model as `churn_model.pkl`
- Save label encoders as `label_encoders.pkl`
- Save feature metadata as `model_features.pkl`

### 3. Start the API Server

```bash
uvicorn main:app --reload
```

The API will be available at: `http://localhost:8000`

### 4. Test the API

```bash
python Test_Api
```

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Check if the API and model are loaded.

### `POST /predict`
Predict churn for a batch of customers.

**Request body:**
```json
{
  "customers": [
    {
      "city_tier": "Tier 1",
      "age_group": "25-34",
      "acquisition_channel": "Instagram",
      "loyalty_tier": "Silver",
      "preferred_category": "Makeup",
      "marketing_consent": "Yes",
      "recency_days": 107,
      "frequency_180d": 1,
      "monetary_180d": 362.73,
      "return_rate_180d": 0.0,
      "avg_discount_pct_180d": 0.23,
      "avg_rating_180d": 3.0,
      "category_diversity_180d": 1,
      "ticket_count_90d": 0,
      "negative_ticket_rate_90d": 0.0,
      "avg_resolution_hours_90d": 0.0,
      "days_since_signup": 524,
      "sessions_30d": 1,
      "product_views_30d": 4,
      "cart_adds_30d": 0,
      "wishlist_adds_30d": 0,
      "abandoned_carts_30d": 0,
      "email_opens_30d": 2,
      "campaign_clicks_30d": 0,
      "last_visit_days_ago": 20
    }
  ]
}
```

### `POST /predict/single`
Predict churn for a single customer.

**Request body:** Same as above but wrapped in `{"customer": {...}}`

## Features Used

### Categorical Features
- city_tier, age_group, acquisition_channel, loyalty_tier, preferred_category, marketing_consent

### Numerical Features (RFM + Behavioral)
- recency_days, frequency_180d, monetary_180d, return_rate_180d
- avg_discount_pct_180d, avg_rating_180d, category_diversity_180d
- ticket_count_90d, negative_ticket_rate_90d, avg_resolution_hours_90d
- days_since_signup, sessions_30d, product_views_30d, cart_adds_30d
- wishlist_adds_30d, abandoned_carts_30d, email_opens_30d, campaign_clicks_30d
- last_visit_days_ago

## Interactive API Docs

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`