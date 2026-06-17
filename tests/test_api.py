"""
Test script for Customer Churn Prediction API
Run this AFTER training the model and starting the FastAPI server.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

sample_customers = [
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
    },
    {
        "city_tier": "Tier 2",
        "age_group": "25-34",
        "acquisition_channel": "Google Search",
        "loyalty_tier": "Gold",
        "preferred_category": "Hair Care",
        "marketing_consent": "Yes",
        "recency_days": 35,
        "frequency_180d": 2,
        "monetary_180d": 1523.14,
        "return_rate_180d": 0.0,
        "avg_discount_pct_180d": 0.285,
        "avg_rating_180d": 5.0,
        "category_diversity_180d": 2,
        "ticket_count_90d": 0,
        "negative_ticket_rate_90d": 0.0,
        "avg_resolution_hours_90d": 0.0,
        "days_since_signup": 319,
        "sessions_30d": 6,
        "product_views_30d": 23,
        "cart_adds_30d": 2,
        "wishlist_adds_30d": 0,
        "abandoned_carts_30d": 1,
        "email_opens_30d": 0,
        "campaign_clicks_30d": 0,
        "last_visit_days_ago": 11
    },
    {
        "city_tier": "Tier 1",
        "age_group": "18-24",
        "acquisition_channel": "Instagram",
        "loyalty_tier": "None",
        "preferred_category": "Skin Care",
        "marketing_consent": "Yes",
        "recency_days": 320,
        "frequency_180d": 0,
        "monetary_180d": 0.0,
        "return_rate_180d": 0.0,
        "avg_discount_pct_180d": 0.0,
        "avg_rating_180d": 3.5,
        "category_diversity_180d": 0,
        "ticket_count_90d": 0,
        "negative_ticket_rate_90d": 0.0,
        "avg_resolution_hours_90d": 0.0,
        "days_since_signup": 555,
        "sessions_30d": 11,
        "product_views_30d": 35,
        "cart_adds_30d": 2,
        "wishlist_adds_30d": 2,
        "abandoned_carts_30d": 1,
        "email_opens_30d": 5,
        "campaign_clicks_30d": 1,
        "last_visit_days_ago": 60
    }
]


def test_health():
    """Test the health endpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {json.dumps(resp.json(), indent=4)}")
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        print("  FAILED: Could not connect to the API. Is it running?")
        return False


def test_single_prediction():
    print("\n" + "=" * 60)
    print("TEST 2: Single Customer Prediction")
    print("=" * 60)
    try:
        payload = {"customer": sample_customers[0]}
        resp = requests.post(f"{BASE_URL}/predict/single", json=payload, timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {json.dumps(resp.json(), indent=4)}")
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        print("  FAILED: Could not connect to the API.")
        return False


def test_batch_prediction():
    """Test batch customer prediction."""
    print("\n" + "=" * 60)
    print("TEST 3: Batch Customer Prediction")
    print("=" * 60)
    try:
        payload = {"customers": sample_customers}
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {json.dumps(resp.json(), indent=4)}")
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        print("  FAILED: Could not connect to the API.")
        return False


def test_root():
    """Test the root endpoint."""
    print("\n" + "=" * 60)
    print("TEST 4: Root Endpoint")
    print("=" * 60)
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {json.dumps(resp.json(), indent=4)}")
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        print("  FAILED: Could not connect to the API.")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CUSTOMER CHURN PREDICTION API TEST SUITE")
    print("=" * 60)
    print("\nMake sure the FastAPI server is running on http://localhost:8000")
    print("Start it with: uvicorn app.main:app --reload\n")
    
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Single Prediction", test_single_prediction()))
    results.append(("Batch Prediction", test_batch_prediction()))
    results.append(("Root Endpoint", test_root()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  {name:25s}: {status}")
    
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
