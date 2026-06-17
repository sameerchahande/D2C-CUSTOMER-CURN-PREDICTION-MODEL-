# Monitoring and Deployment Plan

## 📊 Overview

This document outlines the monitoring, performance tracking, and deployment strategy for the Customer Churn Prediction API.

---

## 🎯 Key Metrics to Monitor

### API Performance Metrics
- **Response Time**: Average latency for predictions
  - Target: < 200ms for single predictions
  - Target: < 500ms for batch predictions (10-50 customers)
- **Availability**: Uptime percentage
  - Target: > 99.5% uptime
- **Error Rate**: Percentage of failed requests
  - Target: < 0.5%
- **Throughput**: Requests per minute
  - Monitor: Track peak and average traffic

### Model Performance Metrics
- **Accuracy**: Overall correctness of predictions
  - Track against validation baseline
- **Precision**: True positive rate for churn predictions
  - Target: > 0.75
- **Recall**: Percentage of actual churns caught
  - Target: > 0.70
- **F1 Score**: Harmonic mean of precision and recall
  - Target: > 0.72
- **AUC-ROC**: Area under the receiver operating characteristic curve
  - Target: > 0.80

### Data Quality Metrics
- **Missing Values**: Track null/invalid inputs
- **Feature Distribution Drift**: Monitor for shifts in input data patterns
- **Label Distribution**: Track changes in churn rate over time
- **Encoding Errors**: Log failed categorical encoding attempts

---

## 🔍 Health Checks

### API Health Endpoint
The `/health` endpoint provides real-time status:

```bash
curl http://localhost:8000/health
```

**Response indicates:**
- API server status
- Model artifact loading status
- Total number of features loaded

### Automated Health Checks
- Implement periodic health checks (every 5 minutes)
- Alert if model fails to load on startup
- Monitor disk space for model artifacts

---

## 📈 Performance Monitoring

### Request Logging
Log all API requests with:
```python
{
  "timestamp": "2024-01-15T10:30:45Z",
  "endpoint": "/predict",
  "request_size": 1024,
  "response_time_ms": 145,
  "status_code": 200,
  "customer_count": 5,
  "model_version": "1.0.0"
}
```

### Error Tracking
- Log all errors with stack traces
- Track error frequency by type (validation, model, etc.)
- Alert on spike in error rate

### Model Prediction Logging
```python
{
  "timestamp": "2024-01-15T10:30:45Z",
  "prediction_id": "uuid",
  "churn_probability": 0.65,
  "churn_prediction": 1,
  "actual_churn": null,  # Update later with ground truth
  "confidence": "high"
}
```

---

## 🚨 Alerts and Thresholds

### Critical Alerts (Immediate Action)
| Metric | Threshold | Action |
|--------|-----------|--------|
| API Downtime | > 5 minutes | Page on-call engineer |
| Error Rate | > 5% | Investigate logs immediately |
| Model Load Failure | Any failure | Restart service, verify artifacts |
| Response Time | > 1000ms | Check server resources |

### Warning Alerts (Review Within 1 Hour)
| Metric | Threshold | Action |
|--------|-----------|--------|
| Response Time | > 500ms average | Analyze slow queries |
| Feature Distribution Drift | Chi-square p-value < 0.05 | Plan model retraining |
| Low Precision | < 0.70 | Schedule model review |

---

## 🔄 Model Retraining Strategy

### Trigger Conditions for Retraining
1. **Performance Degradation**: Precision or Recall drops below 0.70
2. **Data Drift**: Significant shift in feature distributions detected
3. **Class Imbalance Change**: Churn rate deviation > 10% from baseline
4. **Scheduled Retraining**: Monthly (or quarterly, as needed)

### Retraining Workflow
```
1. Collect new training data (last 30-90 days)
2. Run train_model.py with updated dataset
3. Evaluate model on held-out test set
4. Compare performance vs. current production model
5. If improved: stage new model for deployment
6. Run smoke tests (tests/test_api.py) with new model
7. Schedule canary deployment (10% traffic)
8. Monitor metrics for 24 hours
9. Full rollout if metrics are acceptable
10. Keep previous model as fallback
```

### Model Versioning
```
Model artifacts structure:
├── models/
│   ├── v1.0.0/
│   │   ├── churn_model.pkl
│   │   ├── label_encoders.pkl
│   │   ├── model_features.pkl
│   │   └── metadata.json (performance metrics)
│   ├── v1.1.0/
│   └── current/ -> v1.0.0/ (symlink)
```

---

## 📊 Monitoring Dashboard Components

### Real-Time Metrics
- API response time (histogram)
- Error rate (line chart)
- Request throughput (requests/min)
- Active connections

### Model Health
- Precision, Recall, F1-Score trends
- Prediction distribution (churn vs. no-churn)
- Confidence score distribution
- Feature importance ranking

### Data Quality
- Feature distribution changes
- Missing value rate per feature
- Encoding error frequency
- Input data statistics

### Business Metrics
- Customers processed per day
- High-risk customers identified
- Retention campaign effectiveness (measured post-campaign)

---

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] Run full test suite: `python tests/test_api.py`
- [ ] Validate model performance on test set
- [ ] Check feature completeness and data quality
- [ ] Verify all dependencies in requirements.txt
- [ ] Review error logs from staging environment
- [ ] Backup current model artifacts

### Deployment Steps
```bash
# 1. Stop current API
sudo systemctl stop churn-api

# 2. Backup model artifacts
cp -r /var/models /var/models.backup.$(date +%Y%m%d)

# 3. Deploy new code
git pull origin main
pip install -r requirements.txt

# 4. Start API
sudo systemctl start churn-api

# 5. Verify health check
curl http://localhost:8000/health

# 6. Run smoke tests
python tests/test_api.py
```

### Post-Deployment Monitoring
- [ ] Monitor error rate for first 1 hour
- [ ] Check response time percentiles (p50, p95, p99)
- [ ] Verify model artifact loading
- [ ] Test with sample customer data
- [ ] Confirm logging is working

---

## 📝 Logging Configuration

### Log Levels
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
```

### Log Files Location
- **API Logs**: `/var/logs/churn_api.log`
- **Error Logs**: `/var/logs/churn_api.error.log`
- **Model Logs**: `/var/logs/churn_model_training.log`

### Log Retention
- Keep logs for 30 days
- Archive older logs to S3/storage
- Rotate logs when they reach 100MB

---

## 🔐 Security Monitoring

### API Security
- Monitor for unusual request patterns (DDoS detection)
- Track failed authentication attempts
- Log all prediction requests with customer IDs
- Encrypt sensitive data in transit (HTTPS)

### Model Security
- Verify model artifact integrity (checksums)
- Control access to model files (chmod 600)
- Track who updates/deploys models
- Audit trail for all model changes

---

## 📞 Escalation Procedure

### On-Call Rotation
- **Tier 1**: Application monitoring (automated alerts)
- **Tier 2**: On-call engineer (responds within 15 minutes)
- **Tier 3**: Team lead (escalate if Tier 2 cannot resolve)

### Incident Response SLA
- **Critical**: Resolution within 30 minutes
- **High**: Resolution within 2 hours
- **Medium**: Resolution within 24 hours
- **Low**: Resolution within 1 week

---

## 📅 Monitoring Schedule

### Daily Tasks
- Review error logs
- Check error rate trends
- Monitor API availability

### Weekly Tasks
- Review model performance metrics
- Analyze prediction accuracy trends
- Check feature distribution changes
- Review customer complaint feedback

### Monthly Tasks
- Full model performance audit
- Retraining decision (if needed)
- Performance optimization review
- Capacity planning review

### Quarterly Tasks
- Comprehensive model evaluation
- Update monitoring alerts based on trends
- Review and update this plan
- Stakeholder reporting

---

## 🛠️ Troubleshooting Guide

### Issue: Model Not Loading on Startup
**Solution:**
1. Verify pickle files exist: `churn_model.pkl`, `label_encoders.pkl`, `model_features.pkl`
2. Check file permissions: `ls -la *.pkl`
3. Verify file integrity (run `train_model.py` to regenerate)

### Issue: High Response Times
**Solution:**
1. Check server CPU/memory: `top`, `free -h`
2. Monitor model prediction latency
3. Increase API workers: `uvicorn app.main:app --workers 4`
4. Consider model optimization or caching

### Issue: High Error Rate
**Solution:**
1. Check error logs: `tail -f /var/logs/churn_api.error.log`
2. Validate incoming data format
3. Verify label encoder compatibility with new data
4. Check for missing categorical values

---

## 📚 Appendix

### Recommended Tools
- **Monitoring**: Prometheus, Grafana, DataDog
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana), Splunk
- **Alerting**: PagerDuty, Opsgenie
- **Model Registry**: MLflow, Weights & Biases

### References
- FastAPI Monitoring Guide: https://fastapi.tiangolo.com/
- Scikit-learn Model Persistence: https://scikit-learn.org/stable/modules/model_persistence.html
- Machine Learning Monitoring Best Practices
