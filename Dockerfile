
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY Requirement.txt .
RUN pip install --no-cache-dir -r Requirement.txt

COPY main.py .
COPY Train_Model.py .
COPY Test_Api .
COPY Readme.md .
COPY churn_model.pkl .
COPY label_encoders.pkl .
COPY model_features.pkl .

# Expose the API port
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]