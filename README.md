# Fraud Detection MLOps Pipeline

An end-to-end MLOps pipeline for fraud detection covering model training, experimentation, model versioning and deployment.

## Features

* Data preprocessing and feature engineering
* Model training and evaluation
* Hyperparameter optimization using Randomized Search
* Classification threshold tuning
* Experiment tracking and model registry with MLflow
* REST API inference using FastAPI
* Interactive prediction interface using Streamlit
* Docker containerization
* Kubernetes deployment using Minikube

## Tech Stack

**Python · Scikit-learn · Pandas · FastAPI · Streamlit · MLflow · Docker · Kubernetes · Minikube**

## ML Workflow

```text
Data → Preprocessing → Model Training → Hyperparameter Tuning
→ Threshold Tuning → MLflow Tracking → Model Registry
→ FastAPI → Docker → Kubernetes
```

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Start the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Start MLflow:

```bash
mlflow ui
```

API documentation: `http://localhost:8000/docs`

MLflow dashboard: `http://localhost:5000`

## Docker

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## Kubernetes

```bash
minikube start
kubectl apply -f k8s/
kubectl get pods
kubectl get services
```

