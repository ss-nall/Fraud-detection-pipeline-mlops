Fraud Detection MLOps Pipeline

An end-to-end machine learning pipeline for fraud detection with experiment tracking, model versioning, API-based inference, containerization and Kubernetes deployment.

Overview

This project implements a production-oriented fraud detection workflow that takes transaction data as input and returns fraud predictions through a REST API.

The system covers the complete ML lifecycle:

Data preprocessing and feature engineering
Model training and evaluation
Hyperparameter optimization
Threshold tuning
Experiment tracking with MLflow
Model registration and versioning
REST API inference using FastAPI
Containerization using Docker
Kubernetes deployment using Minikube
Streamlit-based interface for interacting with the model

Tech Stack
Component	Technology
Programming	Python
ML	Scikit-learn
API	FastAPI
UI	Streamlit
Experiment Tracking	MLflow
Model Registry	MLflow Model Registry
Containerization	Docker
Orchestration	Kubernetes
Local Cluster	Minikube
Data/Model Processing	Pandas, NumPy
ML Pipeline

The machine learning workflow consists of the following stages:

1. Data Preprocessing

Transaction data is cleaned and transformed before being passed to the model. The preprocessing pipeline handles the required feature transformations and ensures consistent processing during training and inference.

2. Baseline Model

A baseline fraud detection model is trained to establish an initial performance benchmark.

The baseline experiment is tracked in MLflow under:

Fraud Detection
└── baseline_model
3. Hyperparameter Optimization

Randomized hyperparameter search is used to explore different model configurations and improve predictive performance.

The experiment is tracked as:

random_search
4. Threshold Tuning

Since fraud detection is typically an imbalanced classification problem, the default classification threshold may not provide the desired precision-recall tradeoff.

A separate threshold tuning stage is used to determine a suitable decision threshold for fraud predictions.

Tracked in MLflow as:

threshold_tuning
5. Model Registry

The selected model is registered in MLflow under:

FraudDetectionModel

This provides versioning and a central location for managing the model used for inference.

API

The FastAPI service exposes endpoints for health checks and file-based predictions.

Health Check
GET /health

Used to verify that the API service is running.

File Prediction
POST /predict-file

Accepts transaction data in CSV/JSON format and returns fraud predictions.

FastAPI automatically provides interactive API documentation at:

/docs
Running Locally
1. Clone the repository
git clone <repository-url>
cd <repository-name>
2. Create a virtual environment
python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Linux/macOS

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Start the FastAPI service
uvicorn app:app --reload

The API will be available at:

http://localhost:8000

Swagger documentation:

http://localhost:8000/docs
5. Start Streamlit
streamlit run streamlit_app.py

The Streamlit application will be available at:

http://localhost:8501
MLflow

MLflow is used for experiment tracking and model management.

The project tracks multiple stages of experimentation:

baseline_model
random_search
threshold_tuning

The final model is registered as:

FraudDetectionModel

To start the MLflow UI:

mlflow ui

Then open:

http://localhost:5000
Docker

The application can be containerized using Docker.

Build the image:

docker build -t fraud-detection .

Run the container:

docker run -p 8501:8501 fraud-detection

The Docker image uses a lightweight Python 3.11 base image.

Kubernetes Deployment

The FastAPI service can be deployed to a local Kubernetes cluster using Minikube.

Start Minikube:

minikube start

Apply the Kubernetes configuration:

kubectl apply -f k8s/

The deployment runs multiple FastAPI replicas for basic service availability.

Check the deployment:

kubectl get deployments

Check the pods:

kubectl get pods

Check the service:

kubectl get services

The FastAPI service is exposed through a Kubernetes NodePort.

Project Structure
fraud-detection-mlops/
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   └── inference/
│
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   └── service.yaml
│
├── app.py
├── streamlit_app.py
├── Dockerfile
├── requirements.txt
└── README.md
MLOps Workflow
Data
  │
  ▼
Preprocessing
  │
  ▼
Baseline Model
  │
  ▼
Hyperparameter Search
  │
  ▼
Threshold Tuning
  │
  ▼
MLflow Experiment Tracking
  │
  ▼
Model Registry
  │
  ▼
FastAPI
  │
  ▼
Docker Container
  │
  ▼
Kubernetes / Minikube
  │
  ▼
Prediction
Key Features
End-to-end fraud classification pipeline
Experiment tracking with MLflow
Hyperparameter optimization
Classification threshold tuning
MLflow model registry
REST API using FastAPI
Interactive Streamlit interface
Dockerized application
Kubernetes deployment
Multi-replica API deployment
Swagger API documentation
Future Improvements
Automated CI/CD pipeline
Automated model retraining
Data and model versioning
Production cloud deployment
Model performance monitoring
Data drift detection
Automated model promotion through MLflow
Centralized logging and observability
