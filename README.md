# End-to-End Customer Churn Prediction Pipeline (MLOps)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-teal)
![Evidently AI](https://img.shields.io/badge/Monitoring-Evidently_AI-orange)

## Overview
This project is a production-ready **MLOps pipeline** designed to predict customer churn. Unlike simple notebook experiments, this system is built for deployment, featuring a REST API for real-time predictions and a monitoring service to detect **data drift** and **model degradation** over time.

## Architecture
The pipeline bridges the gap between data science and engineering:

* **Training Pipeline:** Automates data preprocessing, feature engineering, and model training using **Scikit-learn**.
* **Serving Layer:** Exposes the trained model as a REST API using **FastAPI**.
* **Containerization:** The entire application is containerized with **Docker** to ensure consistency across environments.
* **Monitoring:** Integrates **Evidently AI** to track distribution changes in input data (Data Drift) and model performance.

## Tech Stack
* **Modeling:** Scikit-learn, Pandas
* **API Framework:** FastAPI, Uvicorn
* **Containerization:** Docker
* **Monitoring:** Evidently AI
* **Version Control:** Git

## Project Structure
```bash
├── src/
│   ├── train_model.py    # Training pipeline script
│   ├── predict.py        # Inference logic
│   └── app.py            # FastAPI endpoints
├── monitoring/
│   └── dashboard.py      # Evidently AI dashboard configuration
├── models/               # Serialized models (.pkl)
├── Dockerfile
└── requirements.txt
