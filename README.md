# Fraud Detection API

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?logo=xgboost&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

REST API that takes a credit card transaction and tells you if it's fraudulent, built with XGBoost and FastAPI, trained on the [Credit Card Fraud Detection 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) dataset (568,630 transactions)

## How it works

`creditcard_2023.csv` → `model/train.py` (preprocess, scale, split, SMOTE, train XGBoost) → `fraud_model.pkl` → FastAPI loads it at startup → `POST /predict` returns fraud probability.

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Precision | 0.9992 |
| Recall    | 1.0000 |
| F1-Score  | 0.9996 |
| ROC-AUC   | 1.0000 |

Confusion matrix on 113,726 test samples: 56,815 TN, 48 FP, 0 FN, 56,863 TP.

These numbers are very high because the 2023 dataset comes pre balanced at 50/50 (284k fraud, 284k legit), so SMOTE didn't actually change anything during training. The pipeline still includes SMOTE so it works correctly if retrained on real world imbalanced data where fraud is typically <1% 

## Quick Start

```bash
git clone https://github.com/zeynepyesilot/fraud-detection-api.git
cd fraud-detection-api
pip install -r requirements.txt
```

Train the model (you will need `data/creditcard_2023.csv` from Kaggle):

```bash
python -m model.train
```

Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker-compose up --build
```

The model file is mounted as a volume, not baked into the image. Train locally first, then start the container.

## API

### `GET /health`

```json
{"status": "healthy", "model_loaded": true, "model_version": "1.0.0"}
```

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": -0.34,
    "V25": -0.07, "V26": -0.06, "V27": -0.03, "V28": -0.01,
    "Amount": 149.62
  }'
```

```json
{"is_fraud": false, "fraud_probability": 0.083701, "model_version": "1.0.0"}
```

Swagger docs at `http://localhost:8000/docs`.

## Project Structure

```
fraud-detection-api/
├── app/
│   ├── main.py                  # FastAPI entry point, lifespan, CORS
│   ├── core/
│   │   └── config.py            # pydantic-settings, reads .env
│   ├── routes/
│   │   └── predict.py           # POST /predict, GET /health
│   ├── schemas/
│   │   └── transaction.py       # Request/response models
│   └── services/
│       └── model_service.py     # Model loading + inference
├── model/
│   ├── config.py                # Hyperparameters, paths
│   ├── preprocess.py            # Scale, split, SMOTE
│   ├── evaluate.py              # Metrics
│   └── train.py                 # Training orchestrator
├── data/
│   └── creditcard_2023.csv
├── saved_models/
│   └── fraud_model.pkl
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Tech Stack

Model: XGBoost
Preprocessing scikit-learn, imbalanced-learn
API: FastAPI + Uvicorn
Validation: Pydantic v2
Serialization: joblib
Container: Docker, Docker Compose

