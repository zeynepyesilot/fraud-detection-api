from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.transaction import PredictionResponse, TransactionRequest
from app.services.model_service import model_service

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest) -> PredictionResponse:
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        is_fraud, probability = model_service.predict(transaction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=probability,
        model_version=settings.APP_VERSION,
    )


@router.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_service.is_loaded else "degraded",
        "model_loaded": model_service.is_loaded,
        "model_version": settings.APP_VERSION,
    }

