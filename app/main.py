from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes.predict import router as predict_router
from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\nStarting {settings.APP_NAME} v{settings.APP_VERSION}")
    model_service.load_model()
    print("Model ready, accepting requests\n")
    yield
    print("\nShutting down\n")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "REST API for detecting fraudulent credit-card transactions "
        "using an XGBoost model trained on the Kaggle Credit Card "
        "Fraud Detection 2023 dataset."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


