import pandas as pd
import joblib

from app.core.config import settings
from app.schemas.transaction import TransactionRequest


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.metrics: dict = {}
        self._is_loaded = False

    def load_model(self, path: str | None = None) -> None:
        path = path or settings.MODEL_PATH
        artefact = joblib.load(path)

        self.model = artefact["model"]
        self.scaler = artefact["scaler"]
        self.feature_names = artefact["feature_names"]
        self.metrics = artefact.get("metrics", {})
        self._is_loaded = True

        print(f"[model_service] Loaded from {path}")
        print(f"[model_service] Features: {len(self.feature_names)} | "
              f"Train ROC-AUC: {self.metrics.get('roc_auc', 'N/A')}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, transaction: TransactionRequest) -> tuple[bool, float]:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        data = transaction.model_dump()
        df = pd.DataFrame([data])

        # Scale Amount the same way it was scaled during training
        if "Amount" in df.columns and self.scaler is not None:
            df[["Amount"]] = self.scaler.transform(df[["Amount"]])

        # Column order must match training
        df = df[self.feature_names]

        proba = float(self.model.predict_proba(df)[0, 1])
        is_fraud = bool(proba >= 0.5)

        return is_fraud, round(proba, 6)


model_service = ModelService()

