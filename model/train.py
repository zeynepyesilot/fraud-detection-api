import os
import time
import joblib
from xgboost import XGBClassifier

from model.config import MODEL_DIR, MODEL_PATH, XGB_PARAMS
from model.preprocess import preprocess_pipeline
from model.evaluate import evaluate_model


def train():
    print("\nStarting training pipeline\n")

    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()

    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(**XGB_PARAMS)

    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # confusion_matrix is a numpy array and not useful in the pkl metadata
    artefact = {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
        "metrics": {k: v for k, v in metrics.items()
                    if k != "confusion_matrix"},
    }
    joblib.dump(artefact, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"File size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB\n")

    return model, scaler, metrics


if __name__ == "__main__":
    train()
