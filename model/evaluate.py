from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, print_report: bool = True) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    if print_report:
        _print_report(metrics, y_test, y_pred)

    return metrics


def _print_report(metrics: dict, y_test, y_pred) -> None:
    cm = metrics["confusion_matrix"]

    print(f"\nPrecision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]:>7,}  FP={cm[0][1]:>7,}")
    print(f"  FN={cm[1][0]:>7,}  TP={cm[1][1]:>7,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")
