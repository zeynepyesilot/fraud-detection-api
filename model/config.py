import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard_2023.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

TARGET_COL = "Class"
DROP_COLS = ["id"]
FEATURE_COLS_TO_SCALE = ["Amount"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

SMOTE_SAMPLING_STRATEGY = "minority"
SMOTE_RANDOM_STATE = RANDOM_STATE
SMOTE_K_NEIGHBORS = 5

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "scale_pos_weight": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
