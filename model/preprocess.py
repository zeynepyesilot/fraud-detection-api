import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from model.config import (
    DATA_PATH,
    TARGET_COL,
    DROP_COLS,
    FEATURE_COLS_TO_SCALE,
    TEST_SIZE,
    RANDOM_STATE,
    SMOTE_SAMPLING_STRATEGY,
    SMOTE_RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[preprocess] Loaded {len(df):,} rows | "
          f"Fraud: {df[TARGET_COL].sum():,} ({df[TARGET_COL].mean() * 100:.2f}%)")
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Returns (df, fitted_scaler). Scaler is persisted for inference."""
    scaler = StandardScaler()
    cols = [c for c in FEATURE_COLS_TO_SCALE if c in df.columns]
    if cols:
        df[cols] = scaler.fit_transform(df[cols])
        print(f"[preprocess] Scaled columns: {cols}")
    return df, scaler


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[preprocess] Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=SMOTE_RANDOM_STATE,
        k_neighbors=SMOTE_K_NEIGHBORS,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[preprocess] After SMOTE - "
          f"Train: {len(X_res):,} | Fraud: {y_res.sum():,} ({y_res.mean() * 100:.2f}%)")
    return X_res, y_res


def preprocess_pipeline(data_path: str = DATA_PATH):
    df = load_data(data_path)
    df, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, y_train = apply_smote(X_train, y_train)
    return X_train, X_test, y_train, y_test, scaler

