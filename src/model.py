"""
model.py

Thin wrapper around an XGBoost classifier for training and inference on the
engineered feature set.
"""

import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score


NON_FEATURE_COLUMNS = ["open_time", "close_time", "label"]


def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def train_model(train_df, params: dict):
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df["label"]

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, feature_cols


def evaluate_model(model, test_df, feature_cols) -> dict:
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "n_test": len(test_df),
        "positive_rate": y_test.mean(),
    }
