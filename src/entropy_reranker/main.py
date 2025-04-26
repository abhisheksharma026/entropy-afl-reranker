import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from .model_pipeline import compute_metrics, evaluate_lr, evaluate_afl, evaluate_final_reranker, evaluate_meta_calibrator, set_random_seed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score
)
from collections import defaultdict
import pandas as pd
from .config import CONFIG

def run_cross_validation(X_full, y_full, n_splits=5):
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=CONFIG["SEED"]
    )
    y_train_full = np.array(y_train_full)
    y_holdout = np.array(y_holdout)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CONFIG["SEED"])
    val_metrics = defaultdict(lambda: defaultdict(list))
    holdout_metrics = defaultdict(lambda: defaultdict(list))

    model_funcs = {
        "Logistic Regression (Sklearn)": evaluate_lr,
        "Asymmetric Focal + Calibration (Dual Constraint)": evaluate_afl,
        "Calibrated Asym Focal XGB Re-Ranker": evaluate_final_reranker,
        "Meta-Calibrated Asym Focal XGB Re-Ranker": evaluate_meta_calibrator,
    }

    for train_idx, val_idx in kf.split(X_train_full, y_train_full):
        X_temp, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_temp, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # Split into model + calibration
        X_model, X_cal, y_model, y_cal = train_test_split(
            X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
        )

        for name, func in model_funcs.items():
            if func in [evaluate_final_reranker, evaluate_meta_calibrator]:
                val_result = func(X_model, y_model, X_cal, y_cal, X_val, y_val)
                holdout_result = func(X_model, y_model, X_cal, y_cal, X_holdout, y_holdout)
            else:
                val_result = func(X_model, y_model, X_val, y_val)
                holdout_result = func(X_model, y_model, X_holdout, y_holdout)

            for metric_name, score in compute_metrics(**val_result).items():
                val_metrics[name][metric_name].append(score)
            for metric_name, score in compute_metrics(**holdout_result).items():
                holdout_metrics[name][metric_name].append(score)

    val_results = {
    name: {
        metric: int(np.sum(values)) if metric == "Support" else round(np.mean(values), 5)
        for metric, values in metric_dict.items()
    }
    for name, metric_dict in val_metrics.items()
    }


    holdout_results = {
    name: {
        metric: int(np.sum(values)) if metric == "Support" else round(np.mean(values), 5)
        for metric, values in metric_dict.items()
    }
    for name, metric_dict in holdout_metrics.items()
    }

    return pd.DataFrame(val_results).T, pd.DataFrame(holdout_results).T

def run_lending():
    df = pd.read_csv(CONFIG["DATA_PATH_LENDING"]).dropna()
    print(f"Lending Data shape: {df.shape}")
    y = df[CONFIG["TARGET_COLUMN_LENDING"]].astype(int).values
    X = df.drop(columns=[CONFIG["TARGET_COLUMN_LENDING"]]).copy()
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    X = X.values

    val_df, holdout_df = run_cross_validation(X, y, n_splits=5)
    print("\\n📊 Average Cross-Validation Metrics:")
    print(val_df)
    print("\\n📌 Average Holdout Set Metrics:")
    print(holdout_df)

def run_credit_card():
    df = pd.read_csv(CONFIG["DATA_PATH_CC"]).dropna()
    print(f"CC Data shape: {df.shape}")
    y = df[CONFIG["TARGET_COLUMN_CC"]].astype(int).values
    X = df.drop(columns=[CONFIG["TARGET_COLUMN_CC"], "Time"]).copy()
    X["Amount"] = StandardScaler().fit_transform(X[["Amount"]])
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    X = X.values

    val_df, holdout_df = run_cross_validation(X, y, n_splits=5)
    print("\\n📊 Average Cross-Validation Metrics:")
    print(val_df)
    print("\\n📌 Average Holdout Set Metrics:")
    print(holdout_df)

def main():
    seed = set_random_seed(42)
    run_lending()
    # run_credit_card()

if __name__ == "__main__":
    main()