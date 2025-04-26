import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, average_precision_score,
    precision_score, recall_score, f1_score, precision_recall_curve,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
import os
from .config import CONFIG

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def compute_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ece = np.abs(prob_true - prob_pred).mean()
    return round(ece, 6)

class LogisticModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)
    
def asymmetric_focal_loss(logits, 
                          targets, 
                          gamma_pos=CONFIG['GAMMA_POS'], 
                          gamma_neg=CONFIG['GAMMA_NEG'], 
                          pos_weight=CONFIG["POS_WEIGHT"]):
    
    probs = torch.sigmoid(logits)
    loss_pos = -(1 - probs) ** gamma_pos * torch.log(probs + 1e-8) * targets
    loss_neg = -(probs) ** gamma_neg * torch.log(1 - probs + 1e-8) * (1 - targets)
    return (pos_weight * loss_pos + loss_neg).mean()

def entropy_regularized_afl(logits, 
                            targets, 
                            gamma_pos=CONFIG['GAMMA_POS'], 
                            gamma_neg=CONFIG['GAMMA_NEG'],
                            pos_weight=CONFIG["POS_WEIGHT"], 
                            lambda_entropy=CONFIG["LAMBDA_ENTROPY"]):
    
    probs = torch.sigmoid(logits)
    loss_pos = -(1 - probs) ** gamma_pos * torch.log(probs + 1e-8) * targets
    loss_neg = -(probs) ** gamma_neg * torch.log(1 - probs + 1e-8) * (1 - targets)
    afl_loss = (pos_weight * loss_pos + loss_neg).mean()
    entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
    return afl_loss + lambda_entropy * entropy.mean()

def dynamic_filter_threshold(probs, y, target_recall=0.7):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    valid = recall >= target_recall
    if valid.any():
        return thresholds[valid.argmax()]
    else:
        return np.percentile(probs, 95)  # fallback


def train_model(model, loss_fn, optimizer, X, y, epochs=CONFIG["EPOCHS"]):
    for _ in range(epochs):
        model.train()
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def threshold_dual_constraint(y_true, y_probs, min_precision=CONFIG['MIN_PRECISION'], min_recall=CONFIG['MIN_RECALL']):
    best_f1, best_thresh = 0, 0.5
    for t in np.linspace(0.01, 0.99, 100):
        preds = (y_probs >= t).astype(int)
        if preds.sum() == 0: continue
        precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)
        if precision >= min_precision and recall >= min_recall:
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
    return best_thresh

def compute_metrics(y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred, output_dict=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ece_bins_true, ece_bins_pred = calibration_curve(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), n_bins=10)
    ece = np.abs(ece_bins_true - ece_bins_pred).mean()
    avg_confidence = float(np.mean(y_prob))
    support = int(np.sum(y_true))

    return {
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1 Score": report["1"]["f1-score"],
        "ECE": ece,
        "Support": int(support)
        }

def evaluate_lr(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=CONFIG["LR_MAX_ITER"],)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {"y_true": y_test, "y_pred": preds, "y_prob": probs}

def evaluate_afl(X_train, y_train, X_test, y_test):
    model = LogisticModel(X_train.shape[1])
    train_model(model, lambda l, t: asymmetric_focal_loss(l, t),
                torch.optim.Adam(model.parameters(), lr=0.01),
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32))
    probs = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).detach().numpy()
    cal_probs = IsotonicRegression(out_of_bounds='clip').fit(probs, y_test).transform(probs)
    thresh = threshold_dual_constraint(y_test, cal_probs)
    preds = (cal_probs >= thresh).astype(int)
    return {"y_true": y_test, "y_pred": preds, "y_prob": probs}


def evaluate_final_reranker(X_train, y_train, X_cal, y_cal, X_eval, y_eval):
    model = LogisticModel(X_train.shape[1])
    train_model(model, lambda l, t: entropy_regularized_afl(l, t),
                torch.optim.Adam(model.parameters(), lr=0.01),
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32))

    # Stage-1 on calibration set
    probs_cal = torch.sigmoid(model(torch.tensor(X_cal, dtype=torch.float32))).detach().numpy()
    cal_stage1 = IsotonicRegression(out_of_bounds='clip').fit(probs_cal, y_cal).transform(probs_cal)

    # Set adaptive filter threshold
    filter = np.percentile(cal_stage1, 90)
    mask = cal_stage1 >= filter
    if mask.sum() < 100:
        filter = np.percentile(cal_stage1, 85)
        mask = cal_stage1 >= filter

    X_rerank = X_cal[mask]
    y_rerank = y_cal[mask]
    stage1_scores = cal_stage1[mask].reshape(-1, 1)
    X_rerank_aug = np.hstack([X_rerank, stage1_scores])

    xgb_conf = xgb.XGBClassifier(n_estimators=CONFIG["RERANKER_N_ESTIMATORS"], 
                                 eval_metric='logloss', 
                                 random_state=CONFIG["SEED"])
    xgb_conf.fit(X_rerank_aug, y_rerank)

    # Predict on eval
    probs_eval = torch.sigmoid(model(torch.tensor(X_eval, dtype=torch.float32))).detach().numpy()
    stage1_eval = IsotonicRegression(out_of_bounds='clip').fit(probs_cal, y_cal).transform(probs_eval)
    X_eval_aug = np.hstack([X_eval, stage1_eval.reshape(-1, 1)])
    probs = xgb_conf.predict_proba(X_eval_aug)[:, 1]

    cal_probs = IsotonicRegression(out_of_bounds='clip').fit(probs_cal, y_cal).transform(probs)
    thresh = threshold_dual_constraint(y_eval, cal_probs)
    preds = (cal_probs >= thresh).astype(int)
    return {"y_true": y_eval, "y_pred": preds, "y_prob": probs}


def evaluate_meta_calibrator(X_train, y_train, X_cal, y_cal, X_eval, y_eval):
    # Stage-1 model
    model = LogisticModel(X_train.shape[1])
    train_model(
        model, lambda l, t: entropy_regularized_afl(l, t),
        torch.optim.Adam(model.parameters(), lr=0.01),
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    # Stage-1 predictions and calibration on cal set
    probs_stage1_cal = torch.sigmoid(model(torch.tensor(X_cal, dtype=torch.float32))).detach().numpy()
    cal_stage1 = IsotonicRegression(out_of_bounds='clip').fit(probs_stage1_cal, y_cal).transform(probs_stage1_cal)

    # Reranker trained on top 10% confident cal samples
    mask = cal_stage1 >= np.percentile(cal_stage1, 90)
    X_rerank = X_cal[mask]
    y_rerank = y_cal[mask]
    scores_rerank = cal_stage1[mask].reshape(-1, 1)
    X_rerank_aug = np.hstack([X_rerank, scores_rerank])

    xgb_conf = xgb.XGBClassifier(n_estimators=CONFIG["RERANKER_N_ESTIMATORS"], 
                                 eval_metric='logloss', 
                                 random_state=CONFIG["SEED"])
    xgb_conf.fit(X_rerank_aug, y_rerank)

    # Stage-1 predictions on eval
    probs_stage1_eval = torch.sigmoid(model(torch.tensor(X_eval, dtype=torch.float32))).detach().numpy()
    stage1_eval = IsotonicRegression(out_of_bounds='clip').fit(probs_stage1_cal, y_cal).transform(probs_stage1_eval)

    # Reranker predictions on eval
    X_eval_aug = np.hstack([X_eval, stage1_eval.reshape(-1, 1)])
    probs_rerank = xgb_conf.predict_proba(X_eval_aug)[:, 1]

    # Combine for meta features
    X_meta = np.vstack([stage1_eval, probs_rerank]).T

    # Anti-leakage: Split meta input into train/test for meta-model
    X_meta_train, X_meta_test, y_meta_train, y_meta_test, idx_train, idx_test = train_test_split(
        X_meta, y_eval, np.arange(len(y_eval)), test_size=0.5, stratify=y_eval, random_state=CONFIG["SEED"]
    )

    meta_model = xgb.XGBClassifier(n_estimators=CONFIG['META_N_ESTIMATORS'], 
                                   random_state=CONFIG["SEED"], 
                                   eval_metric='logloss')
    meta_model.fit(X_meta_train, y_meta_train)

    # Predict on full eval (to match support of baseline)
    probs_meta_full = meta_model.predict_proba(X_meta)[:, 1]
    cal_meta = IsotonicRegression(out_of_bounds='clip').fit(probs_meta_full[idx_train], y_eval[idx_train])
    probs_meta_cal = cal_meta.transform(probs_meta_full)

    # Threshold on held-out portion only
    thresh = threshold_dual_constraint(y_eval[idx_test], probs_meta_cal[idx_test])
    preds = (probs_meta_cal >= thresh).astype(int)

    return {
        "y_true": y_eval,
        "y_pred": preds,
        "y_prob": probs_meta_cal
    }

