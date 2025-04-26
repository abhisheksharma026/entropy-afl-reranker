import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from entropy_reranker.model_pipeline import evaluate_lr, evaluate_xgb, evaluate_reranker

def get_dummy_data(n_samples=5000, n_features=10, imbalance_ratio=0.1, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        weights=[1 - imbalance_ratio],
        flip_y=0,
        random_state=random_state
    )
    return train_test_split(X, y, test_size=0.3, random_state=random_state)

def test_evaluate_lr():
    X_train, X_test, y_train, y_test = get_dummy_data()
    preds, probs = evaluate_lr(X_train, y_train, X_test, y_test)
    assert len(preds) == len(y_test)
    assert len(probs) == len(y_test)
    assert (0 <= np.min(probs)) and (np.max(probs) <= 1)

def test_evaluate_xgb():
    X_train, X_test, y_train, y_test = get_dummy_data()
    preds, probs = evaluate_xgb(X_train, y_train, X_test, y_test)
    assert len(preds) == len(y_test)
    assert len(probs) == len(y_test)
    assert (0 <= np.min(probs)) and (np.max(probs) <= 1)

def test_evaluate_reranker():
    X_train, X_test, y_train, y_test = get_dummy_data()
    preds, probs = evaluate_reranker(X_train, y_train, X_test, y_test)
    assert len(preds) == len(y_test)
    assert len(probs) == len(y_test)
    assert (0 <= np.min(probs)) and (np.max(probs) <= 1)
