CONFIG = {
    "SEED": 42,
    "N_SPLITS": 5,
    "CALIBRATION_THRESHOLD": 0.9,  # Top 10% filtering, percentile-based
    "DATA_PATH_LENDING": "data/lending.csv",
    "TARGET_COLUMN_LENDING": "SeriousDlqin2yrs",
    "DATA_PATH_CC": "data/creditcard.csv",
    "TARGET_COLUMN_CC": "Class",

    # Loss and Optimization
    "GAMMA_POS": 2.0,
    "GAMMA_NEG": 1.0,
    "POS_WEIGHT": 15.0,
    "LAMBDA_ENTROPY": 1e-5,
    "EPOCHS": 100,

    # Constraints for thresholding
    "MIN_PRECISION": 0.2,
    "MIN_RECALL": 0.1,

    # Models
    "LR_MAX_ITER": 1000,
    "RERANKER_N_ESTIMATORS": 1000,
    "META_N_ESTIMATORS": 500,
}
