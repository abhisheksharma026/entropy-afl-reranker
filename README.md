## Entropy-AFL Reranker: Entropy-Regularized Focal Filtering and Calibrated Reranking for Rare Event Classification

This repository implements a three-stage modular architecture for rare-event classification, focused on high recall, precision, and calibrated probability estimates.
The pipeline is specifically designed for highly imbalanced datasets (e.g., loan delinquency, fraud detection).

### Our method combines:

- Entropy-Regularized Asymmetric Focal Loss (ER-AFL): Improving recall while maintaining calibrated predictions.
- Confidence-Guided Reranking via Calibrated XGBoost: Precision refinement over high-confidence samples.
- Meta-Calibrated Fusion: Final ensemble improving both recall and precision with enhanced calibration.

### Features
- Custom PyTorch implementation of Entropy-Regularized Asymmetric Focal Loss (ER-AFL)
- Confidence-thresholded reranking based on calibrated outputs
- Meta-calibrated ensemble fusion using XGBoost
- K-Fold Cross-Validation + Holdout Evaluation
- Full calibration analysis (ECE, Precision-Recall curves)
- Designed for Credit Card Fraud Detection and Lending Club Loan Delinquency datasets

📂 Project Structure
```javascript
src/
  entropy_reranker/
    model_pipeline.py   # Core ER-AFL Reranker pipeline
tests/
  credit_card_afl.py    # Cross-validation for Credit Card dataset
  lending_afl.py        # Cross-validation for Lending dataset
requirements.txt        # Exported dependencies (optional)
poetry.lock             # Dependency lock file (Poetry)
pyproject.toml          # Project metadata (Poetry)
.gitignore              # Ignored files and folders
LICENSE
README.md
```

### Clone the repository
- git clone https://github.com/abhisheksharma026/entropy-afl-reranker.git
- cd entropy-afl-reranker

### Install dependencies
poetry install

### Manually place the datasets inside a data/ folder:
```javascript
data/
  creditcard.csv
  lending.csv
```

### Run the pipeline
poetry run entropy-reranker

---
If you use this work, please cite it as: 

```
@misc{abhishek_sharma_2025_entropy_afl_reranker,
  author       = {Abhishek Sharma},
  title        = {Recall-Rich, Precision Respectful:A Meta-Ensemble Paradigm of Rare Outcomes},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15288525},
  url          = {https://doi.org/10.5281/zenodo.15288525}
}
```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15288525.svg)](https://doi.org/10.5281/zenodo.15288525)
