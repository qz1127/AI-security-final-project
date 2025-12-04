# AI Security Final Project: Differential Privacy on Adult Dataset

This project evaluates the trade-off between privacy and utility in machine learning models using Differential Privacy (DP). We train a standard classifier and Differentially Private SGD (DP-SGD) models on the UCI Adult Income dataset to predict income levels (>50K vs <=50K). We analyze the accuracy and the privacy guarantees provided by different noise levels, demonstrating how DP mitigates membership inference attacks.

## Source of Dataset
https://archive.ics.uci.edu/dataset/2/adult

## How to Run

### Install Dependencies
```bash
pip install torch opacus pandas scikit-learn matplotlib
```

### Run Experiment
```bash
python src/main.py
```

This script will:
1. Load and preprocess the Adult dataset.
2. Train a baseline non-private model.
3. Train three DP models with varying privacy budgets (Low, Medium, High noise).
4. Save results to `results/experiment_results.csv`.
5. Generate a trade-off plot at `results/accuracy_vs_epsilon.png`.
