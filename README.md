# AI Security Final Project: Differential Privacy on Adult Dataset

## Background & Motivation
Machine learning models are often trained on sensitive personal data, such as medical records or financial history. A critical security vulnerability in standard ML training is **unintended memorization**: models can overfit to specific training examples, allowing adversaries to infer whether a specific individual was in the training set (Membership Inference Attacks).

**Differential Privacy (DP)** provides a rigorous mathematical framework to quantify and limit this privacy leakage. By adding calibrated noise during the training process (DP-SGD), we can guarantee that the model's output does not depend significantly on any single individual's data. However, this comes at a cost: privacy often degrades model accuracy.

## Project Overview
This project empirically evaluates the **Privacy-Utility Trade-off** in tabular machine learning. We use the classic **UCI Adult (Census Income)** dataset to predict whether an individual earns more than $50K/year based on demographic features like age, education, and occupation.

### What We Do
1.  **Baseline Model**: We train a standard Multi-Layer Perceptron (MLP) without any privacy protections to establish a performance benchmark (~84-85% accuracy).
2.  **Differentially Private Training**: We retrain the same architecture using **DP-SGD** (via Meta's `Opacus` library) with varying noise levels. This allows us to observe how the "Privacy Budget" ($\epsilon$) affects model performance.
3.  **Attack Simulation**: We perform a simple **Membership Inference Attack (Gap Attack)** by measuring the difference between Training Accuracy and Test Accuracy. A large gap indicates overfitting and high privacy leakage; a small gap indicates better generalization and privacy.

### Key Results
- **High Privacy ($\epsilon \approx 0.17$)**: The model retains ~82% accuracy, demonstrating that we can achieve strong privacy guarantees with only a minor drop in utility.
- **Defense Effectiveness**: The DP models show near-zero overfitting gaps, effectively neutralizing the risk of membership inference attacks compared to the baseline.

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
