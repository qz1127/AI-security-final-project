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

## Experimental Results

### Privacy vs. Utility (Differential Privacy)
*   **Baseline Accuracy**: ~84.8% (Gap: 0.5%)
*   **High Privacy ($\epsilon \approx 0.17$)**: ~82.1% Accuracy (Gap: -0.1%)
*   **Result**: Strong differential privacy guarantees can be achieved with a relatively minor drop in accuracy (~2.7%). The "Gap Attack" showed that DP models successfully eliminated overfitting.

### Privacy vs. Efficiency (SMPC)
*   **SMPC Accuracy**: ~84.6% (Comparable to Baseline)
*   **SMPC Time**: ~129.6s (vs. ~1.9s for Baseline)
*   **Result**: SMPC preserves original model accuracy perfectly but incurs a massive computational overhead (~70x slower in our simulation).

### Optimization Techniques
*   **Regularization & Distillation**: Achieved ~84.2% - 84.4% accuracy.
*   **Result**: These methods improved robustness but did not significantly outperform the standard baseline, serving primarily as stabilization techniques.

## Extended Privacy & Optimization Techniques

This project explores several advanced techniques beyond basic Differential Privacy:

### 1. Differential Privacy (DP-SGD)
We use **DP-SGD** to clip gradients and significantly add noise during training. This ensures that the model's weights do not memorize any single training example.
-   **Key Library**: `Opacus` (by Meta).
-   **Metrics**: We track $\epsilon$ (Privacy Budget) vs. Accuracy.

### 2. Secure Multi-Party Computation (SMPC)
We implement a **simulation** of SMPC (specifically Additive Secret Sharing) to demonstrate the privacy-efficiency trade-off.
-   **Concept**: Data is split among $N$ parties. No single party sees the raw data. They compute partial gradients and securely aggregate them.
-   **Simulation**: We simulate the network overhead and cryptographic delays to show how SMPC is much slower than plaintext training but offers strong privacy guarantees without accuracy loss (unlike DP).

### 3. Optimization Techniques
To recover some of the accuracy lost due to DP or to improve generalization, we implement:
-   **Mixup Regularization**: Training on linear combinations of examples to improve robustness.
-   **Knowledge Distillation**: Training a "Student" model to mimic a larger "Teacher" model, often combined with **Temperature Scaling** to calibrate probabilities.

## Source of Dataset
**Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.**

https://archive.ics.uci.edu/dataset/2/adult

## How to Run

### Install Dependencies
```bash
pip install torch opacus pandas scikit-learn matplotlib
```

### 1. Run Differential Privacy & Optimization Experiment
This script runs the core DP experiment (varying noise levels) and the optimization techniques (Mixup, Distillation).
```bash
python src/main.py
```
**Output**:
-   `results/experiment_results.csv`: Metrics for Baseline, DP (Low/Med/High noise), and Optimization models.
-   `results/accuracy_vs_epsilon.png`: Trade-off plot.

### 2. Run SMPC Simulation Experiment
This script runs the simulation comparing the training time and accuracy of SMPC vs. Baseline vs. DP.
```bash
python src/SMPC_main.py
```
**Output**:
-   `results/smpc_experiment_results.csv`: Comparison metrics (Time vs Accuracy).
-   `results/smpc_comparison.png`: Bar charts visualizing the trade-off.

### Project Structure
-   `src/main.py`: Main driver for Differential Privacy and Optimization experiments.
-   `src/SMPC_main.py`: Driver for Secure Multi-Party Computation simulation.
-   `src/models.py`: PyTorch model definitions (MLP).
-   `src/train_utils.py`: Helper functions for training, evaluation, and distillation.
-   `src/data_loader.py`: Data preprocessing for the Adult dataset.
