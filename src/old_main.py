import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
from data_loader import load_data
from models import AdultMLP
from train_utils import train_model, evaluate_model, get_device, fit_temperature, train_student_with_distillation

def run_experiment():
    print("Starting Differential Privacy Experiment...")
    
    # Setup
    data_dir = os.path.join(os.path.dirname(__file__), '../adult-dataset')
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    
    device = get_device()
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    MAX_GRAD_NORM = 1.0
    DELTA = 1e-5
    
    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader, input_dim = load_data(data_dir, batch_size=BATCH_SIZE)
    
    results = []
    
    # --- 1. Baseline Model (Non-DP) ---
    print("\n--- Training Baseline Model (Non-DP) ---")
    model = AdultMLP(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    train_model(model, train_loader, optimizer, criterion, device, epochs=EPOCHS)
    
    train_acc, train_f1 = evaluate_model(model, train_loader, device)
    test_acc, test_f1 = evaluate_model(model, test_loader, device)
    gap = train_acc - test_acc
    
    print(f"Baseline - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Gap: {gap:.4f}")
    results.append({
        "Model": "Baseline (Non-DP)",
        "Epsilon": float('inf'),
        "Noise Multiplier": 0,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Gap": gap,
        "F1 Score": test_f1
    })
    
    # --- 2. DP Models ---
    # Noise multipliers to test: Low (0.5), Medium (1.0), High (3.0)
    noise_multipliers = [0.5, 1.0, 3.0]
    
    for noise_mult in noise_multipliers:
        print(f"\n--- Training DP Model (Noise: {noise_mult}) ---")
        model = AdultMLP(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader_dp = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_mult,
            max_grad_norm=MAX_GRAD_NORM,
        )
        
        train_model(model, train_loader_dp, optimizer, criterion, device, epochs=EPOCHS)
        
        epsilon = privacy_engine.get_epsilon(DELTA)
        print(f"Privacy Budget (Epsilon): {epsilon:.2f}")
        
        train_acc, train_f1 = evaluate_model(model, train_loader, device) # Use original loader for eval
        test_acc, test_f1 = evaluate_model(model, test_loader, device)
        gap = train_acc - test_acc
        
        print(f"DP (Noise {noise_mult}) - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Gap: {gap:.4f}")
        results.append({
            "Model": f"DP (Noise {noise_mult})",
            "Epsilon": epsilon,
            "Noise Multiplier": noise_mult,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Gap": gap,
            "F1 Score": test_f1
        })

    print("\n--- Training Regularized+Mixup+EarlyStopping ---")
    model_reg = AdultMLP(input_dim).to(device)
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=LR, weight_decay=1e-4)
    train_model(model_reg, train_loader, optimizer_reg, criterion, device, epochs=EPOCHS, val_loader=val_loader, patience=3, mixup_alpha=0.2)
    train_acc, train_f1 = evaluate_model(model_reg, train_loader, device)
    test_acc, test_f1 = evaluate_model(model_reg, test_loader, device)
    gap = train_acc - test_acc
    results.append({
        "Model": "Regularized+Mixup+EarlyStopping",
        "Epsilon": None,
        "Noise Multiplier": 0,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Gap": gap,
        "F1 Score": test_f1
    })

    print("\n--- Training Distilled+TempScaled ---")
    teacher = AdultMLP(input_dim).to(device)
    opt_t = optim.Adam(teacher.parameters(), lr=LR, weight_decay=1e-4)
    train_model(teacher, train_loader, opt_t, criterion, device, epochs=EPOCHS, val_loader=val_loader, patience=3)
    student = AdultMLP(input_dim).to(device)
    opt_s = optim.Adam(student.parameters(), lr=LR, weight_decay=1e-4)
    train_student_with_distillation(student, teacher, train_loader, opt_s, device, epochs=EPOCHS, alpha=0.5, temperature=2.0)
    T = fit_temperature(student, val_loader, device, max_iter=100)
    train_acc, train_f1 = evaluate_model(student, train_loader, device)
    test_acc, test_f1 = evaluate_model(student, test_loader, device)
    gap = train_acc - test_acc
    results.append({
        "Model": "Distilled+TempScaled",
        "Epsilon": None,
        "Noise Multiplier": 0,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Gap": gap,
        "F1 Score": test_f1
    })

    # --- 3. Save and Plot Results ---
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "experiment_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df_results)
    
    # Plot Accuracy vs Epsilon
    plt.figure(figsize=(10, 6))
    
    dp_results = df_results[df_results["Model"].str.startswith("DP")]
    baseline_acc = df_results[df_results["Model"] == "Baseline (Non-DP)"]["Test Accuracy"].values[0]
    
    plt.plot(dp_results["Epsilon"], dp_results["Test Accuracy"], marker='o', label='DP Models')
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline (Non-DP)')
    
    plt.xlabel('Privacy Budget (Epsilon)')
    plt.ylabel('Test Accuracy')
    plt.title('Privacy vs. Utility Trade-off')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(results_dir, "accuracy_vs_epsilon.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_experiment()
