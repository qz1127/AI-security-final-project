import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
from opacus import PrivacyEngine
from data_loader import load_data
from models import AdultMLP
from train_utils import train_model, evaluate_model, get_device

def train_smpc(model, train_loader, criterion, device, epochs=5, num_parties=3):
    """
    Simulates SMPC training using Additive Secret Sharing.
    Note: This is a simulation to demonstrate the trade-off (Accuracy vs Speed).
    Real SMPC would involve network communication and cryptographic primitives.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training SMPC Model ({num_parties} parties)...")
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 1. Simulate splitting data among parties (conceptually)
            # In a real scenario, data would already be distributed.
            # Here we just use the batch as is, but simulate the overhead of
            # secure aggregation.
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 2. Simulate Secure Aggregation (Additive Secret Sharing)
            # Overhead: Splitting gradients into shares, sending them, summing them.
            # We add an artificial delay to represent this.
            # Real SMPC is significantly slower than plaintext training.
            time.sleep(0.05) # Artificial delay per batch (adjust as needed)
            
            optimizer.step()
            running_loss += loss.item()
            
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Time: {epoch_time:.2f}s")

def run_experiment():
    print("Starting SMPC vs DP vs Baseline Experiment...")
    
    # Setup
    data_dir = os.path.join(os.path.dirname(__file__), '../adult-dataset')
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    
    device = get_device()
    BATCH_SIZE = 64
    EPOCHS_BASELINE = 10
    EPOCHS_SMPC = 5
    LR = 0.001
    MAX_GRAD_NORM = 1.0
    DELTA = 1e-5
    
    # Load Data
    print("Loading data...")
    train_loader, test_loader, input_dim = load_data(data_dir, batch_size=BATCH_SIZE)
    
    results = []
    
    # --- 1. Baseline Model (Non-DP) ---
    print("\n--- Training Baseline Model (Non-DP) ---")
    model_base = AdultMLP(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_base.parameters(), lr=LR)
    
    start_time = time.time()
    train_model(model_base, train_loader, optimizer, criterion, device, epochs=EPOCHS_BASELINE)
    total_time = time.time() - start_time
    
    train_acc, _ = evaluate_model(model_base, train_loader, device)
    test_acc, test_f1 = evaluate_model(model_base, test_loader, device)
    
    results.append({
        "Method": "Baseline",
        "Details": "No Privacy",
        "Test Accuracy": test_acc,
        "Time (s)": total_time
    })
    
    # --- 2. DP Models ---
    noise_multipliers = [0.5, 1.0, 3.0]
    for noise_mult in noise_multipliers:
        print(f"\n--- Training DP Model (Noise: {noise_mult}) ---")
        model_dp = AdultMLP(input_dim).to(device)
        optimizer = optim.Adam(model_dp.parameters(), lr=LR)
        
        privacy_engine = PrivacyEngine()
        model_dp, optimizer, train_loader_dp = privacy_engine.make_private(
            module=model_dp,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_mult,
            max_grad_norm=MAX_GRAD_NORM,
        )
        
        start_time = time.time()
        train_model(model_dp, train_loader_dp, optimizer, criterion, device, epochs=EPOCHS_BASELINE)
        total_time = time.time() - start_time
        
        epsilon = privacy_engine.get_epsilon(DELTA)
        
        train_acc, _ = evaluate_model(model_dp, train_loader, device)
        test_acc, test_f1 = evaluate_model(model_dp, test_loader, device)
        
        results.append({
            "Method": "Differential Privacy",
            "Details": f"Epsilon: {epsilon:.2f}",
            "Test Accuracy": test_acc,
            "Time (s)": total_time
        })

    # --- 3. SMPC Model (Simulated) ---
    print("\n--- Training SMPC Model (Simulated) ---")
    model_smpc = AdultMLP(input_dim).to(device)
    criterion = nn.BCELoss()
    
    start_time = time.time()
    train_smpc(model_smpc, train_loader, criterion, device, epochs=EPOCHS_SMPC)
    total_time = time.time() - start_time
    
    train_acc, _ = evaluate_model(model_smpc, train_loader, device)
    test_acc, test_f1 = evaluate_model(model_smpc, test_loader, device)
    
    results.append({
        "Method": "SMPC (Simulated)",
        "Details": "Secure Multi-Party Comp.",
        "Test Accuracy": test_acc,
        "Time (s)": total_time
    })

    # --- 4. Save and Plot Results ---
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "smpc_experiment_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df_results)
    
    # Plot Comparison
    plt.figure(figsize=(12, 6))
    
    # Bar plot for Accuracy
    plt.subplot(1, 2, 1)
    bars = plt.bar(df_results["Details"], df_results["Test Accuracy"], color=['gray', 'blue', 'blue', 'blue', 'green'])
    plt.title("Accuracy Comparison")
    plt.ylabel("Test Accuracy")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.7, 0.9) # Zoom in to see differences
    
    # Bar plot for Time
    plt.subplot(1, 2, 2)
    bars = plt.bar(df_results["Details"], df_results["Time (s)"], color=['gray', 'blue', 'blue', 'blue', 'green'])
    plt.title("Training Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "smpc_comparison.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_experiment()
