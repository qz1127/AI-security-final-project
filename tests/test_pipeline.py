import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_loader import load_data
from models import AdultMLP
from train_utils import train_model, evaluate_model, get_device

def test_pipeline():
    print("Testing pipeline...")
    data_dir = os.path.join(os.path.dirname(__file__), '../adult-dataset')
    
    # Test Data Loading
    print("Loading data...")
    train_loader, test_loader, input_dim = load_data(data_dir, batch_size=64)
    print(f"Data loaded. Input dim: {input_dim}")
    
    # Test Model Creation
    print("Creating model...")
    device = get_device()
    model = AdultMLP(input_dim).to(device)
    print("Model created.")
    
    # Test Training (1 epoch)
    print("Training for 1 epoch...")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, optimizer, criterion, device, epochs=1)
    print("Training completed.")
    
    # Test Evaluation
    print("Evaluating...")
    acc, f1 = evaluate_model(model, test_loader, device)
    print(f"Accuracy: {acc}, F1: {f1}")
    print("Pipeline test passed!")

if __name__ == "__main__":
    test_pipeline()
