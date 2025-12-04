import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

NUMERICAL_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
]

class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(data_dir, batch_size=64):
    """
    Loads and preprocesses the Adult dataset.
    Returns train_loader, test_loader, and input_dim.
    """
    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")

    # Load data
    train_df = pd.read_csv(train_path, names=COLUMN_NAMES, sep=r',\s*', engine='python', na_values="?")
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES, sep=r',\s*', engine='python', na_values="?", skiprows=1) # Skip first line in test file which might be a header or weird line

    # Drop missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Clean target variable
    # Train: <=50K, >50K
    # Test: <=50K., >50K. (has dot at end)
    train_df['income'] = train_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    test_df['income'] = test_df['income'].apply(lambda x: 1 if x == '>50K.' else 0)

    # Separate features and target
    X_train = train_df.drop('income', axis=1)
    y_train = train_df['income'].values
    X_test = test_df.drop('income', axis=1)
    y_test = test_df['income'].values

    # Preprocessing pipeline
    # One-hot encode categorical, Normalize numerical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )

    # Fit on train, transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Create Datasets
    train_dataset = AdultDataset(X_train_processed, y_train)
    test_dataset = AdultDataset(X_test_processed, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train_processed.shape[1]

    return train_loader, test_loader, input_dim
