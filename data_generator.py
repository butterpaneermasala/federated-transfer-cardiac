"""
Synthetic Data Generator for Hospital Datasets
Creates datasets with different feature dimensions to simulate
heterogeneous hospital data.
"""

import torch
import numpy as np
from sklearn.datasets import make_classification


def generate_hospital_data(num_samples, input_dim, num_classes=2, random_state=None):
    """
    Generates synthetic classification data for a hospital.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Number of input features (hospital-specific)
        num_classes: Number of classes for classification
        random_state: Random seed for reproducibility
    
    Returns:
        X: Features (torch.Tensor)
        y: Labels (torch.Tensor)
    """
    # Generate synthetic data
    X, y = make_classification(
        n_samples=num_samples,
        n_features=input_dim,
        n_informative=max(2, input_dim // 2),
        n_redundant=max(0, input_dim // 4),
        n_classes=num_classes,
        n_clusters_per_class=2,
        random_state=random_state,
        flip_y=0.1  # Add some noise
    )
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    return X, y


def split_data(X, y, train_ratio=0.8):
    """
    Splits data into train and test sets.
    
    Args:
        X: Features
        y: Labels
        train_ratio: Proportion of data for training
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    num_samples = len(X)
    num_train = int(num_samples * train_ratio)
    
    # Shuffle indices
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test


def generate_all_hospital_data(config):
    """
    Generates data for all hospitals defined in config.
    
    Args:
        config: Configuration object
    
    Returns:
        Dictionary mapping hospital_id to (X_train, y_train, X_test, y_test)
    """
    hospital_data = {}
    
    print("\n📊 Generating synthetic hospital datasets...")
    
    for idx, (hospital_id, hospital_config) in enumerate(config.HOSPITAL_CONFIGS.items()):
        # Generate data with different random seeds for each hospital
        X, y = generate_hospital_data(
            num_samples=hospital_config['num_samples'],
            input_dim=hospital_config['input_dim'],
            num_classes=config.NUM_CLASSES,
            random_state=42 + idx  # Different seed for each hospital
        )
        
        # Split into train/test
        X_train, y_train, X_test, y_test = split_data(X, y, train_ratio=0.8)
        
        hospital_data[hospital_id] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
        print(f"  ✓ {hospital_id}:")
        print(f"    - Features: {hospital_config['input_dim']}")
        print(f"    - Train samples: {len(X_train)}")
        print(f"    - Test samples: {len(X_test)}")
    
    return hospital_data
