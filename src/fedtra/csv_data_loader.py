"""
CSV Data Loader for Federated Learning
Automatically loads and preprocesses CSV datasets for hospitals.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


class CSVDataLoader:
    """
    Loads CSV data for hospitals and prepares it for federated learning.
    """
    
    def __init__(self, config):
        self.config = config
        self.hospital_data = {}
        self.scalers = {}  # Store scalers for each hospital
        self.label_encoders = {}  # Store label encoders
        
    def load_all_hospitals(self):
        """
        Loads data for all hospitals defined in config.
        
        Returns:
            Dictionary mapping hospital_id to data and metadata
        """
        print("\n" + "="*60)
        print("LOADING CSV DATASETS")
        print("="*60)
        # Optional: build a global label vocabulary to ensure consistent mapping
        global_label_vocab = None
        # Pre-scan label space across hospitals if labels are non-numeric
        collected_labels = set()
        for hospital_id, hospital_config in self.config.HOSPITAL_CONFIGS.items():
            csv_path = hospital_config['csv_path']
            target_column = hospital_config['target_column']
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            df_head = pd.read_csv(csv_path, usecols=[target_column])
            # If dtype is object, accumulate string labels
            if df_head[target_column].dtype == 'object':
                collected_labels.update(df_head[target_column].astype(str).unique().tolist())
        if len(collected_labels) > 0:
            # Deterministic ordering: sort labels
            global_label_vocab = sorted(collected_labels)

        for hospital_id, hospital_config in self.config.HOSPITAL_CONFIGS.items():
            print(f"\n📂 Loading {hospital_id}...")
            
            csv_path = hospital_config['csv_path']
            target_column = hospital_config['target_column']
            
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Load and process data
            data = self._load_and_preprocess(
                csv_path, 
                target_column, 
                hospital_id,
                global_label_vocab=global_label_vocab
            )
            
            # Store data and metadata
            self.hospital_data[hospital_id] = data
            
            # Update hospital config with detected dimensions
            hospital_config['input_dim'] = data['input_dim']
            hospital_config['num_samples'] = data['num_samples']
            
            print(f"  ✓ Loaded: {data['num_samples']} samples")
            print(f"  ✓ Features: {data['input_dim']}")
            print(f"  ✓ Train: {len(data['X_train'])}, Test: {len(data['X_test'])}")
            print(f"  ✓ Classes: {data['num_classes']} ({data['class_names']})")
        
        print("\n" + "="*60)
        print("✓ All datasets loaded successfully")
        print("="*60 + "\n")
        
        return self.hospital_data
    
    def _load_and_preprocess(self, csv_path, target_column, hospital_id, global_label_vocab=None):
        """
        Loads and preprocesses a single CSV file.
        
        Args:
            csv_path: Path to CSV file
            target_column: Name of the target/label column
            hospital_id: Identifier for the hospital
            global_label_vocab: Optional list defining a consistent global label order
        
        Returns:
            Dictionary with processed data and metadata
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"  - Raw data shape: {df.shape}")
        
        # Handle missing values
        if self.config.HANDLE_MISSING == 'drop':
            df = df.dropna()
            print(f"  - After dropping NaN: {df.shape}")
        elif self.config.HANDLE_MISSING == 'mean':
            df = df.fillna(df.mean())
        elif self.config.HANDLE_MISSING == 'median':
            df = df.fillna(df.median())
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {csv_path}")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle non-numeric features (encode categorical variables)
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"  - Encoding categorical column: {col}")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Convert to numpy first
        X = X.values.astype(np.float32)
        y_values = y.values if hasattr(y, 'values') else y
        
        # Encode labels consistently across hospitals
        if y.dtype == 'object' or not np.issubdtype(y_values.dtype, np.integer):
            print(f"  - Encoding target labels")
            labels_str = y.astype(str)
            if global_label_vocab is not None:
                # Map using global vocabulary
                vocab_index = {lab: i for i, lab in enumerate(global_label_vocab)}
                unknown = set(labels_str.unique()) - set(global_label_vocab)
                if len(unknown) > 0:
                    raise ValueError(f"Unknown labels found in {csv_path}: {sorted(list(unknown))} not in global label vocab {global_label_vocab}")
                y_values = labels_str.map(vocab_index).values.astype(np.int64)
                class_names = list(global_label_vocab)
                # Store a no-op encoder info
                self.label_encoders[hospital_id] = vocab_index
            else:
                # Fallback to per-hospital encoding (single-site case)
                le = LabelEncoder()
                y_values = le.fit_transform(labels_str)
                self.label_encoders[hospital_id] = le
                class_names = list(le.classes_)
        else:
            y_values = y_values.astype(np.int64)
            class_names = list(np.unique(y_values))
        
        y = y_values
        
        # Normalize features
        if self.config.NORMALIZE_FEATURES:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scalers[hospital_id] = scaler
            print(f"  - Features normalized (StandardScaler)")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1 - self.config.TRAIN_TEST_SPLIT,
            random_state=42,
            stratify=y
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Get metadata and validations
        input_dim = X.shape[1]
        num_samples = len(X)
        num_classes = len(np.unique(y))
        if getattr(self.config, 'ENFORCE_NUM_CLASSES', False):
            if num_classes != self.config.NUM_CLASSES:
                raise ValueError(
                    f"Derived num_classes={num_classes} from {csv_path} does not match Config.NUM_CLASSES={self.config.NUM_CLASSES}"
                )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'input_dim': input_dim,
            'num_samples': num_samples,
            'num_classes': num_classes,
            'class_names': class_names,
            'feature_names': list(df.drop(columns=[target_column]).columns)
        }
    
    def get_hospital_data(self, hospital_id):
        """
        Returns data for a specific hospital.
        """
        if hospital_id not in self.hospital_data:
            raise ValueError(f"Hospital {hospital_id} not found")
        return self.hospital_data[hospital_id]
    
    def get_feature_info(self, hospital_id):
        """
        Returns feature information for a hospital.
        """
        data = self.get_hospital_data(hospital_id)
        return {
            'feature_names': data['feature_names'],
            'input_dim': data['input_dim'],
            'num_samples': data['num_samples']
        }
    
    def print_data_summary(self):
        """
        Prints a summary of all loaded datasets.
        """
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        for hospital_id, data in self.hospital_data.items():
            print(f"\n{hospital_id}:")
            print(f"  Features: {data['input_dim']}")
            print(f"  Feature names: {', '.join(data['feature_names'][:5])}...")
            print(f"  Total samples: {data['num_samples']}")
            print(f"  Train samples: {len(data['X_train'])}")
            print(f"  Test samples: {len(data['X_test'])}")
            print(f"  Classes: {data['class_names']}")
            print(f"  Train label distribution: {torch.bincount(data['y_train']).tolist()}")
            print(f"  Test label distribution: {torch.bincount(data['y_test']).tolist()}")
        
        print("\n" + "="*60 + "\n")
