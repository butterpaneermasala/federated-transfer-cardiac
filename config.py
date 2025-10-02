"""
Configuration file for Federated Learning with Private Input Adapters
"""

class Config:
    # Global Model Architecture
    LATENT_DIM = 128  # Fixed latent dimension for encoder output
    ENCODER_HIDDEN_DIMS = [256, 128]  # Hidden layers in encoder
    NUM_CLASSES = 2  # Binary classification (adjust as needed)
    
    # Hospital-specific configurations with CSV paths
    HOSPITAL_CONFIGS = {
        'hospital_1': {
            'csv_path': 'datasets/Medicaldataset.csv',
            'target_column': 'Result',  # Column name for labels
            'adapter_hidden_dim': 64,
            # input_dim will be auto-detected from CSV
            # num_samples will be auto-detected from CSV
        },
        'hospital_2': {
            'csv_path': 'datasets/cardiac arrest dataset.csv',
            'target_column': 'target',  # Column name for labels
            'adapter_hidden_dim': 64,
            # input_dim will be auto-detected from CSV
            # num_samples will be auto-detected from CSV
        }
    }
    
    # Data preprocessing
    TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
    NORMALIZE_FEATURES = True  # Standardize features
    HANDLE_MISSING = 'drop'  # Options: 'drop', 'mean', 'median'
    
    # Training parameters
    LOCAL_EPOCHS = 5  # Epochs per local training round
    GLOBAL_ROUNDS = 10  # Number of federated rounds
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Federated Learning
    AGGREGATION_METHOD = 'fedavg'  # Federated Averaging
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_DIR = './checkpoints'
