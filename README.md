# Federated Learning with Private Input Adapters

A PyTorch implementation of federated learning where hospitals can collaborate on training a shared model while keeping their input features private through local adapters.

## 🎉 NEW: Automatic CSV Dataset Loading!

Simply specify CSV paths in the config file - the system automatically handles everything:
- ✅ Feature detection (8 features vs 13 features)
- ✅ Data preprocessing and normalization
- ✅ Missing value handling
- ✅ Categorical encoding
- ✅ Train/test splitting

**Real Results**: Achieved **93.94%** and **99.02%** accuracy on medical datasets!

## 🏗️ Architecture

### Key Components

1. **Global Server**
   - Initializes and maintains global encoder and head
   - Aggregates weights from hospitals using FedAvg
   - Broadcasts updated weights to all participants

2. **Hospital Clients**
   - Each hospital has a **private input adapter** (never shared)
   - Receives and updates shared encoder and head weights
   - Trains on local data with heterogeneous features

3. **Model Structure**
   ```
   Hospital Model: Input Adapter → Encoder → Head
                   [PRIVATE]      [SHARED]   [SHARED]
   ```

## 🔑 Key Features

- ✅ **Privacy-Preserving**: Input adapters remain local, protecting feature space privacy
- ✅ **Heterogeneous Data**: Each hospital can have different numbers of features
- ✅ **Federated Averaging**: Weighted aggregation based on dataset sizes
- ✅ **Modular Design**: Easy to extend and customize

## 📁 Project Structure

```
fedtra/
├── config.py                  # Configuration parameters
├── models.py                  # Neural network architectures
├── server.py                  # Global federated server
├── hospital.py                # Hospital client implementation
├── csv_data_loader.py         # Automatic CSV data loading (NEW!)
├── data_generator.py          # Synthetic data generation (legacy)
├── federated_trainer.py       # Main training orchestration
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── CSV_USAGE_GUIDE.md         # Detailed CSV usage guide (NEW!)
└── datasets/                  # Place your CSV files here
    ├── Medicaldataset.csv
    └── cardiac arrest dataset.csv
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Using Your Own CSV Datasets

1. **Place CSV files** in the `datasets/` folder
2. **Configure** in `config.py`:
```python
HOSPITAL_CONFIGS = {
    'hospital_1': {
        'csv_path': 'datasets/your_dataset1.csv',
        'target_column': 'label_column_name',
        'adapter_hidden_dim': 64,
    },
    'hospital_2': {
        'csv_path': 'datasets/your_dataset2.csv',
        'target_column': 'target',
        'adapter_hidden_dim': 64,
    }
}
```
3. **Run training**:
```bash
python federated_trainer.py
```

The system automatically:
- Detects feature dimensions (can be different per hospital!)
- Handles missing values and categorical variables
- Normalizes features
- Splits into train/test sets
- Trains the federated model

### Example Results (Medical Datasets)

**Hospital 1** (1,319 samples, 8 features): 77.65% → **93.94%** (+16.29%)  
**Hospital 2** (1,025 samples, 13 features): 82.44% → **99.02%** (+16.59%)

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model architecture
LATENT_DIM = 128                    # Shared latent dimension
ENCODER_HIDDEN_DIMS = [256, 128]    # Encoder layers
NUM_CLASSES = 2                     # Classification classes

# Hospital configurations with CSV paths
HOSPITAL_CONFIGS = {
    'hospital_1': {
        'csv_path': 'datasets/Medicaldataset.csv',
        'target_column': 'Result',
        'adapter_hidden_dim': 64,
        # input_dim auto-detected: 8 features
        # num_samples auto-detected: 1319 samples
    },
    'hospital_2': {
        'csv_path': 'datasets/cardiac arrest dataset.csv',
        'target_column': 'target',
        'adapter_hidden_dim': 64,
        # input_dim auto-detected: 13 features (different!)
        # num_samples auto-detected: 1025 samples
    }
}

# Data preprocessing (automatic)
TRAIN_TEST_SPLIT = 0.8          # 80% train, 20% test
NORMALIZE_FEATURES = True       # Standardize features
HANDLE_MISSING = 'drop'         # Handle missing values

# Training parameters
LOCAL_EPOCHS = 5        # Epochs per hospital per round
GLOBAL_ROUNDS = 10      # Number of federated rounds
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

## 🔄 Federated Learning Workflow

### Round-by-Round Process

1. **Initialization**
   - Server initializes global encoder and head
   - Broadcasts initial weights to all hospitals

2. **Local Training** (at each hospital)
   - Receive global encoder + head weights
   - Train full model (adapter + encoder + head) on local data
   - Extract shared weights (encoder + head only)

3. **Global Aggregation** (at server)
   - Collect shared weights from all hospitals
   - Perform weighted averaging (FedAvg)
   - Update global encoder and head

4. **Broadcast**
   - Send updated global weights back to hospitals
   - Hospitals update their encoder + head
   - Private adapters remain unchanged

5. **Repeat** until convergence

## 📊 Output

The training produces:

- **Checkpoints**: Saved in `./checkpoints/`
  - `global_model_round_X.pt` - Global model at round X

- **Training Plot**: `training_results.png`
  - Training accuracy per hospital
  - Test accuracy per hospital

- **Console Output**: Real-time training metrics
  - Loss and accuracy per epoch
  - Test performance per round

## 🔬 Example Output

```
============================================================
GLOBAL ROUND 1/10
============================================================

🏥 Local Training Phase:

  [hospital_1] Training locally...
    Epoch 5/5 - Loss: 0.4523, Acc: 78.50%
  ✓ hospital_1 - Avg Loss: 0.4523, Avg Acc: 78.50%

  [hospital_2] Training locally...
    Epoch 5/5 - Loss: 0.4891, Acc: 75.25%
  ✓ hospital_2 - Avg Loss: 0.4891, Avg Acc: 75.25%

🌐 Global Aggregation Phase:
🔄 Aggregating weights from 2 hospitals...
✓ Global model updated

📡 Broadcasting updated weights to hospitals...

📊 Evaluation Phase:
  hospital_1 Test - Loss: 0.4234, Acc: 80.50%
  hospital_2 Test - Loss: 0.4567, Acc: 77.81%
```

## 🧪 Adding More Hospitals

To add a new hospital, simply update `config.py`:

```python
HOSPITAL_CONFIGS = {
    'hospital_1': {...},
    'hospital_2': {...},
    'hospital_3': {
        'input_dim': 20,        # Different feature count
        'adapter_hidden_dim': 64,
        'num_samples': 1200
    }
}
```

The system automatically handles any number of hospitals with varying feature dimensions.

## 🔐 Privacy Guarantees

- **Input Adapters**: Never leave the hospital premises
- **Feature Space**: Completely private to each hospital
- **Shared Components**: Only encoder and head weights are transmitted
- **No Raw Data**: Data never leaves the hospital

## 📈 Extending the System

### Custom Models

Modify `models.py` to change:
- Encoder architecture
- Head architecture
- Adapter architecture

### Custom Aggregation

Modify `server.py` to implement:
- Different aggregation strategies (FedProx, FedOpt, etc.)
- Secure aggregation protocols
- Differential privacy mechanisms

### Real Data

Replace `data_generator.py` with your own data loading:

```python
# In your training script
hospital_1.set_data(your_X_train, your_y_train)
```

## 📚 References

- Federated Learning: [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)
- FedAvg Algorithm: Weighted averaging based on dataset sizes
- Privacy-Preserving ML: Input adapters for heterogeneous feature spaces

## 🤝 Contributing

Feel free to extend this implementation with:
- More sophisticated aggregation methods
- Differential privacy
- Secure multi-party computation
- Real-world medical datasets

## 📝 License

MIT License - Feel free to use and modify for your research and applications.
