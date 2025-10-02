# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Tests (Optional but Recommended)
```bash
python test_components.py
```

### Step 3: Run Federated Training
```bash
python federated_trainer.py
```

---

## 📊 What You'll See

### Console Output
```
============================================================
GLOBAL ROUND 1/10
============================================================

🏥 Local Training Phase:
  [hospital_1] Training locally...
    Epoch 5/5 - Loss: 0.4523, Acc: 78.50%
  ✓ hospital_1 - Avg Loss: 0.4523, Avg Acc: 78.50%

🌐 Global Aggregation Phase:
  🔄 Aggregating weights from 2 hospitals...
  ✓ Global model updated

📊 Evaluation Phase:
  hospital_1 Test - Loss: 0.3419, Acc: 89.00%
  hospital_2 Test - Loss: 0.4962, Acc: 82.50%
```

### Generated Files
- `checkpoints/global_model_round_5.pt` - Checkpoint at round 5
- `checkpoints/global_model_round_10.pt` - Final checkpoint
- `training_results.png` - Training curves visualization

---

## ⚙️ Customize Your Setup

### Add More Hospitals

Edit `config.py`:
```python
HOSPITAL_CONFIGS = {
    'hospital_1': {
        'input_dim': 10,
        'adapter_hidden_dim': 64,
        'num_samples': 1000
    },
    'hospital_2': {
        'input_dim': 15,
        'adapter_hidden_dim': 64,
        'num_samples': 800
    },
    'hospital_3': {  # NEW HOSPITAL
        'input_dim': 20,
        'adapter_hidden_dim': 64,
        'num_samples': 1200
    }
}
```

### Change Training Parameters

```python
LOCAL_EPOCHS = 10       # More epochs per round
GLOBAL_ROUNDS = 20      # More federated rounds
BATCH_SIZE = 64         # Larger batches
LEARNING_RATE = 0.0005  # Different learning rate
```

### Modify Model Architecture

```python
LATENT_DIM = 256                     # Larger latent space
ENCODER_HIDDEN_DIMS = [512, 256, 128]  # Deeper encoder
NUM_CLASSES = 5                      # Multi-class classification
```

---

## 🔬 Use Your Own Data

### Option 1: Modify Data Generator

Edit `data_generator.py` to load your real datasets:

```python
def load_hospital_data(hospital_id):
    # Load your CSV, database, etc.
    X = load_features(hospital_id)
    y = load_labels(hospital_id)
    return X, y
```

### Option 2: Direct Data Loading

In your training script:

```python
from hospital import Hospital
from config import Config

config = Config()
hospital_config = config.HOSPITAL_CONFIGS['hospital_1']

# Create hospital
hospital = Hospital('hospital_1', config, hospital_config)

# Load your data
X_train = torch.tensor(your_features)  # Shape: [num_samples, num_features]
y_train = torch.tensor(your_labels)    # Shape: [num_samples]

# Set data
hospital.set_data(X_train, y_train)
```

---

## 📈 Understanding the Output

### Training Accuracy Plot (Left)
- Shows how well each hospital learns on its training data
- Should generally increase over rounds
- Both hospitals improve through collaboration

### Test Accuracy Plot (Right)
- Shows generalization to unseen test data
- More important metric than training accuracy
- May fluctuate but should trend upward

### Key Metrics
- **Loss**: Lower is better (how wrong predictions are)
- **Accuracy**: Higher is better (% correct predictions)
- **Improvement**: Final - Initial accuracy

---

## 🔐 Privacy Features

### What Stays Private
✅ Input adapters (never leave hospital)  
✅ Raw data (never transmitted)  
✅ Feature space (hospital-specific)  
✅ Local data statistics  

### What Gets Shared
📤 Encoder weights (after local training)  
📤 Head weights (after local training)  
📤 Number of samples (for weighted averaging)  

---

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size in `config.py`
```python
BATCH_SIZE = 16  # or smaller
```

### Issue: Training Too Slow
**Solution**: Reduce epochs or rounds
```python
LOCAL_EPOCHS = 3
GLOBAL_ROUNDS = 5
```

### Issue: Poor Accuracy
**Solutions**:
1. Increase model capacity (larger hidden dims)
2. More training rounds
3. Adjust learning rate
4. Check data quality

### Issue: Import Errors
**Solution**: Install dependencies
```bash
pip install torch numpy scikit-learn matplotlib
```

---

## 📚 Next Steps

### For Learning
1. Read `README.md` for detailed documentation
2. Review `IMPLEMENTATION_SUMMARY.md` for architecture details
3. Explore `models.py` to understand the neural networks
4. Check `test_components.py` for usage examples

### For Development
1. Implement custom aggregation methods in `server.py`
2. Add differential privacy mechanisms
3. Create custom model architectures in `models.py`
4. Add support for more data types (images, text, etc.)

### For Production
1. Replace synthetic data with real datasets
2. Add secure communication protocols
3. Implement authentication and authorization
4. Add monitoring and logging infrastructure
5. Deploy on distributed infrastructure

---

## 💡 Tips

### Best Practices
- Start with fewer rounds to test quickly
- Use test mode to verify everything works
- Monitor both training and test accuracy
- Save checkpoints frequently
- Visualize results to understand behavior

### Performance Optimization
- Use GPU if available (automatic with PyTorch)
- Increase batch size for faster training
- Use data loaders with multiple workers
- Profile code to find bottlenecks

### Experimentation
- Try different latent dimensions
- Experiment with encoder architectures
- Test various aggregation strategies
- Compare with centralized learning baseline

---

## 🎯 Common Use Cases

### Medical Research
- Different hospitals with different EHR systems
- Privacy-preserving collaborative learning
- Heterogeneous patient features

### Financial Services
- Banks with different customer data schemas
- Fraud detection across institutions
- Privacy-compliant model training

### IoT/Edge Computing
- Devices with different sensor configurations
- Distributed learning on edge devices
- Bandwidth-efficient model updates

---

## 📞 Need Help?

Check these resources:
1. `README.md` - Full documentation
2. `IMPLEMENTATION_SUMMARY.md` - Technical details
3. `test_components.py` - Code examples
4. PyTorch documentation - https://pytorch.org/docs/

---

**Happy Federated Learning! 🚀**
