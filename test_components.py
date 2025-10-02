"""
Unit tests for federated learning components.
Tests individual components before full training.
"""

import torch
from models import Encoder, GlobalHead, InputAdapter, HospitalModel
from server import GlobalServer
from hospital import Hospital
from config import Config


def test_models():
    """Test model architectures."""
    print("\n" + "="*60)
    print("Testing Model Components")
    print("="*60)
    
    config = Config()
    
    # Test Input Adapter
    print("\n1. Testing Input Adapter...")
    adapter = InputAdapter(input_dim=10, latent_dim=128, hidden_dim=64)
    x = torch.randn(4, 10)  # Batch of 4 samples
    out = adapter(x)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"
    print(f"   ✓ Input: {x.shape} → Output: {out.shape}")
    
    # Test Encoder
    print("\n2. Testing Encoder...")
    encoder = Encoder(latent_dim=128, hidden_dims=[256, 128])
    out = encoder(out)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"
    print(f"   ✓ Output shape: {out.shape}")
    
    # Test Head
    print("\n3. Testing Global Head...")
    head = GlobalHead(input_dim=128, num_classes=2)
    out = head(out)
    assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"
    print(f"   ✓ Output shape: {out.shape}")
    
    # Test Full Hospital Model
    print("\n4. Testing Complete Hospital Model...")
    adapter = InputAdapter(input_dim=10, latent_dim=128)
    encoder = Encoder(latent_dim=128, hidden_dims=[256, 128])
    head = GlobalHead(input_dim=128, num_classes=2)
    model = HospitalModel(adapter, encoder, head)
    
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"
    print(f"   ✓ End-to-end: {x.shape} → {out.shape}")
    
    print("\n✓ All model tests passed!")


def test_weight_extraction():
    """Test weight extraction and loading."""
    print("\n" + "="*60)
    print("Testing Weight Extraction & Loading")
    print("="*60)
    
    config = Config()
    
    # Create model
    adapter = InputAdapter(input_dim=10, latent_dim=128)
    encoder = Encoder(latent_dim=128, hidden_dims=[256, 128])
    head = GlobalHead(input_dim=128, num_classes=2)
    model = HospitalModel(adapter, encoder, head)
    
    # Get shared parameters
    print("\n1. Extracting shared parameters...")
    shared_params = model.get_shared_parameters()
    print(f"   ✓ Extracted shared parameters")
    
    # Verify structure
    assert 'encoder' in shared_params, "Encoder should be in shared params"
    assert 'head' in shared_params, "Head should be in shared params"
    assert 'adapter' not in shared_params, "Adapter should not be in shared params"
    print("   ✓ Input adapter excluded (private)")
    
    # Verify encoder and head are included
    encoder_params = len(shared_params['encoder'])
    head_params = len(shared_params['head'])
    print(f"   ✓ Encoder parameters: {encoder_params}")
    print(f"   ✓ Head parameters: {head_params}")
    
    # Test loading shared parameters
    print("\n2. Loading shared parameters...")
    model.set_shared_parameters(shared_params)
    print("   ✓ Parameters loaded successfully")
    
    print("\n✓ Weight extraction tests passed!")


def test_server():
    """Test global server functionality."""
    print("\n" + "="*60)
    print("Testing Global Server")
    print("="*60)
    
    config = Config()
    server = GlobalServer(config)
    
    # Get global weights
    print("\n1. Getting global weights...")
    global_weights = server.get_global_weights()
    print(f"   ✓ Retrieved global weights (encoder + head)")
    
    # Test aggregation
    print("\n2. Testing weight aggregation...")
    # Simulate 2 hospitals with same weights
    hospital_weights = [global_weights, global_weights]
    hospital_samples = [1000, 800]
    
    aggregated = server.aggregate_weights(hospital_weights, hospital_samples)
    print(f"   ✓ Aggregated weights from {len(hospital_weights)} hospitals")
    
    # Update global model
    print("\n3. Updating global model...")
    server.update_global_model(aggregated)
    print("   ✓ Global model updated")
    
    print("\n✓ Server tests passed!")


def test_hospital():
    """Test hospital client functionality."""
    print("\n" + "="*60)
    print("Testing Hospital Client")
    print("="*60)
    
    config = Config()
    hospital_config = config.HOSPITAL_CONFIGS['hospital_1']
    
    # Create hospital
    print("\n1. Creating hospital...")
    hospital = Hospital('hospital_1', config, hospital_config)
    print("   ✓ Hospital created")
    
    # Set data
    print("\n2. Setting local data...")
    X = torch.randn(100, hospital_config['input_dim'])
    y = torch.randint(0, 2, (100,))
    hospital.set_data(X, y)
    print("   ✓ Data loaded")
    
    # Get shared weights
    print("\n3. Extracting shared weights...")
    weights = hospital.get_shared_weights()
    print(f"   ✓ Extracted shared weights (encoder + head)")
    
    # Receive global weights
    print("\n4. Receiving global weights...")
    hospital.receive_global_weights(weights)
    print("   ✓ Global weights received")
    
    print("\n✓ Hospital tests passed!")


def test_heterogeneous_features():
    """Test that hospitals with different feature dimensions work."""
    print("\n" + "="*60)
    print("Testing Heterogeneous Feature Dimensions")
    print("="*60)
    
    config = Config()
    
    # Create two hospitals with different input dimensions
    h1_config = {'input_dim': 10, 'adapter_hidden_dim': 64, 'num_samples': 100}
    h2_config = {'input_dim': 15, 'adapter_hidden_dim': 64, 'num_samples': 100}
    
    print("\n1. Creating hospitals with different feature counts...")
    h1 = Hospital('hospital_1', config, h1_config)
    h2 = Hospital('hospital_2', config, h2_config)
    print(f"   ✓ Hospital 1: {h1_config['input_dim']} features")
    print(f"   ✓ Hospital 2: {h2_config['input_dim']} features")
    
    # Set different data
    print("\n2. Setting heterogeneous data...")
    X1 = torch.randn(100, 10)  # 10 features
    y1 = torch.randint(0, 2, (100,))
    h1.set_data(X1, y1)
    
    X2 = torch.randn(100, 15)  # 15 features
    y2 = torch.randint(0, 2, (100,))
    h2.set_data(X2, y2)
    print("   ✓ Different datasets loaded")
    
    # Get weights from both
    print("\n3. Extracting weights from both hospitals...")
    w1 = h1.get_shared_weights()
    w2 = h2.get_shared_weights()
    
    # Verify they have the same keys (encoder + head)
    assert set(w1.keys()) == set(w2.keys()), "Shared weights should have same structure!"
    print(f"   ✓ Both hospitals have same weight structure")
    print("   ✓ Weight structures match (encoder + head)")
    
    # Verify shapes match for encoder and head
    print("\n4. Verifying weight shapes match...")
    for component in ['encoder', 'head']:
        for key in w1[component].keys():
            assert w1[component][key].shape == w2[component][key].shape, \
                f"Shape mismatch for {component}.{key}"
    print("   ✓ All weight shapes compatible for aggregation")
    
    print("\n✓ Heterogeneous feature tests passed!")


def run_all_tests():
    """Run all component tests."""
    print("\n" + "="*60)
    print("FEDERATED LEARNING COMPONENT TESTS")
    print("="*60)
    
    try:
        test_models()
        test_weight_extraction()
        test_server()
        test_hospital()
        test_heterogeneous_features()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour federated learning system is ready to use.")
        print("Run: python federated_trainer.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
