"""
Hospital Client for Federated Learning
- Maintains private input adapter
- Trains on local data
- Uploads only shared weights (encoder + head)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from .models import InputAdapter, Encoder, GlobalHead, HospitalModel


class Hospital:
    """
    Represents a hospital participating in federated learning.
    Each hospital has:
    - Private input adapter (never shared)
    - Shared encoder (synchronized with global)
    - Shared head (synchronized with global)
    """
    def __init__(self, hospital_id, config, hospital_config, device='cpu'):
        self.hospital_id = hospital_id
        self.config = config
        self.hospital_config = hospital_config
        self.device = device
        
        # Create private input adapter
        self.input_adapter = InputAdapter(
            input_dim=hospital_config['input_dim'],
            latent_dim=config.LATENT_DIM,
            hidden_dim=hospital_config['adapter_hidden_dim']
        ).to(device)
        
        # Create encoder (will be synchronized with global)
        self.encoder = Encoder(
            latent_dim=config.LATENT_DIM,
            hidden_dims=config.ENCODER_HIDDEN_DIMS
        ).to(device)
        
        # Create head (will be synchronized with global)
        self.head = GlobalHead(
            input_dim=config.ENCODER_HIDDEN_DIMS[-1],
            num_classes=config.NUM_CLASSES
        ).to(device)
        
        # Complete model
        self.model = HospitalModel(
            self.input_adapter,
            self.encoder,
            self.head
        ).to(device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # Data
        self.train_loader = None
        self.num_samples = 0
        
        print(f"✓ {hospital_id} initialized")
        print(f"  - Input dim: {hospital_config['input_dim']}")
        print(f"  - Private adapter: {hospital_config['input_dim']} → {config.LATENT_DIM}")
    
    def set_data(self, X, y):
        """
        Sets the local training data for this hospital.
        
        Args:
            X: Features (torch.Tensor)
            y: Labels (torch.Tensor)
        """
        dataset = TensorDataset(X, y)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        self.num_samples = len(X)
        print(f"  - {self.hospital_id}: {self.num_samples} samples loaded")
    
    def receive_global_weights(self, global_weights):
        """
        Receives and applies global encoder + head weights from server.
        Keeps the private input adapter unchanged.
        """
        self.model.set_shared_parameters(global_weights)
        print(f"  - {self.hospital_id}: Global weights received")
    
    def train_local(self, epochs):
        """
        Trains the model on local data for specified epochs.
        Updates all parameters (adapter, encoder, head).
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        # Capture a snapshot of the current shared params for FedProx
        mu = getattr(self.config, 'FEDPROX_MU', 0.0)
        if mu > 0:
            with torch.no_grad():
                # Save initial shared parameters (encoder + head) for proximal term
                init_shared = self.model.get_shared_parameters()
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                # FedProx proximal term: (mu/2) * ||w - w_global||^2 over shared params
                if mu > 0:
                    prox = 0.0
                    current_shared = self.model.get_shared_parameters()
                    for comp in ['encoder', 'head']:
                        for k in current_shared[comp].keys():
                            diff = current_shared[comp][k] - init_shared[comp][k]
                            prox = prox + (diff.pow(2).sum())
                    loss = loss + (mu / 2.0) * prox
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                epoch_total += target.size(0)
                epoch_correct += predicted.eq(target).sum().item()
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
            
            if (epoch + 1) % self.config.LOG_INTERVAL == 0 or epoch == epochs - 1:
                avg_loss = epoch_loss / len(self.train_loader)
                accuracy = 100. * epoch_correct / epoch_total
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        avg_loss = total_loss / (epochs * len(self.train_loader))
        avg_accuracy = 100. * correct / total
        
        return avg_loss, avg_accuracy
    
    def get_shared_weights(self):
        """
        Returns only the shared weights (encoder + head).
        The private input adapter is NOT included.
        """
        return self.model.get_shared_parameters()
    
    def evaluate(self, test_loader):
        """
        Evaluates the model on test data.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_adapter(self, save_path):
        """
        Saves the private input adapter (for hospital's own use).
        """
        torch.save(self.input_adapter.state_dict(), save_path)
        print(f"✓ {self.hospital_id}: Private adapter saved to {save_path}")
    
    def load_adapter(self, load_path):
        """
        Loads a saved private input adapter.
        """
        self.input_adapter.load_state_dict(torch.load(load_path))
        print(f"✓ {self.hospital_id}: Private adapter loaded from {load_path}")
