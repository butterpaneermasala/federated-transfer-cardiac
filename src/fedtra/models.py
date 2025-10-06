"""
Neural Network Models for Federated Learning
- Encoder: Shared across all hospitals
- Head: Global classifier/regressor
- InputAdapter: Private to each hospital
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Shared encoder that transforms latent representations.
    This is shared globally across all hospitals.
    """
    def __init__(self, latent_dim, hidden_dims):
        super(Encoder, self).__init__()
        
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Replace BatchNorm with LayerNorm to reduce cross-client BN drift
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.encoder(x)


class GlobalHead(nn.Module):
    """
    Global classification/regression head.
    Shared across all hospitals.
    """
    def __init__(self, input_dim, num_classes):
        super(GlobalHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)


class InputAdapter(nn.Module):
    """
    Private input adapter for each hospital.
    Maps hospital-specific features to the shared latent dimension.
    This remains LOCAL and is never shared.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super(InputAdapter, self).__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # Replace BatchNorm with LayerNorm
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.adapter(x)


class HospitalModel(nn.Module):
    """
    Complete model for a hospital: Adapter → Encoder → Head
    """
    def __init__(self, input_adapter, encoder, head):
        super(HospitalModel, self).__init__()
        
        self.input_adapter = input_adapter  # Private
        self.encoder = encoder  # Shared
        self.head = head  # Shared
    
    def forward(self, x):
        x = self.input_adapter(x)  # Hospital-specific transformation
        x = self.encoder(x)  # Shared encoding
        x = self.head(x)  # Shared prediction
        return x
    
    def get_shared_parameters(self):
        """
        Returns only the shared parameters (encoder + head).
        Excludes the private input adapter.
        """
        shared_params = {
            'encoder': self.encoder.state_dict(),
            'head': self.head.state_dict()
        }
        return shared_params
    
    def set_shared_parameters(self, shared_params):
        """
        Updates only the shared parameters (encoder + head).
        Keeps the private input adapter unchanged.
        """
        self.encoder.load_state_dict(shared_params['encoder'])
        self.head.load_state_dict(shared_params['head'])
