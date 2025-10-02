"""
Global Federated Learning Server
- Initializes global encoder and head
- Aggregates weights from hospitals
- Broadcasts updated weights
"""

import torch
import torch.nn as nn
import copy
from models import Encoder, GlobalHead
from config import Config


class GlobalServer:
    """
    Central server for federated learning.
    Manages global encoder and head, performs aggregation.
    """
    def __init__(self, config):
        self.config = config
        
        # Initialize global encoder
        self.global_encoder = Encoder(
            latent_dim=config.LATENT_DIM,
            hidden_dims=config.ENCODER_HIDDEN_DIMS
        )
        
        # Initialize global head
        self.global_head = GlobalHead(
            input_dim=config.ENCODER_HIDDEN_DIMS[-1],
            num_classes=config.NUM_CLASSES
        )
        
        print("✓ Global Server Initialized")
        print(f"  - Encoder: {config.LATENT_DIM} → {config.ENCODER_HIDDEN_DIMS}")
        print(f"  - Head: {config.ENCODER_HIDDEN_DIMS[-1]} → {config.NUM_CLASSES}")
    
    def get_global_weights(self):
        """
        Returns the current global encoder and head weights.
        These will be sent to hospitals.
        """
        global_weights = {
            'encoder': self.global_encoder.state_dict(),
            'head': self.global_head.state_dict()
        }
        return global_weights
    
    def aggregate_weights(self, hospital_weights, hospital_sample_counts):
        """
        Federated Averaging (FedAvg) aggregation.
        
        Args:
            hospital_weights: List of weight dictionaries from each hospital
            hospital_sample_counts: List of sample counts for weighted averaging
        
        Returns:
            Aggregated global weights
        """
        total_samples = sum(hospital_sample_counts)
        aggregated_weights = {'encoder': {}, 'head': {}}
        
        # Aggregate encoder weights
        for param_name in hospital_weights[0]['encoder'].keys():
            param_tensor = hospital_weights[0]['encoder'][param_name]
            # Handle both trainable parameters and buffers (like running_mean)
            if param_tensor.dtype in [torch.float32, torch.float64, torch.float16]:
                aggregated_weights['encoder'][param_name] = torch.zeros_like(param_tensor)
                for weights, num_samples in zip(hospital_weights, hospital_sample_counts):
                    weight_factor = num_samples / total_samples
                    aggregated_weights['encoder'][param_name] += \
                        weights['encoder'][param_name] * weight_factor
            else:
                # For integer types (like num_batches_tracked), just copy from first hospital
                aggregated_weights['encoder'][param_name] = param_tensor.clone()
        
        # Aggregate head weights
        for param_name in hospital_weights[0]['head'].keys():
            param_tensor = hospital_weights[0]['head'][param_name]
            if param_tensor.dtype in [torch.float32, torch.float64, torch.float16]:
                aggregated_weights['head'][param_name] = torch.zeros_like(param_tensor)
                for weights, num_samples in zip(hospital_weights, hospital_sample_counts):
                    weight_factor = num_samples / total_samples
                    aggregated_weights['head'][param_name] += \
                        weights['head'][param_name] * weight_factor
            else:
                # For integer types, just copy from first hospital
                aggregated_weights['head'][param_name] = param_tensor.clone()
        
        return aggregated_weights
    
    def update_global_model(self, aggregated_weights):
        """
        Updates the global encoder and head with aggregated weights.
        """
        self.global_encoder.load_state_dict(aggregated_weights['encoder'])
        self.global_head.load_state_dict(aggregated_weights['head'])
    
    def federated_round(self, hospital_weights, hospital_sample_counts):
        """
        Performs one round of federated aggregation.
        
        Args:
            hospital_weights: List of weight dictionaries from hospitals
            hospital_sample_counts: List of sample counts
        
        Returns:
            Updated global weights to broadcast
        """
        print(f"\n🔄 Aggregating weights from {len(hospital_weights)} hospitals...")
        
        # Aggregate weights using FedAvg
        aggregated_weights = self.aggregate_weights(
            hospital_weights, 
            hospital_sample_counts
        )
        
        # Update global model
        self.update_global_model(aggregated_weights)
        
        print("✓ Global model updated")
        
        # Return updated weights for broadcasting
        return self.get_global_weights()
    
    def save_checkpoint(self, round_num, save_path):
        """
        Saves the global model checkpoint.
        """
        checkpoint = {
            'round': round_num,
            'encoder_state_dict': self.global_encoder.state_dict(),
            'head_state_dict': self.global_head.state_dict()
        }
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Loads a saved checkpoint.
        """
        checkpoint = torch.load(checkpoint_path)
        self.global_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.global_head.load_state_dict(checkpoint['head_state_dict'])
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        return checkpoint['round']
