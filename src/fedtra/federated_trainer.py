"""
Main Federated Learning Training Loop
Orchestrates communication between global server and hospitals.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from server import GlobalServer
from hospital import Hospital
from csv_data_loader import CSVDataLoader
from config import Config
import matplotlib.pyplot as plt


class FederatedTrainer:
    """
    Orchestrates the federated learning process.
    """
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Load CSV data first
        self.data_loader = CSVDataLoader(config)
        self.hospital_data = self.data_loader.load_all_hospitals()
        
        # Initialize global server
        self.server = GlobalServer(config)
        
        # Initialize hospitals (now with auto-detected input_dim)
        self.hospitals = {}
        for hospital_id, hospital_config in config.HOSPITAL_CONFIGS.items():
            self.hospitals[hospital_id] = Hospital(
                hospital_id=hospital_id,
                config=config,
                hospital_config=hospital_config,
                device=device
            )
        
        # Metrics tracking
        self.history = {
            'global_rounds': [],
            'hospital_losses': {h_id: [] for h_id in self.hospitals.keys()},
            'hospital_accuracies': {h_id: [] for h_id in self.hospitals.keys()},
            'test_accuracies': {h_id: [] for h_id in self.hospitals.keys()}
        }
        
        # Create save directory
        os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    def setup_data(self):
        """
        Distributes loaded CSV data to all hospitals.
        """
        print("\n" + "="*60)
        print("FEDERATED LEARNING SETUP")
        print("="*60)
        
        # Print data summary
        self.data_loader.print_data_summary()
        
        # Distribute data to hospitals
        print("\n📤 Distributing data to hospitals...")
        for hospital_id, hospital in self.hospitals.items():
            data = self.hospital_data[hospital_id]
            hospital.set_data(data['X_train'], data['y_train'])
        
        # Store test data for evaluation
        self.test_data = {}
        for hospital_id, data in self.hospital_data.items():
            test_dataset = TensorDataset(data['X_test'], data['y_test'])
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False
            )
            self.test_data[hospital_id] = test_loader
        
        print("✓ Data setup complete\n")
    
    def train(self):
        """
        Executes the complete federated learning training process.
        """
        print("="*60)
        print("STARTING FEDERATED TRAINING")
        print("="*60)
        print(f"Global Rounds: {self.config.GLOBAL_ROUNDS}")
        print(f"Local Epochs per Round: {self.config.LOCAL_EPOCHS}")
        print(f"Hospitals: {list(self.hospitals.keys())}")
        print("="*60 + "\n")
        
        # Initial global weights
        global_weights = self.server.get_global_weights()
        
        # Broadcast initial weights to all hospitals
        print("📡 Broadcasting initial global weights to hospitals...")
        for hospital in self.hospitals.values():
            hospital.receive_global_weights(global_weights)
        print()
        
        # Federated training rounds
        for round_num in range(1, self.config.GLOBAL_ROUNDS + 1):
            print("\n" + "="*60)
            print(f"GLOBAL ROUND {round_num}/{self.config.GLOBAL_ROUNDS}")
            print("="*60)
            
            # Local training at each hospital
            print("\n🏥 Local Training Phase:")
            hospital_weights = []
            hospital_sample_counts = []
            
            for hospital_id, hospital in self.hospitals.items():
                print(f"\n  [{hospital_id}] Training locally...")
                loss, accuracy = hospital.train_local(self.config.LOCAL_EPOCHS)
                
                # Collect shared weights (excluding private adapter)
                weights = hospital.get_shared_weights()
                hospital_weights.append(weights)
                hospital_sample_counts.append(hospital.num_samples)
                
                # Track metrics
                self.history['hospital_losses'][hospital_id].append(loss)
                self.history['hospital_accuracies'][hospital_id].append(accuracy)
                
                print(f"  ✓ {hospital_id} - Avg Loss: {loss:.4f}, Avg Acc: {accuracy:.2f}%")
            
            # Global aggregation
            print("\n🌐 Global Aggregation Phase:")
            global_weights = self.server.federated_round(
                hospital_weights,
                hospital_sample_counts
            )
            
            # Broadcast updated weights
            print("\n📡 Broadcasting updated weights to hospitals...")
            for hospital in self.hospitals.values():
                hospital.receive_global_weights(global_weights)
            
            # Evaluation on test sets
            print("\n📊 Evaluation Phase:")
            for hospital_id, hospital in self.hospitals.items():
                test_loader = self.test_data[hospital_id]
                test_loss, test_accuracy = hospital.evaluate(test_loader)
                self.history['test_accuracies'][hospital_id].append(test_accuracy)
                print(f"  {hospital_id} Test - Loss: {test_loss:.4f}, Acc: {test_accuracy:.2f}%")
            
            self.history['global_rounds'].append(round_num)
            
            # Save checkpoint
            if round_num % 5 == 0 or round_num == self.config.GLOBAL_ROUNDS:
                checkpoint_path = os.path.join(
                    self.config.SAVE_DIR,
                    f'global_model_round_{round_num}.pt'
                )
                self.server.save_checkpoint(round_num, checkpoint_path)
        
        print("\n" + "="*60)
        print("FEDERATED TRAINING COMPLETE")
        print("="*60)
        
        # Final evaluation summary
        self.print_final_summary()
        
        # Save serving artifacts (global model + per-hospital adapters and preprocessors)
        self.save_serving_artifacts()
    
    def print_final_summary(self):
        """
        Prints final training summary.
        """
        print("\n📈 Final Results Summary:")
        print("-" * 60)
        
        for hospital_id in self.hospitals.keys():
            final_test_acc = self.history['test_accuracies'][hospital_id][-1]
            initial_test_acc = self.history['test_accuracies'][hospital_id][0]
            improvement = final_test_acc - initial_test_acc
            
            print(f"\n{hospital_id}:")
            print(f"  Initial Test Accuracy: {initial_test_acc:.2f}%")
            print(f"  Final Test Accuracy:   {final_test_acc:.2f}%")
            print(f"  Improvement:           {improvement:+.2f}%")
        
        print("\n" + "="*60)
    
    def plot_results(self, save_path='training_results.png'):
        """
        Plots training and test accuracy curves.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training accuracy
        ax1 = axes[0]
        for hospital_id, accuracies in self.history['hospital_accuracies'].items():
            ax1.plot(self.history['global_rounds'], accuracies, 
                    marker='o', label=hospital_id)
        ax1.set_xlabel('Global Round')
        ax1.set_ylabel('Training Accuracy (%)')
        ax1.set_title('Training Accuracy per Hospital')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test accuracy
        ax2 = axes[1]
        for hospital_id, accuracies in self.history['test_accuracies'].items():
            ax2.plot(self.history['global_rounds'], accuracies, 
                    marker='s', label=hospital_id)
        ax2.set_xlabel('Global Round')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Test Accuracy per Hospital')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Results plot saved to: {save_path}")
        plt.close()

    def save_serving_artifacts(self, model_dir='hospital_models'):
        """
        Saves global encoder/head, and each hospital's private adapter and preprocessing metadata
        for real-world inference.
        """
        import os
        import json
        import torch
        import pickle
        os.makedirs(model_dir, exist_ok=True)
        
        # Save final global encoder and head
        torch.save({
            'encoder_state_dict': self.server.global_encoder.state_dict(),
            'head_state_dict': self.server.global_head.state_dict(),
            'config': {
                'LATENT_DIM': self.config.LATENT_DIM,
                'ENCODER_HIDDEN_DIMS': self.config.ENCODER_HIDDEN_DIMS,
                'NUM_CLASSES': self.config.NUM_CLASSES,
            }
        }, os.path.join(model_dir, 'global_model_final.pt'))
        print(f"✓ Saved global model to {os.path.join(model_dir, 'global_model_final.pt')}")
        
        # Save per-hospital artifacts
        for hospital_id, hospital in self.hospitals.items():
            # Adapter weights
            adapter_path = os.path.join(model_dir, f"{hospital_id}_adapter.pt")
            torch.save(hospital.input_adapter.state_dict(), adapter_path)
            
            # Preprocessing (scaler) and metadata
            scaler = self.data_loader.scalers.get(hospital_id)
            scaler_path = os.path.join(model_dir, f"{hospital_id}_scaler.pkl")
            if scaler is not None:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            
            data = self.hospital_data[hospital_id]
            meta = {
                'hospital_id': hospital_id,
                'input_dim': data['input_dim'],
                'feature_names': data['feature_names'],
                'adapter_hidden_dim': hospital.hospital_config['adapter_hidden_dim'],
                'latent_dim': self.config.LATENT_DIM,
                'encoder_hidden_dims': self.config.ENCODER_HIDDEN_DIMS,
                'num_classes': data['num_classes'],
                'class_names': data['class_names']
            }
            meta_path = os.path.join(model_dir, f"{hospital_id}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            print(f"✓ Saved adapter ({adapter_path}), scaler ({scaler_path}), and metadata ({meta_path}) for {hospital_id}")


def main():
    """
    Main entry point for federated learning.
    """
    # Configuration
    config = Config()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}\n")
    
    # Create trainer
    trainer = FederatedTrainer(config, device=device)
    
    # Setup data
    trainer.setup_data()
    
    # Train
    trainer.train()
    
    # Plot results
    trainer.plot_results()


if __name__ == '__main__':
    main()
