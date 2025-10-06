"""
Client Agent for Federated Learning (MVP)
- Runs at a single site (hospital)
- Loads local CSV, trains with private adapter, and uploads shared updates
- Coordinates via shared filesystem with orchestrator

Env vars:
- CLIENT_HOSPITAL_ID (e.g., hospital_1)
- CLIENT_CSV_PATH (e.g., datasets/Medicaldataset.csv)
- CLIENT_TARGET_COLUMN (e.g., Result)
- CLIENT_SHARE_DIR (default: ./orchestrator_share)
- CLIENT_LOCAL_EPOCHS (optional; defaults to Config.LOCAL_EPOCHS)
- CLIENT_DEVICE (cpu|cuda; default cpu)
"""
import os
import time
import torch
import csv
from datetime import datetime

from fedtra.config import Config
from fedtra.hospital import Hospital
from fedtra.csv_data_loader import CSVDataLoader


def load_global_weights(path: str):
    data = torch.load(path, map_location="cpu")
    assert 'encoder' in data and 'head' in data, f"Malformed global weights: {path}"
    return { 'encoder': data['encoder'], 'head': data['head'] }


def save_client_update(path: str, hospital_id: str, shared_weights, num_samples: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'hospital_id': hospital_id,
        'encoder': shared_weights['encoder'],
        'head': shared_weights['head'],
        'num_samples': int(num_samples)
    }, path)


def main():
    cfg = Config()
    hospital_id = os.getenv('CLIENT_HOSPITAL_ID', 'hospital_1')
    csv_path = os.getenv('CLIENT_CSV_PATH', 'datasets/Medicaldataset.csv')
    target_column = os.getenv('CLIENT_TARGET_COLUMN', 'Result')
    share_dir = os.getenv('CLIENT_SHARE_DIR', './orchestrator_share')
    local_epochs = int(os.getenv('CLIENT_LOCAL_EPOCHS', cfg.LOCAL_EPOCHS))
    device = os.getenv('CLIENT_DEVICE', 'cpu')

    global_dir = os.path.join(share_dir, 'global')
    updates_dir = os.path.join(share_dir, 'updates')
    os.makedirs(global_dir, exist_ok=True)
    os.makedirs(updates_dir, exist_ok=True)

    print("\n" + "="*60)
    print(f"CLIENT AGENT STARTED: {hospital_id}")
    print("="*60)
    print(f"CSV: {csv_path}")
    print(f"Target: {target_column}")
    print(f"Share dir: {share_dir}")

    # Prepare single-hospital loader
    loader = CSVDataLoader(cfg)
    # Use internal preprocess for a single file
    data = loader._load_and_preprocess(csv_path, target_column, hospital_id)

    # Hospital config (detected dims)
    hospital_config = {
        'input_dim': data['input_dim'],
        'adapter_hidden_dim': 64,
        'num_samples': len(data['X_train'])
    }
    hospital = Hospital(hospital_id, cfg, hospital_config, device=device)
    hospital.set_data(data['X_train'], data['y_train'])
    # Metrics log path per hospital
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = os.path.join(reports_dir, f'metrics_{hospital_id}.csv')
    # Write header if file does not exist
    if not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'hospital_id', 'round',
                'loss', 'accuracy',
                'num_samples', 'local_epochs', 'learning_rate', 'fedprox_mu',
                'device', 'input_dim', 'train_time_sec'
            ])

    # Rounds loop
    current_round = 0
    while True:
        gpath0 = os.path.join(global_dir, 'global_round_0.pt')
        if current_round == 0:
            # Wait for initial global weights
            if not os.path.exists(gpath0):
                time.sleep(1.0)
                continue
            gw = load_global_weights(gpath0)
            hospital.receive_global_weights(gw)
            # Train and submit update for round 1
            submit_round = 1
            print(f"Training locally for round {submit_round}...")
            t0 = time.time()
            loss, acc = hospital.train_local(local_epochs)
            train_time = time.time() - t0
            # Append metrics
            with open(metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(), hospital_id, submit_round,
                    f"{loss:.6f}", f"{acc:.2f}",
                    hospital.num_samples, local_epochs, cfg.LEARNING_RATE, getattr(cfg, 'FEDPROX_MU', 0.0),
                    device, hospital_config['input_dim'], f"{train_time:.3f}"
                ])
            shared = hospital.get_shared_weights()
            upd_path = os.path.join(updates_dir, f'client_{hospital_id}_round_{submit_round}.pt')
            save_client_update(upd_path, hospital_id, shared, hospital.num_samples)
            print(f"Submitted update for round {submit_round}: {upd_path}")
            current_round = submit_round
            continue
        else:
            # Server publishes global_round_{current_round}.pt after aggregating round {current_round}
            expected_gpath = os.path.join(global_dir, f'global_round_{current_round}.pt')
            if not os.path.exists(expected_gpath):
                time.sleep(1.0)
                continue
            gw = load_global_weights(expected_gpath)
            hospital.receive_global_weights(gw)
            # If we've reached the configured number of global rounds, save artifacts and stop gracefully
            max_rounds = int(os.getenv('ORCH_GLOBAL_ROUNDS', cfg.GLOBAL_ROUNDS))
            if current_round >= max_rounds:
                print(f"Reached final global round ({max_rounds}). Saving artifacts and exiting client {hospital_id}.")
                # Save per-hospital artifacts into the shared dir so inference can use --model_dir orchestrator_share/hospital_models
                hm_dir = os.path.join(share_dir, 'hospital_models')
                os.makedirs(hm_dir, exist_ok=True)
                # Save adapter
                adapter_path = os.path.join(hm_dir, f"{hospital_id}_adapter.pt")
                torch.save(hospital.input_adapter.state_dict(), adapter_path)
                # Save scaler if available
                scaler = loader.scalers.get(hospital_id)
                scaler_path = os.path.join(hm_dir, f"{hospital_id}_scaler.pkl")
                if scaler is not None:
                    import pickle
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler, f)
                # Save meta (cast numpy types to native JSON types)
                def _to_native(x):
                    try:
                        import numpy as np
                        if isinstance(x, (np.integer,)):
                            return int(x)
                        if isinstance(x, (np.floating,)):
                            return float(x)
                        if isinstance(x, (np.bool_,)):
                            return bool(x)
                    except Exception:
                        pass
                    return x
                meta = {
                    'hospital_id': hospital_id,
                    'input_dim': int(_to_native(data['input_dim'])),
                    'feature_names': [str(n) for n in data['feature_names']],
                    'adapter_hidden_dim': int(_to_native(hospital.hospital_config['adapter_hidden_dim'])),
                    'latent_dim': int(_to_native(cfg.LATENT_DIM)),
                    'encoder_hidden_dims': [int(_to_native(v)) for v in cfg.ENCODER_HIDDEN_DIMS],
                    'num_classes': int(_to_native(data['num_classes'])),
                    'class_names': [str(c) for c in data['class_names']]
                }
                import json, tempfile, shutil
                meta_path = os.path.join(hm_dir, f"{hospital_id}_meta.json")
                # atomic write to avoid partial files
                with tempfile.NamedTemporaryFile('w', delete=False) as tf:
                    json.dump(meta, tf, indent=2)
                    tmp_name = tf.name
                shutil.move(tmp_name, meta_path)
                # Also convert and save the latest global weights to global_model_final.pt
                # expected_gpath contains keys 'encoder' and 'head'; inference expects '*_state_dict'
                final_global_path = os.path.join(hm_dir, 'global_model_final.pt')
                torch.save({
                    'encoder_state_dict': gw['encoder'],
                    'head_state_dict': gw['head'],
                    'config': {
                        'LATENT_DIM': cfg.LATENT_DIM,
                        'ENCODER_HIDDEN_DIMS': cfg.ENCODER_HIDDEN_DIMS,
                        'NUM_CLASSES': cfg.NUM_CLASSES,
                    }
                }, final_global_path)
                print(f"Saved artifacts: {adapter_path}, {scaler_path if scaler is not None else '(no scaler)'}, {meta_path}, {final_global_path}")
                break
            # Train and submit update for the next round
            submit_round = current_round + 1
            print(f"Training locally for round {submit_round}...")
            loss, acc = hospital.train_local(local_epochs)
            shared = hospital.get_shared_weights()
            upd_path = os.path.join(updates_dir, f'client_{hospital_id}_round_{submit_round}.pt')
            save_client_update(upd_path, hospital_id, shared, hospital.num_samples)
            print(f"Submitted update for round {submit_round}: {upd_path}")
            current_round = submit_round


if __name__ == '__main__':
    main()
