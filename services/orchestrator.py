"""
Simple file-based Federated Learning orchestrator (MVP)
- Coordinates rounds across multiple client agents via a shared filesystem
- Aggregates encoder/head weights using FedAvg from client update files
- Publishes updated global weights each round

Usage (env/config):
- Configure participating clients via ORCH_CLIENT_IDS env (comma-separated)
- Rounds via ORCH_GLOBAL_ROUNDS (default: Config.GLOBAL_ROUNDS)
- Shared directory via ORCH_SHARE_DIR (default: ./orchestrator_share)

This MVP is intended for free local/demo deployment with docker-compose using a shared volume.
"""
import os
import time
import json
import glob
import torch
from typing import List

from fedtra.config import Config
from fedtra.server import GlobalServer


def load_client_update(path: str):
    """Load a client update file containing shared weights and num_samples."""
    data = torch.load(path, map_location="cpu")
    # Expected keys: 'encoder', 'head', 'num_samples', 'hospital_id'
    assert 'encoder' in data and 'head' in data and 'num_samples' in data, f"Malformed update: {path}"
    return data


def save_global_weights(server: GlobalServer, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'encoder': server.global_encoder.state_dict(),
        'head': server.global_head.state_dict(),
        'config': {
            'LATENT_DIM': server.config.LATENT_DIM,
            'ENCODER_HIDDEN_DIMS': server.config.ENCODER_HIDDEN_DIMS,
            'NUM_CLASSES': server.config.NUM_CLASSES,
        }
    }, path)


def main():
    cfg = Config()

    # Read orchestrator config from env
    client_ids = os.getenv('ORCH_CLIENT_IDS', 'hospital_1,hospital_2').split(',')
    client_ids = [c.strip() for c in client_ids if c.strip()]
    total_clients = len(client_ids)

    share_dir = os.getenv('ORCH_SHARE_DIR', './orchestrator_share')
    rounds = int(os.getenv('ORCH_GLOBAL_ROUNDS', cfg.GLOBAL_ROUNDS))
    poll_interval = float(os.getenv('ORCH_POLL_INTERVAL', '2.0'))

    print("\n" + "="*60)
    print("ORCHESTRATOR STARTED")
    print("="*60)
    print(f"Clients: {client_ids}")
    print(f"Rounds: {rounds}")
    print(f"Share dir: {share_dir}")

    os.makedirs(share_dir, exist_ok=True)
    global_dir = os.path.join(share_dir, 'global')
    updates_dir = os.path.join(share_dir, 'updates')
    os.makedirs(global_dir, exist_ok=True)
    os.makedirs(updates_dir, exist_ok=True)

    # Initialize global server and publish round_0 weights
    server = GlobalServer(cfg)
    round_idx = 0
    global_path = os.path.join(global_dir, f'global_round_{round_idx}.pt')
    save_global_weights(server, global_path)
    print(f"Published initial global weights: {global_path}")

    # Federated rounds
    for r in range(1, rounds + 1):
        print("\n" + "-"*60)
        print(f"Waiting for client updates for round {r}...")

        # Wait until we have all expected client updates
        expected = {cid: None for cid in client_ids}
        deadline = time.time() + 3600  # 1 hour safety
        while time.time() < deadline and any(v is None for v in expected.values()):
            # Look for files updates/client_<id>_round_<r>.pt
            pattern = os.path.join(updates_dir, f'client_*_round_{r}.pt')
            for upd_path in glob.glob(pattern):
                try:
                    upd = load_client_update(upd_path)
                    cid = upd.get('hospital_id')
                    if cid in expected and expected[cid] is None:
                        expected[cid] = upd
                        print(f"✔ Received update from {cid} for round {r}")
                except Exception as e:
                    print(f"Skipping malformed update {upd_path}: {e}")
            if any(v is None for v in expected.values()):
                time.sleep(poll_interval)
        missing = [cid for cid, v in expected.items() if v is None]
        if missing:
            print(f"⚠ Missing updates from: {missing}. Proceeding with available clients.")

        # Aggregate
        updates = [v for v in expected.values() if v is not None]
        if not updates:
            print("No updates received; stopping.")
            break
        hospital_weights = [{ 'encoder': u['encoder'], 'head': u['head'] } for u in updates]
        sample_counts = [int(u['num_samples']) for u in updates]
        aggregated = server.aggregate_weights(hospital_weights, sample_counts)
        server.update_global_model(aggregated)
        print(f"Aggregated {len(updates)} updates for round {r}")

        # Publish new global weights
        round_idx = r
        global_path = os.path.join(global_dir, f'global_round_{round_idx}.pt')
        save_global_weights(server, global_path)
        print(f"Published global weights: {global_path}")

    # Save final model artifact compatible with our trainer format
    final_path = os.path.join(share_dir, 'hospital_models', 'global_model_final.pt')
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({
        'encoder_state_dict': server.global_encoder.state_dict(),
        'head_state_dict': server.global_head.state_dict(),
        'config': {
            'LATENT_DIM': cfg.LATENT_DIM,
            'ENCODER_HIDDEN_DIMS': cfg.ENCODER_HIDDEN_DIMS,
            'NUM_CLASSES': cfg.NUM_CLASSES,
        }
    }, final_path)
    print(f"\nSaved final global model: {final_path}")
    print("Orchestration complete.")


if __name__ == '__main__':
    main()
