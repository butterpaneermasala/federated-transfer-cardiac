"""
Real inference using saved artifacts.

Usage:
  python scripts/inference.py --hospital_id hospital_1 --json '{"Age":64, "Gender":1, ...}'
  python scripts/inference.py --hospital_id hospital_1 --csv path/to/rows.csv

Requires artifacts in hospital_models/:
- global_model_final.pt
- <hospital_id>_adapter.pt
- <hospital_id>_scaler.pkl
- <hospital_id>_meta.json
"""
import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import pickle

from fedtra.models import InputAdapter, Encoder, GlobalHead, HospitalModel


def load_artifacts(model_dir, hospital_id):
    global_path = os.path.join(model_dir, 'global_model_final.pt')
    adapter_path = os.path.join(model_dir, f'{hospital_id}_adapter.pt')
    scaler_path = os.path.join(model_dir, f'{hospital_id}_scaler.pkl')
    meta_path = os.path.join(model_dir, f'{hospital_id}_meta.json')

    if not all(os.path.exists(p) for p in [global_path, adapter_path, scaler_path, meta_path]):
        raise FileNotFoundError('Missing required artifacts in hospital_models/')

    global_blob = torch.load(global_path, map_location='cpu')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Build model
    adapter = InputAdapter(input_dim=meta['input_dim'], latent_dim=meta['latent_dim'], hidden_dim=meta['adapter_hidden_dim'])
    encoder = Encoder(latent_dim=meta['latent_dim'], hidden_dims=meta['encoder_hidden_dims'])
    head = GlobalHead(input_dim=meta['encoder_hidden_dims'][-1], num_classes=meta['num_classes'])

    adapter.load_state_dict(torch.load(adapter_path, map_location='cpu'))
    encoder.load_state_dict(global_blob['encoder_state_dict'])
    head.load_state_dict(global_blob['head_state_dict'])

    model = HospitalModel(adapter, encoder, head)
    model.eval()
    return model, scaler, meta


def predict_json(model, scaler, meta, row_json):
    # Order features using meta['feature_names']
    feat_names = meta['feature_names']
    row_aligned = np.array([row_json.get(k, 0.0) for k in feat_names], dtype=np.float32).reshape(1, -1)
    row_scaled = scaler.transform(row_aligned)
    with torch.no_grad():
        x = torch.from_numpy(row_scaled).float()
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return dict(zip(meta['class_names'], probs.tolist()))


def predict_csv(model, scaler, meta, csv_path):
    df = pd.read_csv(csv_path)
    feat_names = meta['feature_names']
    X = df.reindex(columns=feat_names, fill_value=0.0).values.astype(np.float32)
    Xs = scaler.transform(X)
    with torch.no_grad():
        x = torch.from_numpy(Xs).float()
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()
    return [dict(zip(meta['class_names'], row.tolist())) for row in probs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hospital_id', required=True)
    ap.add_argument('--json', help='JSON string with feature:value pairs')
    ap.add_argument('--csv', help='CSV file with rows to predict')
    ap.add_argument('--model_dir', default='hospital_models')
    args = ap.parse_args()

    model, scaler, meta = load_artifacts(args.model_dir, args.hospital_id)

    if args.json:
        row = json.loads(args.json)
        probs = predict_json(model, scaler, meta, row)
        print(json.dumps({'prediction': probs}, indent=2))
    elif args.csv:
        preds = predict_csv(model, scaler, meta, args.csv)
        print(json.dumps({'predictions': preds}, indent=2))
    else:
        raise SystemExit('Provide --json or --csv')


if __name__ == '__main__':
    main()
