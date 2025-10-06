from flask import Flask, render_template, request, jsonify
import os
import json
import pickle
import numpy as np
import torch

# Import model components
from fedtra.models import InputAdapter, Encoder, GlobalHead, HospitalModel

app = Flask(__name__)

# Default model directory (shared by orchestrator/clients)
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(os.getcwd(), 'orchestrator_share', 'hospital_models'))


def load_artifacts(hospital_id: str, model_dir: str):
    global_path = os.path.join(model_dir, 'global_model_final.pt')
    adapter_path = os.path.join(model_dir, f'{hospital_id}_adapter.pt')
    scaler_path = os.path.join(model_dir, f'{hospital_id}_scaler.pkl')
    meta_path = os.path.join(model_dir, f'{hospital_id}_meta.json')
    if not all(os.path.exists(p) for p in [global_path, adapter_path, scaler_path, meta_path]):
        raise FileNotFoundError(f"Missing artifacts for {hospital_id} in {model_dir}. Expected: global_model_final.pt, {hospital_id}_adapter.pt, {hospital_id}_scaler.pkl, {hospital_id}_meta.json")

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


@app.route('/')
def index():
    # Discover available hospitals by scanning model_dir
    hospitals = []
    if os.path.isdir(MODEL_DIR):
        for name in os.listdir(MODEL_DIR):
            if name.endswith('_meta.json'):
                hospitals.append(name.replace('_meta.json', ''))
    hospitals.sort()
    return render_template('index.html', hospitals=hospitals, model_dir=MODEL_DIR)


@app.get('/meta/<hospital_id>')
def get_meta(hospital_id):
    meta_path = os.path.join(MODEL_DIR, f'{hospital_id}_meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'error': f'meta not found for {hospital_id}'}), 404
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return jsonify({'feature_names': meta['feature_names'], 'class_names': meta['class_names']})


@app.post('/predict')
def predict():
    hospital_id = request.form.get('hospital_id', '').strip()
    if not hospital_id:
        return jsonify({'error': 'hospital_id is required'}), 400
    try:
        model, scaler, meta = load_artifacts(hospital_id, MODEL_DIR)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Align features by name
    feat_names = meta['feature_names']
    try:
        row = []
        for fn in feat_names:
            v = request.form.get(fn, '')
            v = 0.0 if v == '' else float(v)
            row.append(v)
        X = np.array(row, dtype=np.float32).reshape(1, -1)
        Xs = scaler.transform(X)
        with torch.no_grad():
            x = torch.from_numpy(Xs).float()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = dict(zip(meta['class_names'], probs.tolist()))
        return jsonify({'hospital_id': hospital_id, 'prediction': pred})
    except Exception as e:
        return jsonify({'error': f'prediction failed: {e}'}), 500


if __name__ == '__main__':
    print(f"Starting web UI. MODEL_DIR={MODEL_DIR}")
    app.run(host='0.0.0.0', port=8000, debug=True)
