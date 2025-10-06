# Demo: Three-Machine Visualization (Orchestrator + Two Hospitals)

This folder helps you demo the system as if three different machines are running:
- `demo/orchestrator/` → global server/orchestrator
- `demo/hospital_1/` → first hospital client
- `demo/hospital_2/` → second hospital client

Open three VS Code windows in each subfolder and run the corresponding `run.sh` script. Only model files are exchanged via a shared directory; raw CSVs stay with the hospital clients.

## Prerequisites
- Python 3.11
- Pip install project deps:
  ```bash
  pip install -r requirements.txt
  ```

## Start order
1) Orchestrator:
   - Open VS Code at `demo/orchestrator/`
   - Terminal:
     ```bash
     bash run.sh
     ```
2) Hospital 1:
   - Open VS Code at `demo/hospital_1/`
   - Terminal:
     ```bash
     bash run.sh
     ```
3) Hospital 2:
   - Open VS Code at `demo/hospital_2/`
   - Terminal:
     ```bash
     bash run.sh
     ```

## Notes
- The scripts compute absolute paths, so you can run from anywhere.
- Shared folder: `<repo>/orchestrator_share/`
- Datasets used:
  - Hospital 1: `<repo>/datasets/Medicaldataset.csv` (target `Result`)
  - Hospital 2: `<repo>/datasets/cardiac arrest dataset.csv` (target `target`)
- If filenames/targets differ, edit the `run.sh` files accordingly.
- Ensure `src/` is importable via `PYTHONPATH` (set in scripts).

## Artifacts to observe
- Orchestrator publishes globals: `orchestrator_share/global/global_round_*.pt`
- Clients submit updates: `orchestrator_share/updates/client_*_round_*.pt`
- Final deployables: `hospital_models/` (global model, adapters, scalers, metadata)
