# Federated Transfer Learning for Heterogeneous Hospitals: An Implementation and Empirical Study

## Abstract
We present a production-ready, open-source Federated Transfer Learning (FTL) system designed for multi-hospital collaboration under strict privacy constraints. Our system supports heterogeneous, non-overlapping feature spaces via private input adapters per site and a shared encoder–head model trained with Federated Averaging (FedAvg) and stabilized with FedProx and LayerNorm. We implement consistent global label mapping across hospitals and a reproducible preprocessing pipeline. We provide a file-based orchestration MVP using isolated containers, real inference artifacts for deployment, and extensive documentation. Empirical results on real medical datasets show that the federated model improves performance while preserving data privacy.

Keywords—Federated Learning, Transfer Learning, Privacy, Heterogeneous Features, Healthcare AI, FedAvg, FedProx, LayerNorm

## 1. Introduction
Federated Learning (FL) enables collaborative model training across data silos without centralizing raw data. In healthcare, data sharing is constrained by privacy regulations and practical concerns (e.g., data governance and residency). Traditional FL assumes consistent feature spaces across clients. In reality, hospitals often maintain distinct EHR schemas with partially overlapping or completely different features. We address this via Federated Transfer Learning (FTL): each site maps its local feature space into a shared latent space with a private adapter while collaboratively learning a shared encoder and head.

Contributions:
- A practical FTL design supporting heterogeneous features via private per-site adapters and shared encoder/head.
- Training stabilization with FedProx and LayerNorm for cross-site shifts.
- Consistent label mapping and validation across hospitals.
- Reproducible preprocessing pipeline and real inference artifacts for deployment.
- A free, demo-ready orchestration MVP with containers; extensible toward secure aggregation and DP.

## 2. Related Work
- Federated Averaging (FedAvg) [McMahan et al., 2017] provides the core aggregation scheme.
- FedProx stabilizes FL under heterogeneity by constraining local updates.
- Transfer Learning in FL leverages site-specific mappings for representation alignment.
- BatchNorm challenges under domain shift are known; alternatives like LayerNorm/GroupNorm mitigate cross-site running-stat issues.
- Privacy mechanisms include Secure Aggregation and Differential Privacy; we outline extensions to integrate them.

## 3. System Overview
The system decomposes into:
- Private Input Adapter (local per hospital): maps `input_dim → LATENT_DIM`.
- Shared Encoder: transforms latent representations; parameters are aggregated globally.
- Shared Head: task-specific classifier; parameters are aggregated globally.
- Global Orchestrator: coordinates rounds, aggregates updates (FedAvg), publishes global weights.
- Client Agent: executes local training with private adapter and returns only shared weights.

Core modules (under `src/fedtra/`):
- `models.py`: `InputAdapter`, `Encoder`, `GlobalHead`, `HospitalModel`.
- `hospital.py`: local training loop, optimizer, evaluation; returns only shared parameters.
- `server.py`: `GlobalServer` with FedAvg aggregation and global weight broadcast.
- `csv_data_loader.py`: reproducible preprocessing—missing values, categorical encoding, normalization; consistent global label mapping; stratified splits.
- `config.py`: hyperparameters, model architecture, preprocessing and training options.
- `federated_trainer.py`: single-process CLI training and artifact saving.

## 4. Federated Transfer Learning Design
### 4.1 Private Adapters for Heterogeneous Features
Each hospital maintains a private `InputAdapter(input_dim, latent_dim)`. Adapters are never shared, preserving schema privacy and enabling non-overlapping features.

### 4.2 Shared Components and Aggregation
The `Encoder(latent_dim → hidden_dims)` and `GlobalHead` are shared. Clients train locally and send only encoder/head weights. The server aggregates via FedAvg weighted by sample counts.

### 4.3 Stabilization: FedProx and LayerNorm
- FedProx: A proximal term (μ/2)||w − w_global||² minimizes client drift; configurable via `Config.FEDPROX_MU`.
- LayerNorm: Replaces BatchNorm to avoid running-stat mismatch across sites and small-batch instability.

### 4.4 Label Consistency and Validation
A global label vocabulary is constructed across hospitals to ensure consistent mapping. Early validation checks that derived label counts equal `Config.NUM_CLASSES`.

## 5. Privacy and Security Considerations
- Data never leaves the hospital; only model parameters of shared components are exchanged.
- Adapters remain private, protecting local schema details.
- Extensions (future work) include:
  - Secure Aggregation (mask-based) so the server cannot inspect individual updates.
  - Differential Privacy (clipping + noise) with privacy budget accounting.
  - mTLS for transport security and workload identity, audit logs, and model registry.

## 6. Implementation Details
### 6.1 Codebase and Packaging
- Python, PyTorch; packaged as `fedtra` under `src/` for clean imports.
- Services for orchestration and clients in `services/`.
- Scripts for inference in `scripts/`.
- Docker-based demo in `ops/` (compose) for a free local deployment.

### 6.2 Training Workflow (CLI)
- Preprocess CSVs per hospital via `CSVDataLoader`.
- Initialize `GlobalServer` and `Hospital` clients.
- For each round: local train, aggregate weights, update global model, evaluate.
- Save artifacts: global encoder/head and per-hospital adapters/scalers/metadata in `hospital_models/`.

### 6.3 Orchestration MVP
- `services/orchestrator.py`: publishes `global_round_k.pt`, aggregates client updates, writes next round weights.
- `services/client_agent.py`: loads CSV, receives global, trains locally (FedProx), submits updates.
- Communication via shared volume; suitable for a controlled/local demo; extensible to gRPC/mTLS.

### 6.4 Inference
- `scripts/inference.py` builds `InputAdapter → Encoder → Head` from saved artifacts.
- Reuses per-hospital scaler and metadata to ensure train/serve consistency.

## 7. Datasets and Preprocessing
- CSV inputs per hospital; heterogeneous feature spaces supported.
- Preprocessing: missing value handling, categorical encoding, normalization (StandardScaler), train/test split (stratified).
- Global label vocabulary ensures consistent class indexing across hospitals.

## 8. Experiments and Results
- Two medical datasets (e.g., `Medicaldataset.csv` with 8 features; `cardiac arrest dataset.csv` with 13 features).
- Metrics: accuracy per hospital across rounds, improvement over local baselines.
- Observations: FTL with private adapters + FedAvg + FedProx + LayerNorm yields stable federated improvements across heterogeneous sites.
- Visualizations: training curves (see `reports/training_results.png`).

## 9. Discussion
- Benefits: privacy-preserving collaboration; supports heterogeneous schemas; stable convergence with minimal assumptions.
- Limitations: no secure aggregation or DP in the MVP; latent space alignment may be further improved with CORAL/MMD; personalization layers can boost site-specific performance.

## 10. Future Work
- Secure Aggregation and Differential Privacy integration.
- Asynchronous rounds and partial aggregation thresholds.
- Latent alignment losses (e.g., CORAL/MMD) and personalization heads.
- mTLS/gRPC orchestration, model registry, audit logs, CI/CD for models.
- Robust aggregation (Byzantine-resilient) and drift detection.

## 11. Technical Architecture and Mathematical Formulation

### 11.1 Formal Problem Setup
- Let there be K hospitals (clients) indexed by k ∈ {1,…,K}. Hospital k holds local dataset D_k = {(x_i^k, y_i^k)} with feature dimension d_k (heterogeneous across k) and label space Y shared across hospitals.
- Each hospital learns a private adapter A_k: R^{d_k} → R^{L} mapping to a shared latent space of dimension L = `Config.LATENT_DIM`.
- A shared encoder E_θ: R^{L} → R^{H} and shared head H_φ: R^{H} → R^{C} are trained collaboratively, where C = `Config.NUM_CLASSES`.

The per-hospital model is: f_k(x) = H_φ(E_θ(A_k(x))). Training minimizes empirical risk per client with cross-entropy loss ℓ.

### 11.2 FedAvg Objective and Aggregation
- Local objective at client k:
  J_k(θ, φ, A_k) = E_{(x,y)∼D_k}[ ℓ(H_φ(E_θ(A_k(x))), y) ].

- Global objective (weighted by |D_k|):
  J(θ, φ, {A_k}) = ∑_{k=1}^{K} (|D_k|/N) · J_k(θ, φ, A_k), where N = ∑_k |D_k|.

- Federated Averaging (parameter-space average over shared components):
  w_{shared}^{t+1} = ∑_{k} (n_k/N) · w_{shared,k}^{t+}, where w_{shared} ∈ {θ, φ} and n_k = |D_k|.

Only (θ, φ) are aggregated; A_k remain private and are not shared.

### 11.3 FedProx Stabilization
To mitigate client drift under heterogeneity, we add a proximal term during local training (implemented in `fedtra.hospital.Hospital.train_local()`):

  L_k^{prox} = J_k(θ, φ, A_k) + (μ/2) · ||w_{shared} − w_{shared}^{(t)}||_2^2,

where μ = `Config.FEDPROX_MU`. The proximal penalty anchors local updates to the received global weights w_{shared}^{(t)}.

### 11.4 Layer Normalization vs Batch Normalization
- BatchNorm maintains batch-dependent running statistics that are inconsistent across clients; cross-site averaging of BN stats is unreliable.
- We replace BN with LayerNorm in `fedtra.models.Encoder` and `InputAdapter`:
  LN(x) = γ ⊙ (x − μ(x)) / σ(x) + β, computed per-sample across features, robust to site-specific batch distributions.

### 11.5 Optional Latent Alignment (future extension)
To further align embeddings across clients, a CORAL loss can be added on intermediate activations z_k = E_θ(A_k(x)):

  L_{CORAL}(S,T) = ||C_S − C_T||_F^2,

where C is covariance of features; extend to multi-client by matching to a running global covariance. This can be integrated into local loss with a small weight λ.

### 11.6 Label Mapping and Validation
- Global label vocabulary V is built by scanning hospitals’ target columns (`CSVDataLoader.load_all_hospitals()`). Labels are mapped via a deterministic index map to ensure consistent class indices across clients.
- Early validation enforces `num_classes == Config.NUM_CLASSES` and fails fast on mismatch.

### 11.7 Preprocessing Pipeline (per hospital)
- Missing values: drop/mean/median per `Config.HANDLE_MISSING`.
- Categorical encoding: LabelEncoder per categorical column.
- Normalization: StandardScaler fit on local features.
- Stratified train/test split with `Config.TRAIN_TEST_SPLIT`.
All steps are deterministic and scalers are serialized per hospital for train/serve parity.

### 11.8 Training Algorithm (Round t)
1) Server broadcasts w_{shared}^{(t)} = {θ^{(t)}, φ^{(t)}}.
2) Each client k:
   - Receives w_{shared}^{(t)}.
   - Trains locally for E epochs minimizing L_k^{prox} (adapters A_k are updated locally; shared params receive local gradients).
   - Produces updated shared weights w_{shared,k}^{(t+)} and sends them with n_k.
3) Server aggregates: w_{shared}^{(t+1)} = ∑_k (n_k/N)·w_{shared,k}^{(t+)}.
4) Optional: checkpoint and evaluate.

### 11.9 Complexity and Communication
- Local step cost per client: O(E · |D_k| · f), where f is forward/backward cost for f_k.
- Communication per round: shared params size |θ|+|φ| per client; adapters A_k are not transmitted.
- Convergence is improved with FedProx μ>0 when client distributions are non-IID/heterogeneous.

### 11.10 Threat Model and Privacy Posture
- Honest-but-curious server; clients do not share raw data, only model updates of shared layers.
- Risk: model inversion or membership inference from updates; mitigations (future work): Secure Aggregation, Differential Privacy, clipping.
- Adapters are private, reducing exposure of site-specific representation details.

### 11.11 Implementation Mapping
- `fedtra.models`: Adapter/Encoder/Head definitions and `HospitalModel.get_shared_parameters()`.
- `fedtra.hospital`: Local training loop with FedProx and evaluation.
- `fedtra.server`: Weighted FedAvg aggregation and global state updates.
- `fedtra.csv_data_loader`: Global label vocabulary, preprocessing, splits, tensorization.
- `services/orchestrator.py`: Round coordination and publishing of global weights.
- `services/client_agent.py`: Local training and update submission per hospital.
- `scripts/inference.py`: Real inference with saved artifacts.

## 11. Conclusion
We implemented a practical FTL system enabling hospitals with heterogeneous feature spaces to collaboratively learn without data sharing. Private adapters, shared encoder/head, FedProx, LayerNorm, and consistent label mapping deliver a robust, deployable solution. The system runs locally for free with Docker Compose and provides real inference artifacts for follow-on deployment. Future extensions will strengthen privacy guarantees and operational robustness.

## References
- H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS, 2017.
- T. Li, A. S. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated Optimization in Heterogeneous Networks,” MLSys, 2020 (FedProx).
- S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” ICML, 2015.
- J. Ba, J. Kiros, and G. Hinton, “Layer Normalization,” arXiv:1607.06450, 2016.
- K. Bonawitz et al., “Practical Secure Aggregation for Privacy-Preserving Machine Learning,” CCS, 2017.
