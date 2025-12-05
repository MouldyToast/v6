# TimeGAN V6: RTSGAN-Style Two-Stage Training for Mouse Trajectory Synthesis

A PyTorch implementation of conditional trajectory generation using a novel two-stage training approach inspired by RTSGAN. V6 achieves stable, high-quality trajectory synthesis by decoupling autoencoder and GAN training.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Usage Guide](#usage-guide)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Configuration](#configuration)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Development Status](#development-status)

---

## Overview

TimeGAN V6 generates realistic mouse movement trajectories conditioned on target distance. Unlike traditional TimeGAN approaches that train all components jointly, V6 uses a **strictly decoupled two-stage approach**:

1. **Stage 1 (Autoencoder)**: Train encoder-decoder to convergence, learning a stable latent representation
2. **Stage 2 (WGAN-GP)**: Freeze autoencoder, train generator-discriminator in the fixed latent space

This approach solves the fundamental problem of joint training: the encoder-generator semantic gap where both networks receive different learning signals and fail to align.

### Why Two-Stage Training?

| Problem in Joint Training | V6 Solution |
|--------------------------|-------------|
| Encoder learns from reconstruction (rich signal) | Autoencoder trains to convergence first |
| Generator learns from weak adversarial signal | Generator trains in fixed, meaningful latent space |
| Networks have no shared objective | Discriminator provides direct latent-space feedback |
| Training instability and mode collapse | WGAN-GP with gradient penalty ensures stable training |

---

## Key Features

- **Two-Stage Training**: Autoencoder → WGAN-GP in frozen latent space
- **Latent Space Pooling**: Compress variable-length sequences to fixed summary vectors
- **Conditional Generation**: Generate trajectories conditioned on target distance
- **Optimized Data Loading**:
  - Length-aware batching (30%+ efficiency improvement)
  - Condition-stratified sampling for uniform distribution
  - Quality filtering to remove outlier trajectories
- **Automatic Run Management**: Timestamped directories prevent overwrites
- **Comprehensive Evaluation**: Discriminative score, predictive score, MMD, visualization
- **Master Control Script**: Single entry point for all operations

---

## Architecture

```
STAGE 1: Autoencoder Training
════════════════════════════════════════════════════════════════════════════

Input                  Encoder              Pooler           Expander            Decoder              Output
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
x_real ──────────→ [LSTM Encoder] ──→ h_seq ──→ [Attention] ──→ z_summary ──→ [LSTM] ──→ h_recon ──→ [Linear] ──→ x_recon
(B, T, 8)             ↓              (B,T,48)      Pool        (B, 96)       Expand    (B,T,48)                  (B, T, 8)
                   3-layer                                                  2-layer
                   LSTM                                                     LSTM

Loss: L_recon + λ * L_latent (MSE reconstruction + latent consistency)


STAGE 2: WGAN-GP in Frozen Latent Space
════════════════════════════════════════════════════════════════════════════

Real Path:
x_real ──→ [Encoder*] ──→ [Pooler*] ──→ z_real ──────────────────────┐
           (frozen)       (frozen)      (B, 96)                       │
                                                                      ▼
Fake Path:                                                    ┌──────────────┐
noise ──┬──→ [Generator] ──→ z_fake ─────────────────────────→│ Discriminator │──→ WGAN Loss
        │       MLP         (B, 96)                           │   (Critic)    │    + GP
condition                                                      └──────────────┘
(B, 1)

Generation:
noise + condition ──→ [Generator] ──→ z_summary ──→ [Expander*] ──→ [Decoder*] ──→ x_fake
                                                     (frozen)       (frozen)
```

### Component Details

| Component | Type | Dimensions | Description |
|-----------|------|------------|-------------|
| **Encoder** | 3-layer LSTM | 8 → latent_dim (64) | Maps trajectories to latent sequences |
| **Decoder** | 3-layer LSTM | latent_dim → 8 | Reconstructs trajectories from latent |
| **Pooler** | Hybrid (attn+mean+max) | latent_dim → summary_dim (192) | Compresses sequence to summary vector |
| **Expander** | 2-layer LSTM | summary_dim → latent_dim | Reconstructs latent sequence from summary |
| **Generator** | 3-layer MLP | noise+cond → summary_dim | Generates fake latent summaries |
| **Discriminator** | 3-layer MLP | summary_dim → 1 | WGAN critic in latent space |

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- matplotlib (for visualization)

### Install Dependencies

```bash
# CPU only
pip install torch numpy matplotlib

# With CUDA (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib
```

### Clone Repository

```bash
git clone <repository-url>
cd v6
```

---

## Quick Start

### 1. Validate Your Data

```python
# In run_v6.py:
RUN_MODE = "validate_data"
CONFIG["data_dir"] = "processed_data_v4"
```

```bash
python run_v6.py
```

### 2. Run Quick Test (with synthetic data)

```python
RUN_MODE = "quick_test"
```

```bash
python run_v6.py
# Creates synthetic data and runs short training to verify setup
```

### 3. Train Full Model

```python
RUN_MODE = "train"
CONFIG["data_dir"] = "processed_data_v4"
CONFIG["device"] = "cuda"  # or "cpu"
```

```bash
python run_v6.py
# Outputs saved to: checkpoints/v6/run_YYYYMMDD_HHMMSS/
```

### 4. Generate Trajectories

```python
RUN_MODE = "generate"
CONFIG["eval_checkpoint"] = "checkpoints/v6/run_20241204_143022/final.pt"
CONFIG["n_samples"] = 100
```

```bash
python run_v6.py
```

### 5. Visualize Results

```python
RUN_MODE = "visualize"
CONFIG["eval_checkpoint"] = "checkpoints/v6/run_20241204_143022/final.pt"
CONFIG["viz_n_samples"] = 5
```

```bash
python run_v6.py
# Creates trajectory_comparison_YYYYMMDD_HHMMSS.png
```

---

## Data Format

### V4 Feature Format (8 dimensions)

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | `dx` | Relative X displacement from start | [-1, 1] |
| 1 | `dy` | Relative Y displacement from start | [-1, 1] |
| 2 | `speed` | Movement speed (normalized) | [-1, 1] |
| 3 | `accel` | Acceleration (normalized) | [-1, 1] |
| 4 | `sin_h` | Sine of heading angle | [-1, 1] |
| 5 | `cos_h` | Cosine of heading angle | [-1, 1] |
| 6 | `ang_vel` | Angular velocity (normalized) | [-1, 1] |
| 7 | `dt` | Time delta (normalized) | [-1, 1] |

### Required Files

Your `data_dir` must contain:

```
processed_data_v4/
├── X_train.npy      # (N_train, max_seq_len, 8) - Feature arrays
├── C_train.npy      # (N_train,) - Distance conditions [-1, 1]
├── L_train.npy      # (N_train,) - Original sequence lengths
├── X_val.npy
├── C_val.npy
├── L_val.npy
├── X_test.npy
├── C_test.npy
├── L_test.npy
└── normalization_params.npy  # (optional) For denormalization
```

### Preprocessing (if needed)

```bash
python preprocess_v4.py --input raw_data/ --output processed_data_v4/
```

---

## Usage Guide

### Master Control Script: `run_v6.py`

All operations are controlled by changing `RUN_MODE` at the top of the file:

```python
RUN_MODE = "train"  # <-- Change this
```

### Available Modes

| Mode | Description | Required Config |
|------|-------------|-----------------|
| `train` | Full training (Stage 1 + Stage 2) | `data_dir` |
| `train_stage1` | Train only autoencoder | `data_dir` |
| `train_stage2` | Train only WGAN-GP | `data_dir`, `stage1_checkpoint` |
| `resume` | Resume from checkpoint | `data_dir`, `resume_checkpoint` |
| `generate` | Generate synthetic trajectories | `eval_checkpoint` |
| `evaluate` | Run evaluation metrics | `data_dir`, `eval_checkpoint` |
| `visualize` | Plot real vs generated | `data_dir`, `eval_checkpoint` |
| `quick_test` | Smoke test with fake data | (none) |
| `validate_data` | Validate data directory | `data_dir` |
| `inspect_model` | Print architecture | (none) |
| `inspect_checkpoint` | Examine checkpoint | `eval_checkpoint` |
| `export_samples` | Generate and save to file | `eval_checkpoint` |

### Example Workflows

#### Full Training Pipeline

```python
# Step 1: Validate data
RUN_MODE = "validate_data"
# Run, verify output

# Step 2: Train
RUN_MODE = "train"
CONFIG["stage1_iterations"] = 15000
CONFIG["stage2_iterations"] = 30000
# Run, wait for completion

# Step 3: Evaluate
RUN_MODE = "evaluate"
CONFIG["eval_checkpoint"] = "checkpoints/v6/run_XXXXXX/final.pt"
# Run, check metrics

# Step 4: Visualize
RUN_MODE = "visualize"
CONFIG["viz_n_samples"] = 10
# Run, view plots
```

#### Resume Interrupted Training

```python
RUN_MODE = "resume"
CONFIG["resume_checkpoint"] = "checkpoints/v6/run_XXXXXX/stage2_iter5000.pt"
```

#### Generate with Specific Conditions

```python
# For programmatic use:
from timegan_v6 import TimeGANV6, TimeGANV6Config
import torch

config = TimeGANV6Config()
model = TimeGANV6(config)

checkpoint = torch.load("checkpoints/v6/run_XXXXXX/final.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate 100 trajectories with distance=0.5 (normalized)
conditions = torch.full((100, 1), 0.5)
trajectories = model.generate(100, conditions, seq_len=100)
# trajectories.shape = (100, 100, 8)
```

---

## Hyperparameter Optimization

TimeGAN V6 includes a full Optuna-based hyperparameter optimization pipeline (`optuna_optimize_v6.py`) for automated tuning of both stages.

### Quick Start

```bash
# Stage 1 optimization (autoencoder)
python optuna_optimize_v6.py --stage 1 --n-trials 50 --data-dir ./processed_data_v4

# Stage 2 optimization (requires Stage 1 checkpoint)
python optuna_optimize_v6.py --stage 2 --n-trials 100 \
    --stage1-checkpoint checkpoints/optuna/stage1_trial_0/best_model.pt

# Full pipeline (Stage 1 → Stage 2)
python optuna_optimize_v6.py --stage both --n-trials 50

# Analyze results
python optuna_optimize_v6.py --analyze --study-name timegan_v6_stage1
```

### Search Spaces

#### Stage 1 (Autoencoder)

| Parameter | Search Range | Type |
|-----------|--------------|------|
| `latent_dim` | [32, 48, 64, 96] | Categorical |
| `summary_dim` | [64, 96, 128, 192] | Categorical |
| `pool_type` | [attention, mean, last, hybrid] | Categorical |
| `expand_type` | [lstm, mlp, repeat] | Categorical |
| `lr` | [1e-5, 1e-2] | Log uniform |
| `lr_schedule` | [none, cosine, step, plateau] | Categorical |
| `lambda_recon` | [0.5, 2.0] | Uniform |
| `lambda_latent` | [0.1, 1.0] | Uniform |

#### Stage 2 (WGAN-GP)

| Parameter | Search Range | Type |
|-----------|--------------|------|
| `noise_dim` | [64, 128, 192, 256] | Categorical |
| `generator_hidden_dims` | Various MLP configs | Categorical |
| `discriminator_hidden_dims` | Various MLP configs | Categorical |
| `lr_G`, `lr_D` | [1e-5, 1e-3] | Log uniform |
| `n_critic` | [1, 10] | Integer |
| `lambda_gp` | [1.0, 50.0] | Log uniform |
| `use_feature_matching` | [True, False] | Categorical |

### Optimization Objectives

**Stage 1** (Multi-objective):
1. Reconstruction MSE (minimize)
2. Latent space quality (minimize)

**Stage 2** (Multi-objective):
1. MMD between real/fake latents (minimize)
2. Discriminative score (minimize → 0.5)

### Features

- **TPE Sampler**: Tree-structured Parzen Estimator for efficient search
- **Hyperband Pruner**: Early stopping of unpromising trials
- **Pareto Front**: Multi-objective optimization finds tradeoff solutions
- **Resume Support**: Studies are saved to SQLite database
- **Baseline Enqueuing**: Known good configs tried first
- **Visualization**: Exports plots (param importance, Pareto front, etc.)

### Output Structure

```
checkpoints/optuna/
├── stage1_trial_0/
│   └── best_model.pt
├── stage1_trial_1/
│   └── best_model.pt
├── ...
└── stage2_trial_0/
    └── best_model.pt

v6_optuna.db              # SQLite study database
runs/optuna/optuna.log    # Optimization log
plots/
├── param_importances.html
├── pareto_front.html
└── optimization_history.html
```

### Programmatic Usage

```python
from optuna_optimize_v6 import OptunaConfig, StudyManager, Stage1Objective
from data_loader_v6 import create_stage1_loader

config = OptunaConfig(data_dir="./processed_data_v4", device="cuda")
loader = create_stage1_loader(config.data_dir, batch_size=64)

objective = Stage1Objective(config, loader)
manager = StudyManager(config)
study = manager.run_stage1(objective, n_trials=50)

# Get best trial
print(study.best_trials[0].params)
```

---

## Configuration

### CONFIG Options

```python
CONFIG = {
    # ─── Data Paths ───────────────────────────────────────────────
    "data_dir": "processed_data_v4",      # Preprocessed data directory

    # ─── Checkpoint Paths ─────────────────────────────────────────
    # WINDOWS: Use forward slashes "D:/path" or raw strings r"D:\path"
    "checkpoint_dir": "./checkpoints/v6", # Base directory for outputs
    "stage1_checkpoint": None,            # .pt file for stage 2 training
    "resume_checkpoint": None,            # .pt file to resume from
    "eval_checkpoint": None,              # .pt file for eval/generate

    # ─── Device ───────────────────────────────────────────────────
    "device": "cuda",                     # "cuda" or "cpu"

    # ─── Training Parameters ──────────────────────────────────────
    "batch_size": 64,
    "num_workers": 0,                     # DataLoader workers (0 for Windows)

    # Stage 1 (Autoencoder)
    "stage1_iterations": 15000,
    "lr_autoencoder": 1e-3,
    "stage1_threshold": 0.05,             # Convergence threshold

    # Stage 2 (WGAN-GP)
    "stage2_iterations": 30000,
    "lr_generator": 1e-4,
    "lr_discriminator": 1e-4,
    "n_critic": 5,                        # D updates per G update

    # ─── Generation ───────────────────────────────────────────────
    "n_samples": 100,
    "seq_len": 100,

    # ─── Visualization ────────────────────────────────────────────
    "viz_n_samples": 5,
    "viz_show_plot": True,                # False for headless servers

    # ─── Run Naming ───────────────────────────────────────────────
    "auto_name_runs": True,               # Timestamp-based run directories
    "run_name": None,                     # Custom name (if auto=False)

    # ─── Logging ──────────────────────────────────────────────────
    "log_interval": 100,
    "eval_interval": 1000,
    "save_interval": 5000,

    # ─── Model Architecture ───────────────────────────────────────
    "config_preset": "default",           # "default", "fast", or "large"
}
```

### Model Configuration Presets

```python
from timegan_v6 import get_default_config, get_fast_config, get_large_config

# Default: Balanced quality/speed
config = get_default_config()
# latent_dim=48, summary_dim=96, ~955K params

# Fast: Quick experiments
config = get_fast_config()
# latent_dim=32, summary_dim=64, fewer iterations

# Large: Maximum quality
config = get_large_config()
# latent_dim=64, summary_dim=128, more iterations
```

### Detailed Model Config

```python
from timegan_v6 import TimeGANV6Config

config = TimeGANV6Config(
    # Architecture
    feature_dim=8,              # Input/output features
    latent_dim=48,              # Encoder output per timestep
    summary_dim=96,             # Pooled trajectory summary
    noise_dim=128,              # Generator noise input
    condition_dim=1,            # Condition dimensions

    # Encoder/Decoder
    encoder_hidden_dim=64,
    encoder_num_layers=3,
    decoder_hidden_dim=64,
    decoder_num_layers=3,

    # Pooler/Expander
    pool_type='attention',      # 'attention', 'mean', 'last', 'hybrid'
    expand_type='lstm',         # 'lstm', 'mlp', 'repeat'

    # Generator/Discriminator
    generator_hidden_dims=[256, 256, 256],
    discriminator_hidden_dims=[256, 256, 256],

    # Training
    lr_autoencoder=1e-3,
    lr_generator=1e-4,
    lr_discriminator=1e-4,
    n_critic=5,
    lambda_gp=10.0,             # Gradient penalty weight
    lambda_fm=1.0,              # Feature matching weight
)
```

---

## Training Pipeline

### Stage 1: Autoencoder Training

**Goal**: Learn a stable, meaningful latent representation of trajectories.

**Pipeline**:
```
Main path (bottleneck):
x_real → Encoder → h_seq → Pooler → z_summary → Expander → h_recon → Decoder → x_recon

Direct path (encoder training):
x_real → Encoder → h_seq → Decoder → x_direct
```

**Losses**:
- **Direct Loss**: `L_direct = MSE(x_real, x_direct)` - Trains encoder directly (bypasses bottleneck)
- **Reconstruction Loss**: `L_recon = MSE(x_real, x_recon)` - Trains bottleneck (pooler/expander)
- **Latent Consistency Loss**: `L_latent = MSE(h_seq, h_recon)` - Expander matches encoder output
- **Total**: `L_total = L_direct + L_recon + λ * L_latent`

**Why Direct Loss?** Without it, the encoder only gets gradients through the long path (encoder→pooler→expander→decoder), causing vanishing gradients. The direct path gives the encoder strong gradients like V4.

**Convergence**: Training continues until direct loss drops below threshold (default: 0.05).

### Stage 2: WGAN-GP Training

**Goal**: Train a generator to produce latent summaries that match the real data distribution.

**Pipeline**:
```
Real: x_real → Encoder* → Pooler* → z_real → Discriminator
Fake: noise + condition → Generator → z_fake → Discriminator
```
(* = frozen)

**Losses**:
- **Discriminator**: `L_D = E[D(z_fake)] - E[D(z_real)] + λ_gp * GP`
- **Generator**: `L_G = -E[D(z_fake)] + λ_fm * L_feature_matching`

**WGAN-GP Details**:
- Uses Wasserstein distance instead of binary classification
- Gradient penalty ensures Lipschitz constraint
- Critic trained `n_critic` times per generator update
- More stable than standard GAN training

### Output Directory Structure

```
checkpoints/v6/
└── run_20241204_143022/           # Timestamped run directory
    ├── stage1_final.pt            # After Stage 1 completion
    ├── stage2_iter5000.pt         # Periodic Stage 2 checkpoints
    ├── stage2_iter10000.pt
    ├── stage2_final.pt            # After Stage 2 completion
    ├── final.pt                   # Final model
    ├── logs/                      # TensorBoard logs
    │   └── events.out.tfevents.*
    ├── generated_samples_20241204_143022.npz
    └── trajectory_comparison_20241204_143022.png
```

---

## Evaluation Metrics

### Discriminative Score

Measures how distinguishable real vs fake trajectories are:
1. Train a classifier (2-layer LSTM) on real=1, fake=0
2. Test accuracy on held-out data
3. **Lower is better** (0.5 = indistinguishable)

### Predictive Score

Measures temporal coherence:
1. Train a predictor to forecast next timestep from sequence
2. Compare prediction error on real vs fake
3. **Lower is better**

### Maximum Mean Discrepancy (MMD)

Distribution distance in kernel space:
- Compares real and fake feature distributions
- Uses RBF kernel
- **Lower is better** (0 = identical distributions)

### Latent Space Coverage

Measures diversity of generated samples:
- Computes coverage of real latent space by generated samples
- **Higher is better** (1.0 = full coverage)

### Running Evaluation

```python
RUN_MODE = "evaluate"
CONFIG["eval_checkpoint"] = "path/to/final.pt"
```

Output:
```
Evaluation Report
─────────────────────────────────────────────
Discriminative Score: 0.52 (ideal: 0.5)
Predictive Score:     0.08 (lower is better)
MMD Score:            0.003 (lower is better)
Latent Coverage:      0.89 (higher is better)
─────────────────────────────────────────────
```

---

## Project Structure

```
v6/
├── run_v6.py                    # Master control script (START HERE)
├── optuna_optimize_v6.py        # Hyperparameter optimization with Optuna
├── README.md                    # This file
├── V6_RTSGAN_IMPLEMENTATION_PLAN.md  # Detailed architecture doc
│
├── timegan_v6/                  # V6 model package
│   ├── __init__.py              # Package exports
│   ├── config_model_v6.py       # Configuration dataclass
│   ├── model_v6.py              # Main TimeGANV6 model
│   ├── trainer_v6.py            # Training pipeline
│   ├── evaluation_v6.py         # Evaluation metrics
│   ├── losses_v6.py             # Loss functions
│   ├── pooler_v6.py             # Latent pooling (attention/mean/last)
│   ├── expander_v6.py           # Latent expansion (LSTM/MLP)
│   ├── latent_generator_v6.py   # Generator network
│   ├── latent_discriminator_v6.py  # Discriminator network
│   └── utils_v6.py              # Utilities (masking, gradient penalty)
│
├── data_loader_v6.py            # V6-optimized data loading
├── data_loader_v4.py            # Basic V4 data loading
├── preprocess_v4.py             # Raw data preprocessing
├── denormalize_utils_v4.py      # Convert back to original scale
│
├── embedder_v4.py               # LSTM encoder (shared with V4)
├── recovery_v4.py               # LSTM decoder (shared with V4)
├── config_model_v4.py           # V4 configuration
├── trainer_v4.py                # V4 trainer (legacy)
├── losses.py                    # Shared loss functions
│
├── tensorboard_logger.py        # TensorBoard utilities
└── view_tensorboard.py          # Launch TensorBoard viewer
```

---

## API Reference

### Main Classes

#### TimeGANV6

```python
from timegan_v6 import TimeGANV6, TimeGANV6Config

config = TimeGANV6Config()
model = TimeGANV6(config)

# Stage 1: Autoencoder
x_recon, losses = model.forward_autoencoder(x_real, lengths)

# Stage 2: GAN (after freezing)
model.freeze_autoencoder()
d_losses = model.forward_discriminator(x_real, condition, lengths)
g_losses = model.forward_generator(batch_size, condition)

# Generation
x_fake = model.generate(n_samples, conditions, seq_len=100)

# Encode/Decode
z = model.encode(x, lengths)
x = model.decode(z, seq_len)

# Interpolation
x_interp = model.interpolate(z1, z2, n_steps=10)
```

#### TimeGANV6Trainer

```python
from timegan_v6 import TimeGANV6Trainer

trainer = TimeGANV6Trainer(model, config)

# Full training
metrics = trainer.train(train_loader)

# Stage-by-stage
trainer.train_stage1(train_loader)
trainer.train_stage2(train_loader)

# Checkpointing
trainer.load_checkpoint("path/to/checkpoint.pt")
```

#### TimeGANV6Evaluator

```python
from timegan_v6 import TimeGANV6Evaluator

evaluator = TimeGANV6Evaluator(model)
metrics = evaluator.evaluate(X_test, C_test, L_test)
```

### Data Loading

```python
from data_loader_v6 import (
    create_dataloaders_v6,      # Standard loaders
    create_stage_loaders,       # Stage-optimized loaders
    create_stage1_loader,       # Length-aware batching
    create_stage2_loader,       # Condition-stratified sampling
    validate_v6_data,           # Validate data directory
    load_v6_data,               # Load all data as dict
)

# Recommended: Stage-specific loaders
stage1_loader, stage2_loader = create_stage_loaders(
    data_dir="processed_data_v4",
    batch_size=64,
    num_workers=0
)
```

### Convenience Functions

```python
from timegan_v6 import train_v6_optimized

# One-line training with optimized data loading
model, metrics = train_v6_optimized(
    data_dir="processed_data_v4",
    device="cuda"
)
```

---

## Troubleshooting

### Common Issues

#### Windows Path Errors

```
SyntaxWarning: invalid escape sequence '\V'
```

**Solution**: Use forward slashes or raw strings:
```python
# Good
"checkpoint_dir": "D:/V6/checkpoints"
"checkpoint_dir": r"D:\V6\checkpoints"

# Bad
"checkpoint_dir": "D:\V6\checkpoints"  # \V is escape sequence
```

#### PermissionError on Checkpoint

```
PermissionError: [Errno 13] Permission denied: 'checkpoints/v6/run_xxx'
```

**Solution**: You're pointing to a directory, not a file:
```python
# Bad
"eval_checkpoint": "checkpoints/v6/run_xxx"

# Good
"eval_checkpoint": "checkpoints/v6/run_xxx/final.pt"
```

#### CUDA Out of Memory

**Solutions**:
1. Reduce batch size: `CONFIG["batch_size"] = 32`
2. Use gradient accumulation
3. Use `config_preset = "fast"` for smaller model

#### Training Not Converging

**Stage 1 not converging**:
- Increase `stage1_iterations`
- Lower `stage1_threshold`
- Check data normalization (should be [-1, 1])

**Stage 2 unstable**:
- Increase `n_critic` (try 10)
- Lower learning rates
- Check for NaN in losses

#### matplotlib Not Showing Plots

```python
CONFIG["viz_show_plot"] = True  # Must be True for interactive
```

On headless servers:
```python
CONFIG["viz_show_plot"] = False  # Save to file only
```

### Debug Mode

```python
# Verbose output during training
CONFIG["log_interval"] = 10

# Short test run
CONFIG["stage1_iterations"] = 100
CONFIG["stage2_iterations"] = 100
```

### Expected Training Output

#### Stage 1 (Autoencoder) - Normal Output

```
[Stage 1] Iter    100/15000 | Direct: 0.3521 | Recon: 0.4521 | Latent: 0.2812 | 25.2 it/s
[Stage 1] Iter    500/15000 | Direct: 0.1423 | Recon: 0.1823 | Latent: 0.0956 | 24.8 it/s
[Stage 1] Iter   1000/15000 | Direct: 0.0792 | Recon: 0.0992 | Latent: 0.0534 | 25.1 it/s
[Stage 1] Iter   2000/15000 | Direct: 0.0523 | Recon: 0.0623 | Latent: 0.0312 | 24.9 it/s
[Stage 1] Iter   3000/15000 | Direct: 0.0398 | Recon: 0.0458 | Latent: 0.0221 | 25.0 it/s

Converged at iteration 3500!
Best direct reconstruction loss: 0.0385
```

**What to expect:**
- **Direct loss**: Encoder quality - should decrease steadily to <0.05 (main convergence target)
- **Recon loss**: Bottleneck quality - follows Direct, usually slightly higher
- **Latent loss**: Expander matching encoder - should stay non-zero, decreases slowly
- Speed: 20-40 it/s on GPU, 2-5 it/s on CPU
- Convergence: Usually 3000-8000 iterations

**Warning signs:**
- Direct stuck >0.1 after 5000 iterations → Learning rate too low or data issue
- Latent → 0 very fast while Direct/Recon stay high → Expander outpacing encoder (lower LR)
- NaN values → Exploding gradients, reduce lr or increase grad_clip

#### Stage 2 (WGAN-GP) - Normal Output

```
[Stage 2] Iter    100/30000 | D: -2.3451 | G: 1.8923 | W: 2.1234 | 38.5 it/s
[Stage 2] Iter   1000/30000 | D: -0.8234 | G: 0.7123 | W: 0.5821 | 38.2 it/s
[Stage 2] Iter   5000/30000 | D: -0.2341 | G: 0.1892 | W: 0.1523 | 38.0 it/s
[Stage 2] Iter  10000/30000 | D: -0.0823 | G: 0.0734 | W: 0.0456 | 37.8 it/s
[Stage 2] Iter  20000/30000 | D: -0.0412 | G: 0.0389 | W: 0.0234 | 37.5 it/s
```

**Metric interpretation:**
| Metric | Meaning | Good Range | Bad Signs |
|--------|---------|------------|-----------|
| D (d_loss) | Discriminator loss | -0.5 to 0.5 | Stuck at 0 or exploding |
| G (g_loss) | Generator loss | -0.5 to 0.5 | Constantly increasing |
| W (wasserstein) | Distribution distance | Decreasing → 0 | Stuck high or oscillating |

**Warning signs:**
- W not decreasing after 5000 iterations → Generator not learning
- D loss = 0 constantly → Mode collapse, discriminator "won"
- G loss constantly increasing → Generator diverging
- NaN in any loss → Training collapsed, need different hyperparameters

#### Evaluation Output - Interpreting Results

```
Evaluation Report
─────────────────────────────────────────────
Discriminative Score: 0.52 (ideal: 0.5)    ← EXCELLENT: Can't tell real from fake
Predictive Score:     0.08                  ← GOOD: Temporal patterns preserved
MMD Score:            0.003                 ← EXCELLENT: Distributions match
Latent Coverage:      0.89                  ← GOOD: Diverse outputs
─────────────────────────────────────────────
```

**Score interpretation:**
| Metric | Excellent | Good | Needs Work | Poor |
|--------|-----------|------|------------|------|
| Discriminative | 0.50-0.55 | 0.55-0.65 | 0.65-0.75 | >0.75 |
| Predictive | <0.1 | 0.1-0.5 | 0.5-1.0 | >1.0 |
| MMD | <0.01 | 0.01-0.05 | 0.05-0.1 | >0.1 |
| Latent Coverage | >0.85 | 0.7-0.85 | 0.5-0.7 | <0.5 |

### Diagnosing Common Issues from Logs

**Issue: Stage 1 loss not decreasing**
```
[Stage 1] Iter   1000/15000 | Recon: 0.4234 | Latent: 0.3891
[Stage 1] Iter   2000/15000 | Recon: 0.4198 | Latent: 0.3845
[Stage 1] Iter   3000/15000 | Recon: 0.4156 | Latent: 0.3802
```
**Diagnosis**: Learning rate too low
**Fix**: Increase `lr_autoencoder` to 1e-3 or 5e-3

**Issue: Stage 2 Wasserstein not decreasing**
```
[Stage 2] Iter   5000/30000 | D: -0.0012 | G: 0.0008 | W: 0.8923
[Stage 2] Iter  10000/30000 | D: -0.0008 | G: 0.0005 | W: 0.8845
```
**Diagnosis**: Discriminator too strong, generator can't learn
**Fix**: Decrease `n_critic` (try 3), or increase `lr_generator`

**Issue: NaN in losses**
```
[Stage 2] Iter    500/30000 | D: nan | G: nan | W: nan
```
**Diagnosis**: Training exploded
**Fix**: Lower learning rates, increase `grad_clip`, check data normalization

**Issue: Mode collapse (D wins completely)**
```
[Stage 2] Iter   5000/30000 | D: 0.0000 | G: 15.234 | W: 0.0000
```
**Diagnosis**: Generator collapsed, only produces one output
**Fix**: Increase `lambda_gp`, use `use_feature_matching=True`, reduce `n_critic`

**Issue: Stage 1 Recon plateaued but Latent → 0 (very fast)**
```
[Stage 1] Iter    100/15000 | Recon: 0.1586 | Latent: 0.0020
[Stage 1] Iter    200/15000 | Recon: 0.1422 | Latent: 0.0003
[Stage 1] Iter    400/15000 | Recon: 0.1427 | Latent: 0.0002
```
**Diagnosis**: The **expander is learning faster than the encoder**. Latent loss→0 means the expander matches h_seq perfectly, but h_seq itself is still random/low-quality because the encoder hasn't learned. The gradients weaken and training stagnates.
**Fix**: Lower the learning rate to slow down the expander, and increase encoder capacity:
```python
# In run_v6.py CONFIG:
"lr_autoencoder": 5e-4,      # Lowered from 1e-3
"latent_dim": 64,            # Increased from 48 for more encoder capacity
"summary_dim": 192,          # Larger bottleneck
"pool_type": "hybrid",       # Better information preservation
```
Or in code:
```python
from timegan_v6 import TimeGANV6Config
config = TimeGANV6Config(
    lr_autoencoder=5e-4,    # Slower learning to prevent expander dominance
    latent_dim=64,          # More encoder capacity
    summary_dim=192,        # Larger bottleneck
    pool_type='hybrid',     # Combines attention + mean + max pooling
)
```

**Issue: Very slow training (< 5 it/s on GPU)**
```
[Stage 1] Iter    100/15000 | Recon: 0.1593 | Latent: 0.0018 | 2.0 it/s
```
**Diagnosis**: CPU bottleneck or small batch not utilizing GPU
**Fix**: Increase `batch_size` (try 128 or 256), ensure `device='cuda'`, check `num_workers` (try 2-4 on Linux, 0 on Windows)

### TensorBoard Monitoring

**Launch TensorBoard:**
```bash
python view_tensorboard.py              # View all runs
python view_tensorboard.py --list       # List available runs
python view_tensorboard.py --latest 3   # View 3 most recent runs
python view_tensorboard.py --runs run_20251205  # Specific run (partial match)
```

**Key Metrics (Scalars tab):**
| Stage | Metric | Target | Description |
|-------|--------|--------|-------------|
| Stage 1 | `loss_direct` | < 0.05 | Encoder quality (main target) |
| Stage 1 | `loss_recon` | follows direct | Bottleneck quality |
| Stage 1 | `loss_latent` | stays non-zero | Expander tracking encoder |
| Stage 2 | `wasserstein` | increases | GAN training progress |
| Stage 2 | `d_loss` / `g_loss` | balanced | D and G competing |

**Trajectory Visualizations (Images tab):**
- `stage1/trajectories`: Real (blue) vs Reconstructed (red) - XY path, speed, error heatmap
- `stage2/generated_trajectories`: Real (blue) vs Generated (green) - XY path, speed, feature means

**Latent Space (Projector tab):**
- `latent_space`: z_summary embeddings visualized with PCA/t-SNE/UMAP
- Colored by condition value - similar conditions should cluster together

---

## Development Status

### Implementation Complete

The following components are fully implemented and tested:

| Component | File | Status |
|-----------|------|--------|
| **Core Model** | `timegan_v6/model_v6.py` | ✅ Complete |
| **Configuration** | `timegan_v6/config_model_v6.py` | ✅ Complete |
| **Trainer** | `timegan_v6/trainer_v6.py` | ✅ Complete |
| **Evaluation** | `timegan_v6/evaluation_v6.py` | ✅ Complete |
| **Pooler** (attention/mean/last/hybrid) | `timegan_v6/pooler_v6.py` | ✅ Complete |
| **Expander** (LSTM/MLP/repeat) | `timegan_v6/expander_v6.py` | ✅ Complete |
| **Generator** (standard/FiLM) | `timegan_v6/latent_generator_v6.py` | ✅ Complete |
| **Discriminator** (standard/projection/multiscale) | `timegan_v6/latent_discriminator_v6.py` | ✅ Complete |
| **Data Loader V6** | `data_loader_v6.py` | ✅ Complete |
| **Master Script** | `run_v6.py` | ✅ Complete |
| **Optuna Optimization** | `optuna_optimize_v6.py` | ✅ Complete |

### Key Implementation Details

**Two-Stage RTSGAN Architecture**:
- Stage 1: Autoencoder trained to convergence (encoder → pooler → expander → decoder)
- Stage 2: WGAN-GP in frozen latent space (generator + discriminator)
- Autoencoder is frozen during Stage 2 to ensure stable latent representations

**Data Loading Optimizations**:
- `LengthAwareSampler`: Groups similar-length sequences to reduce padding (30%+ efficiency)
- `ConditionStratifiedSampler`: Ensures uniform condition distribution for Stage 2
- Quality filtering removes outlier trajectories using Z-score threshold

**V4 Data Compatibility**:
- V6 uses the same preprocessed data format as V4 (8 features: dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
- No new preprocessing required - `processed_data_v4/` works directly

### File Quick Reference

| To Do This... | Use This File |
|---------------|---------------|
| Run training/evaluation/generation | `run_v6.py` |
| Optimize hyperparameters | `optuna_optimize_v6.py` |
| Understand model architecture | `timegan_v6/model_v6.py` |
| Modify training loop | `timegan_v6/trainer_v6.py` |
| Change model config | `timegan_v6/config_model_v6.py` |
| Add evaluation metrics | `timegan_v6/evaluation_v6.py` |
| Modify data loading | `data_loader_v6.py` |
| Preprocess raw data | `preprocess_v4.py` |

### Continuation Notes for Future Development

When resuming development in a new session:

1. **Start with this README** - Contains complete architecture and usage documentation
2. **Check `run_v6.py`** - Master control script with all run modes
3. **Review `timegan_v6/__init__.py`** - Shows all exported classes/functions
4. **Data uses V4 format** - 8 features, normalized to [-1, 1], uses `processed_data_v4/`

**Common next steps**:
- To train: Set `RUN_MODE = "train"` in `run_v6.py`
- To optimize hyperparameters: Run `python optuna_optimize_v6.py --stage 1`
- To evaluate: Set `RUN_MODE = "evaluate"` with checkpoint path
- To add new features: Check component files in `timegan_v6/`

**IMPORTANT - Keep Documentation Updated**:
When making code changes in a new session, **always update this README** to reflect:
- New features or configuration options added
- Changes to training behavior or expected output
- New troubleshooting entries for issues encountered
- Architecture changes (component dimensions, new modules, etc.)

This ensures the next session has accurate context without needing to re-discover implementation details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{timegan_v6,
  title={TimeGAN V6: RTSGAN-Style Two-Stage Training for Trajectory Synthesis},
  year={2024},
  url={https://github.com/your-repo/v6}
}
```

## License

[Add license information here]

## Acknowledgments

- Original TimeGAN: [Jinsung Yoon et al.](https://github.com/jsyoon0823/TimeGAN)
- RTSGAN inspiration: [Real-valued Time Series GAN](https://arxiv.org/abs/2006.16477)
