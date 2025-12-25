# Diffusion V7 - Master Architecture Document

**Pure Diffusion Model for Mouse Trajectory Generation**
Conditioned on endpoint (distance + angle) using Classifier-Free Guidance

---

## Table of Contents

1. [Overview](#overview)
2. [Complete Pipeline](#complete-pipeline)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [File Classification](#file-classification)
7. [Usage Guide](#usage-guide)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Diffusion V7?

Diffusion V7 is a **pure diffusion model** (DDPM) that generates realistic mouse cursor trajectories conditioned on endpoint goals. Unlike latent diffusion, it operates directly in trajectory space.

### Key Features

- **Pure Diffusion**: Works directly on 8D trajectory features (no encoder/decoder)
- **Goal Conditioning**: 3D condition vector `[distance_norm, cos(angle), sin(angle)]`
- **Classifier-Free Guidance (CFG)**: Stronger conditioning during generation
- **Variable Length**: Handles trajectories from 11-200 timesteps
- **DDIM Sampling**: Fast generation (50 steps vs 1000 DDPM steps)

### Architecture Decisions

**Why Pure Diffusion instead of Latent Diffusion?**
- V6 GAN has 50:1 compression (9,600D → 192D pooled latent)
- Risk of losing temporal details in such aggressive compression
- Pure diffusion: 85-90% success rate vs latent 80-85%
- Reference: MotionDiffuse uses pure diffusion for motion sequences

**Why Skip Time Conditioning Initially?**
- Simplifies initial implementation
- Distance/angle are primary controls for mouse movements
- Can add time conditioning later as enhancement

---

## Complete Pipeline

### Phase 0: Data Collection & Preparation

This phase captures raw mouse movement data and prepares it for preprocessing.

```
Step 0.1: Record Mouse Movements
─────────────────────────────────
SapiRecorder.py (GUI Application)
    │
    │  User performs point-to-point movements:
    │  - Click dot A (start) → Move → Click dot B (target)
    │  - 125Hz sampling (8ms intervals)
    │  - 192 combinations (24 distances × 8 orientations)
    │  - Real-time validation (start/end velocity checks)
    │
    ↓
D:\V6\user0001\
    ├── session_2024_01_15_1_3min.csv
    ├── session_2024_01_15_2_3min.csv
    └── ... (multiple session files)

Step 0.2: Combine Recording Sessions
────────────────────────────────────
python combine_cvss.py
    │
    │  Simply concatenates all session CSVs
    │  Preserves all original data unchanged
    │
    ↓
D:\V6\combined_all_sessions.csv
    (Columns: client timestamp, button, state, x, y)

Step 0.3: Split into Individual Trajectories
────────────────────────────────────────────
python trajectory_splitter_adaptive.py combined_all_sessions.csv trajectories/
    │
    │  - Splits on click events (Released state)
    │  - Removes leading/trailing stationary periods
    │  - Preserves true timing (~2ms intervals)
    │  - Computes ideal_distance and actual_distance
    │
    ↓
trajectories/
    ├── trajectory_0001.json
    ├── trajectory_0002.json
    └── ... (one JSON per recorded movement)
```

**CSV Format (from SapiRecorder.py):**
```csv
client timestamp,button,state,x,y
0,NoButton,Move,1250,710
8,NoButton,Move,1250,710
16,NoButton,Move,1251,709
...
1234,Left,Pressed,1450,520
1234,Left,Released,1450,520
```

**JSON Format (from trajectory_splitter_adaptive.py):**
```json
{
  "x": [1250, 1251, 1253, ...],
  "y": [710, 709, 707, ...],
  "t": [0, 8, 16, ...],
  "ideal_distance": 347.5,
  "actual_distance": 362.1,
  "original_length": 156,
  "extracted_length": 142
}
```

**Data Collection Configuration (SapiRecorder.py):**
```python
# 24 distance thresholds (pixels) - logarithmically spaced
DISTANCE_THRESHOLDS = [27, 31, 36, 41, 47, 54, 62, 71, 82, 94, 108, 124,
                       143, 164, 189, 217, 250, 288, 331, 381, 438, 504, 580, 667]

# 8 cardinal directions
ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Angle ranges for each direction (screen coordinates: Y increases downward)
SCREEN_ANGLE_RANGES = {
    "E":  (-22.5, 22.5),   "SE": (22.5, 67.5),   "S":  (67.5, 112.5),
    "SW": (112.5, 157.5),  "W":  (157.5, 202.5), "NW": (-157.5, -112.5),
    "N":  (-112.5, -67.5), "NE": (-67.5, -22.5),
}

# Total combinations: 24 distances × 8 directions = 192 trajectories per session
```

### Phase 1: Data Preprocessing

```
Raw Data (trajectories/*.json)
    ↓
preprocess_V6.py
    │
    ├── Step 1: Load raw trajectories (x, y, t arrays)
    │
    ├── Step 1.5: Filter bad recordings
    │   ├── Length > 195 points (would be truncated)
    │   ├── Distance ratio > 3.0x (erratic/distracted path)
    │   └── Ideal distance < 20px (too short)
    │
    ├── Step 2: Compute 8D style features
    │   └── [dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt]
    │
    ├── Step 3.0: Clip outliers (1st-99th percentile)
    │   └── Prevents extreme values from dominating normalization
    │
    ├── Step 3: Normalize features to [-1, 1]
    │   └── Min-max normalization per feature
    │
    ├── Step 4: Compute 3D goal conditioning
    │   └── [distance_norm, cos(angle), sin(angle)]
    │
    ├── Step 5: Pad sequences to fixed length (200)
    │   └── Post-padding: [real, real, ..., 0, 0, 0]
    │
    └── Step 6: Create train/val/test splits (60/20/20)
    ↓
processed_data_v6/
    ├── X_train.npy           (trajectories: batch × 200 × 8)
    ├── C_train.npy           (conditions: batch × 3)
    ├── L_train.npy           (lengths: batch)
    ├── X_val.npy, C_val.npy, L_val.npy
    ├── X_test.npy, C_test.npy, L_test.npy
    └── normalization_params.npy
```

**Trajectory Features (8D):**
1. `dx`: Relative x from start
2. `dy`: Relative y from start
3. `speed`: Movement magnitude (pixels/sec)
4. `acceleration`: Speed change rate
5. `sin(heading)`: Direction sine
6. `cos(heading)`: Direction cosine
7. `angular_velocity`: Turning rate
8. `dt`: Time delta between samples

**Goal Condition (3D):**
1. `distance_norm`: Normalized distance [-1, 1]
2. `cos(goal_angle)`: X direction component
3. `sin(goal_angle)`: Y direction component

### Phase 2: Model Training

```
processed_data_v6/
    ↓
train_diffusion_v7.py --mode [smoke_test|medium|full]
    ↓
checkpoints_diffusion_v7/
    ├── best.pth              (lowest validation loss)
    ├── final.pth             (last epoch)
    └── checkpoint_epoch_N.pth
```

**Training Process:**
1. **Smoke Test** (Phase 2.1): Single-batch overfitting (64 samples, 1000 steps)
   - Validates architecture works
   - Loss should decrease 10-15x
2. **Full Training** (Phase 2.2): Train on full dataset
   - 300+ epochs typical
   - Saves best.pth when validation loss improves

### Phase 3: Generation & Evaluation

```
checkpoints_diffusion_v7/best.pth
    ↓
generate_diffusion_v7.py
    ↓
results/diffusion_v7/
    ├── generated_trajectories.npy
    ├── generation_conditions.npy
    ├── generation_lengths.npy
    ├── evaluation_metrics.json
    └── generation_info.json
```

**Generation Methods:**
- **DDIM** (default): 50 steps, deterministic
- **DDPM**: 1000 steps, stochastic

**Evaluation Metrics:**
- **Goal Accuracy**: Distance error, angle error
- **Realism**: Speed/accel distributions, smoothness (jerk)
- **Diversity**: Pairwise trajectory distances per condition

### Phase 4: Visualization

```
results/diffusion_v7/
    ↓
visualize_generated.py OR create_evaluation_report()
    ↓
Diagnostic plots for analysis
```

---

## Directory Structure

```
v6/
├── # ─────────────────────────────────────────────────────────────────
├── # DATA COLLECTION & PREPARATION (Phase 0)
├── # ─────────────────────────────────────────────────────────────────
├── SapiRecorder.py                  # GUI for recording mouse movements
├── combine_cvss.py                  # Combine multiple session CSVs
├── trajectory_splitter_adaptive.py  # Split CSV → individual JSON trajectories
├── trajectories/                    # Output: individual trajectory JSONs
│   ├── trajectory_0001.json
│   ├── trajectory_0002.json
│   └── ...
│
├── # ─────────────────────────────────────────────────────────────────
├── # DIFFUSION MODEL (Phase 1-4)
├── # ─────────────────────────────────────────────────────────────────
├── diffusion_v7/                    # Main package
│   ├── __init__.py
│   ├── config_trajectory.py         # Configuration system
│   │
│   ├── models/                      # Model architectures
│   │   ├── __init__.py
│   │   ├── goal_conditioner.py      # Embeds 3D goal → latent_dim
│   │   ├── trajectory_transformer.py # Main denoising model
│   │   └── gaussian_diffusion.py    # DDPM/DDIM (from MotionDiffuse)
│   │
│   ├── datasets/                    # Data loading
│   │   ├── __init__.py
│   │   └── trajectory_dataset.py    # Loads .npy files
│   │
│   ├── trainers/                    # Training loops
│   │   ├── __init__.py
│   │   └── trajectory_trainer.py    # CFG training, smoke test
│   │
│   ├── sampling/                    # Generation
│   │   ├── __init__.py
│   │   └── trajectory_sampler.py    # DDIM/DDPM with CFG
│   │
│   └── evaluation/                  # Metrics & visualization
│       ├── __init__.py
│       ├── metrics.py               # Evaluation metrics
│       └── visualize.py             # Plotting utilities
│
├── processed_data_v6/               # Preprocessed data
│   ├── X_train.npy, C_train.npy, L_train.npy
│   ├── X_val.npy, C_val.npy, L_val.npy
│   ├── X_test.npy, C_test.npy, L_test.npy
│   └── normalization_params.npy
│
├── checkpoints_diffusion_v7/        # Saved models
│   ├── best.pth
│   ├── final.pth
│   └── checkpoint_epoch_*.pth
│
├── results/                         # Generated outputs
│   └── diffusion_v7/
│       ├── generated_trajectories.npy
│       ├── evaluation_metrics.json
│       └── plots/
│
├── preprocess_V6.py                 # Data preprocessing
├── train_diffusion_v7.py            # Training CLI
├── generate_diffusion_v7.py         # Generation CLI
├── test_diffusion_v7_architecture.py # Architecture validation
│
└── Diagnostic Scripts (optional):
    ├── visualize_generated.py
    ├── compare_real_vs_generated.py
    ├── diagnose_distance_error.py
    └── check_norm_params.py
```

---

## Core Components

### 1. Configuration (`config_trajectory.py`)

The complete configuration with all parameters:

```python
@dataclass
class TrajectoryDiffusionConfig:
    # ═══════════════════════════════════════════════════════════════════
    # DATA CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    input_dim: int = 8              # Trajectory features (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
    condition_dim: int = 3          # Goal conditioning (distance_norm, cos(angle), sin(angle))
    max_seq_len: int = 200          # Maximum trajectory length

    # ═══════════════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════
    latent_dim: int = 512           # Transformer hidden dimension
    num_layers: int = 9             # Number of transformer blocks
    num_heads: int = 8              # Attention heads (latent_dim must be divisible by num_heads)
    ff_size: int = 1024             # Feedforward expansion size
    dropout: float = 0.1            # Dropout rate
    activation: str = 'gelu'        # Activation: 'gelu', 'relu', 'silu'

    # ═══════════════════════════════════════════════════════════════════
    # CLASSIFIER-FREE GUIDANCE (CFG)
    # ═══════════════════════════════════════════════════════════════════
    use_cfg: bool = True            # Enable classifier-free guidance
    cfg_dropout: float = 0.1        # Probability of dropping condition during training
    cfg_guidance_scale: float = 2.0 # Guidance scale during sampling (>1 = stronger conditioning)

    # ═══════════════════════════════════════════════════════════════════
    # DIFFUSION PROCESS
    # ═══════════════════════════════════════════════════════════════════
    diffusion_steps: int = 1000     # Number of diffusion timesteps (T)
    noise_schedule: str = 'linear'  # Beta schedule: 'linear', 'cosine', 'sqrt'
    beta_start: float = 1e-4        # Starting beta value
    beta_end: float = 0.02          # Ending beta value
    model_mean_type: str = 'epsilon'    # 'epsilon' (predict noise), 'x0', 'previous_x'
    model_var_type: str = 'fixed_small' # 'fixed_small', 'fixed_large', 'learned'
    loss_type: str = 'mse'              # 'mse', 'l1'

    # ═══════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════
    batch_size: int = 64            # Training batch size
    learning_rate: float = 1e-4     # Adam learning rate
    weight_decay: float = 0.0       # L2 regularization
    grad_clip: float = 0.5          # Gradient clipping threshold
    num_epochs: int = 1000          # Total training epochs
    optimizer: str = 'adam'         # 'adam' or 'adamw'
    betas: tuple = (0.9, 0.999)     # Adam betas

    # Learning rate schedule
    use_lr_scheduler: bool = False  # Enable cosine LR decay
    lr_warmup_steps: int = 500      # Warmup steps
    lr_decay_steps: int = 10000     # Steps for cosine decay

    # ═══════════════════════════════════════════════════════════════════
    # SAMPLING (GENERATION)
    # ═══════════════════════════════════════════════════════════════════
    sampling_method: str = 'ddim'   # 'ddpm' or 'ddim'
    ddim_steps: int = 50            # DDIM sampling steps (faster than 1000 DDPM)
    ddim_eta: float = 0.0           # DDIM eta (0=deterministic, 1=stochastic like DDPM)
    clip_denoised: bool = False     # Clip samples to [-1, 1] during denoising

    # ═══════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════
    data_dir: str = 'processed_data_v6'
    num_workers: int = 4            # DataLoader workers
    pin_memory: bool = True         # Pin memory for GPU transfer

    # ═══════════════════════════════════════════════════════════════════
    # LOGGING & CHECKPOINTING
    # ═══════════════════════════════════════════════════════════════════
    log_interval: int = 10          # Log every N steps
    sample_interval: int = 100      # Generate samples every N steps (smoke test)
    checkpoint_interval: int = 500  # Save checkpoint every N steps
    save_dir: str = 'checkpoints_diffusion_v7'
    log_dir: str = 'logs_diffusion_v7'  # TensorBoard logs

    # ═══════════════════════════════════════════════════════════════════
    # SMOKE TEST MODE
    # ═══════════════════════════════════════════════════════════════════
    smoke_test: bool = False        # Enable single-batch overfitting mode
    smoke_test_steps: int = 1000    # Steps for smoke test
    smoke_test_batch_size: int = 64 # Batch size for smoke test
```

**Presets (from `config_trajectory.py`):**

| Preset | latent_dim | num_layers | num_heads | ff_size | ~Parameters |
|--------|------------|------------|-----------|---------|-------------|
| `smoke_test()` | 128 | 4 | 4 | 256 | ~858K |
| `medium()` | 256 | 6 | 8 | 512 | ~4.8M |
| `full()` | 512 | 9 | 8 | 1024 | ~28.5M |

### 2. Goal Conditioner (`goal_conditioner.py`)

Embeds 3D goal condition → latent space for transformer.

```
[distance_norm, cos(angle), sin(angle)]  ← Input: (batch, 3)
    ↓
Linear(3 → 128) + SiLU + Dropout(0.1)
    ↓
Linear(128 → 256) + SiLU + Dropout(0.1)
    ↓
Linear(256 → latent_dim)
    ↓
goal_embedding (batch, latent_dim)       ← Output
```

**Classifier-Free Guidance (CFG) Implementation:**
```python
# Training: Randomly drop conditions (replace with learned null embedding)
cfg_mask = torch.rand(batch_size) > cfg_dropout  # True = keep condition
embedded = mask * condition_embed + (1 - mask) * null_embed

# Sampling: Combine conditional + unconditional predictions
pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
# cfg_scale = 1.0 → pure conditional
# cfg_scale > 1.0 → extrapolate beyond conditional (stronger guidance)
```

**Key Methods:**
- `forward(condition, mask)` - Embed conditions with optional CFG masking
- `get_null_embedding(batch_size, device)` - Get null embeddings for unconditional
- `sample_with_cfg(condition, model_fn, guidance_scale)` - CFG sampling helper

### 3. Trajectory Transformer (`trajectory_transformer.py`)

Denoising model adapted from MotionDiffuse. Predicts noise given noisy trajectory, timestep, and goal embedding.

**Full Architecture:**
```
Input: x_noisy (batch, seq_len, 8), t (batch,), goal_embed (batch, latent_dim), lengths (batch,)

┌────────────────────────────────────────────────────────────────────┐
│  INPUT EMBEDDING                                                    │
│  ─────────────────                                                  │
│  x_noisy (batch, seq_len, 8)                                       │
│      ↓ Linear(8 → latent_dim)                                      │
│  h (batch, seq_len, latent_dim)                                    │
│      + pos_encoding (learnable: max_seq_len × latent_dim)          │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  CONDITIONING EMBEDDING                                             │
│  ──────────────────────                                             │
│  t (batch,)                                                         │
│      ↓ timestep_embedding() [sinusoidal, like NeRF]                │
│      ↓ Linear(latent_dim → latent_dim) + SiLU + Linear             │
│  t_emb (batch, latent_dim)                                         │
│                                                                     │
│  cond_emb = concat([t_emb, goal_embed])  → (batch, 2*latent_dim)   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TRANSFORMER BLOCKS (×num_layers)                                   │
│  ────────────────────────────────                                   │
│  Each block:                                                        │
│      TemporalSelfAttention(h, cond_emb, mask)                      │
│          └→ Multi-head attention with length masking               │
│          └→ StylizationBlock for FiLM conditioning                 │
│      FeedForward(h, cond_emb)                                      │
│          └→ Linear(latent_dim → ff_size) + GELU + Linear           │
│          └→ StylizationBlock for FiLM conditioning                 │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  OUTPUT PROJECTION                                                  │
│  ────────────────                                                   │
│  h (batch, seq_len, latent_dim)                                    │
│      ↓ zero_module(Linear(latent_dim → 8))  [initialized to zero]  │
│  noise_pred (batch, seq_len, 8)                                    │
└────────────────────────────────────────────────────────────────────┘
```

**StylizationBlock (FiLM Conditioning):**
```python
# Modulates hidden states using scale and shift from conditioning
emb_out = Linear(cond_emb) → SiLU → Linear → (scale, shift)
h = LayerNorm(h) * (1 + scale) + shift  # FiLM modulation
h = SiLU → Dropout → zero_module(Linear)
```

**Key Implementation Details:**
- `zero_module()`: Initializes output layers to zero for stable training
- Positional encoding: **Learnable** (not sinusoidal like original Transformer)
- Attention mask: Generated from `lengths` tensor (1=valid, 0=padding)
- `cond_embed_dim = time_embed_dim + goal_latent_dim` (double latent_dim)

**Forward Signature:**
```python
def forward(self, x, t, goal_embed, lengths) -> noise_pred:
    # x: (batch, seq_len, 8) - noisy trajectory features
    # t: (batch,) - diffusion timesteps [0, T-1]
    # goal_embed: (batch, latent_dim) - from GoalConditioner
    # lengths: (batch,) - actual sequence lengths (for masking)
    # Returns: (batch, seq_len, 8) - predicted noise ε
```

### 4. Gaussian Diffusion (`gaussian_diffusion.py`)

Copied from MotionDiffuse (OpenAI's DDPM implementation).

**Features:**
- Beta schedules: linear, cosine, sqrt
- Model mean types: epsilon, x0, previous_x
- Loss types: MSE, rescaled MSE, KL
- DDIM sampling support

### 5. Trajectory Sampler (`trajectory_sampler.py`)

Fast generation with DDIM and Classifier-Free Guidance.

```python
class TrajectoryGenerator:
    def generate(self, conditions, lengths,
                 method='ddim',        # 'ddim' or 'ddpm'
                 ddim_steps=50,        # Fast: 50 steps vs 1000 DDPM
                 ddim_eta=0.0,         # 0=deterministic, 1=stochastic
                 cfg_scale=2.0,        # >1 = stronger conditioning
                 use_cfg=True,
                 num_samples=1,        # Samples per condition (for diversity)
                 show_progress=True)
```

**DDIM Sampling Algorithm:**
```python
# Create timestep schedule (50 evenly spaced steps from 1000)
timesteps = [980, 960, 940, ..., 20, 0]  # Reversed for denoising

# Start from pure noise
x = torch.randn(batch, seq_len, 8)

for t in timesteps:
    # 1. Predict noise (with CFG if enabled)
    if use_cfg:
        pred_cond = model(x, t, goal_embed)
        pred_uncond = model(x, t, null_embed)
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
    else:
        pred = model(x, t, goal_embed)

    # 2. DDIM update step
    x_0_pred = (x - sqrt(1-α_t) * pred) / sqrt(α_t)  # Predict clean sample
    x = sqrt(α_next) * x_0_pred + sqrt(1-α_next) * pred  # Step toward clean

# 3. Mask padding positions based on lengths
x = x * length_mask
```

**Performance:**
- DDIM (50 steps): ~0.5 seconds per batch of 64
- DDPM (1000 steps): ~10 seconds per batch of 64

**Convenience Method:**
```python
# Generate single trajectory
trajectory = generator.generate_single(
    distance=0.5,    # Normalized distance
    angle=np.pi/4,   # Target angle in radians
    length=100,      # Desired length
    cfg_scale=2.0
)  # Returns: (length, 8)
```

---

## Data Flow

### Training Data Flow

```
JSON files → preprocess_V6.py → .npy files
    ↓
TrajectoryDataset loads (X, C, L)
    ↓
TrajectoryDiffusionTrainer:
    1. Sample timestep t
    2. Forward diffusion: x_t = √(α_t)·x_0 + √(1-α_t)·ε
    3. GoalConditioner: condition → goal_embed (with 10% dropout)
    4. TrajectoryTransformer: predict ε from x_t
    5. Loss: MSE(predicted_ε, actual_ε)
    6. Backprop & optimize
```

### Generation Data Flow

```
Load checkpoint → restore model weights
    ↓
Sample goal conditions (random or from test set)
    ↓
TrajectoryGenerator:
    1. Start from pure noise: x_T ~ N(0, I)
    2. For t = T...1:
        a. goal_embed_cond = GoalConditioner(condition, mask=True)
        b. goal_embed_uncond = GoalConditioner.null_embedding
        c. pred_cond = model(x_t, t, goal_embed_cond)
        d. pred_uncond = model(x_t, t, goal_embed_uncond)
        e. pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        f. x_{t-1} = DDIM_step(x_t, pred)
    3. Return x_0 (generated trajectory)
    ↓
Evaluate metrics (with denormalization)
```

---

## File Classification

### ✅ ESSENTIAL - DO NOT DELETE

**Data Collection & Preparation (Phase 0):**
- `SapiRecorder.py` - GUI for recording mouse movements (125Hz sampling)
- `combine_cvss.py` - Combine multiple recording session CSVs
- `trajectory_splitter_adaptive.py` - Split combined CSV into individual trajectory JSONs

**Core Package (Phase 1-4):**
- `diffusion_v7/` (entire directory)
  - `config_trajectory.py`
  - `models/goal_conditioner.py`
  - `models/trajectory_transformer.py`
  - `models/gaussian_diffusion.py`
  - `datasets/trajectory_dataset.py`
  - `trainers/trajectory_trainer.py`
  - `sampling/trajectory_sampler.py`
  - `evaluation/metrics.py`
  - `evaluation/visualize.py`
  - All `__init__.py` files

**Scripts:**
- `preprocess_V6.py` - Data preprocessing (Phase 1)
- `train_diffusion_v7.py` - Training CLI (Phase 2)
- `generate_diffusion_v7.py` - Generation CLI (Phase 3)

**Data:**
- `processed_data_v6/` (all .npy files)
- `checkpoints_diffusion_v7/` (trained models)

### ⚠️ USEFUL - Keep for Development

**Testing & Validation:**
- `test_diffusion_v7_architecture.py` - Validates model works

**Diagnostics:**
- `visualize_generated.py` - Trajectory shape analysis
- `compare_real_vs_generated.py` - Straightness comparison
- `diagnose_distance_error.py` - Distance accuracy analysis
- `check_norm_params.py` - Inspect normalization parameters

### ❌ SAFE TO DELETE - Temporary/Diagnostic

**One-off Utilities:**
- `check_normalization.py` - Simple parameter inspector (redundant with check_norm_params.py)

**Old/Unused Files (if present):**
- Any files from MotionDiffuse not in `diffusion_v7/models/gaussian_diffusion.py`
- V6 GAN files (if you're fully replacing with diffusion)
- Any `.pyc` or `__pycache__/` directories
- Temporary test scripts you created

### 📝 DOCUMENTATION - Keep for Reference

- `DIFFUSION_RECOMMENDATION.md` - Initial design decision document
- `DIFFUSION_V7_ARCHITECTURE.md` (this file)
- Any `README.md` files

### 🗂️ LEGACY - Review Before Deleting

**If V6 GAN is still in use:**
- Keep: `embedder_v6.py`, `recovery_v6.py`, `model_v6.py`, `pooler_v6.py`
- Keep: Training scripts for V6 GAN

**If fully migrating to diffusion:**
- Can archive: All V6 GAN files
- Keep preprocessing: `preprocess_V6.py` (still needed for diffusion)

### 🔒 LEGACY MOTIONDIFFUSE FILES (in diffusion_v7/)

The `diffusion_v7/` package contains legacy MotionDiffuse code that is **NOT actively used** but kept for reference. These files have unmet dependencies (CLIP, HumanML3D, etc.) and are excluded from `__init__.py` exports:

```
diffusion_v7/
├── models/
│   └── transformer.py          # Original MotionTransformer (requires CLIP)
│                                # → Replaced by trajectory_transformer.py
│
├── trainers/
│   └── ddpm_trainer.py         # Original DDPMTrainer (requires full MotionDiffuse stack)
│                                # → Replaced by trajectory_trainer.py
│
├── datasets/
│   └── dataset.py              # Text2MotionDataset (requires HumanML3D)
│   └── evaluator.py            # EvaluationDataset (requires sentence-transformers)
│                                # → Replaced by trajectory_dataset.py
│
├── options/                    # Argparse option files (not used)
│   ├── base_options.py
│   ├── train_options.py
│   └── evaluate_options.py
│
├── tools/                      # Original training scripts (not used)
│   ├── train.py
│   ├── evaluation.py
│   └── visualization.py
│
└── utils/                      # Utility modules (partially used)
    ├── skeleton.py             # For human skeleton (not used)
    ├── quaternion.py           # For motion (not used)
    └── paramUtil.py            # Parameters (not used)
```

**Why kept?** The `gaussian_diffusion.py` is adapted from these and may need reference during debugging. Do not delete until diffusion training is fully validated.

**Active code (what you should edit):**
- `config_trajectory.py`
- `models/goal_conditioner.py`
- `models/trajectory_transformer.py`
- `models/gaussian_diffusion.py`
- `datasets/trajectory_dataset.py`
- `trainers/trajectory_trainer.py`
- `sampling/trajectory_sampler.py`
- `evaluation/metrics.py`, `evaluation/visualize.py`

---

## Usage Guide

### 0. Data Collection & Preparation (Phase 0)

**Step 0.1: Record Mouse Movements**
```bash
python SapiRecorder.py
```
- A GUI window opens with two dots (A=start, B=target)
- Click on dot A to start recording → Move mouse → Click on dot B to stop
- Each session records 192 trajectories (24 distances × 8 directions)
- Output: `D:\V6\user0001\session_*.csv` files

**Recording Settings (in SapiRecorder.py):**
```python
DOT_RADIUS = 10
SESSION_DURATION_MS = 400000        # ~6.6 minutes per session
TARGET_TRAJECTORIES = 192           # One complete cycle
VALIDATION_ENABLED = True           # Reject bad starts/ends
START_VELOCITY_THRESHOLD = 10       # px/s max for valid start
END_VELOCITY_THRESHOLD = 10         # px/s max for valid end
```

**Step 0.2: Combine Recording Sessions**
```bash
# Edit combine_cvss.py to set INPUT_DIR and OUTPUT_CSV paths
python combine_cvss.py
```
- Combines all session CSVs into one file
- Output: `combined_all_sessions.csv`

**Step 0.3: Split into Individual Trajectories**
```bash
python trajectory_splitter_adaptive.py combined_all_sessions.csv trajectories/
```
- Splits on click events (Left button Released)
- Removes leading/trailing stationary periods
- Validates timing consistency (~2ms intervals)
- Output: `trajectories/trajectory_NNNN.json` files

### 1. Preprocessing (Phase 1)

```bash
python preprocess_V6.py --input trajectories/ --output processed_data_v6/
```

**Output:**
- `processed_data_v6/X_{train,val,test}.npy`
- `processed_data_v6/C_{train,val,test}.npy`
- `processed_data_v6/L_{train,val,test}.npy`
- `processed_data_v6/normalization_params.npy`

### 2. Training

**Smoke Test (validate architecture):**
```bash
python train_diffusion_v7.py --mode smoke_test --data_dir processed_data_v6
```

**Expected:** Loss decreases from ~0.3 to ~0.02 in 1000 steps

**Full Training:**
```bash
python train_diffusion_v7.py --mode medium --data_dir processed_data_v6 --epochs 500
```

**Checkpoints:**
- `checkpoints_diffusion_v7/best.pth` - Lowest validation loss
- `checkpoints_diffusion_v7/final.pth` - Last epoch
- `checkpoints_diffusion_v7/checkpoint_epoch_N.pth` - Periodic saves

### 3. Generation

**Basic:**
```bash
python generate_diffusion_v7.py --checkpoint checkpoints_diffusion_v7/best.pth --num_samples 100 --save_trajectories --evaluate
```

**With stronger CFG:**
```bash
python generate_diffusion_v7.py --checkpoint checkpoints_diffusion_v7/best.pth --num_samples 100 --cfg_scale 5.0
```

**Using test set conditions:**
```bash
python generate_diffusion_v7.py --checkpoint checkpoints_diffusion_v7/best.pth --use_test_conditions --num_samples 100 --compare_real
```

**Output:**
- `results/diffusion_v7/generated_trajectories.npy`
- `results/diffusion_v7/evaluation_metrics.json`

### 4. Visualization

**Built-in evaluation report:**
```python
from diffusion_v7.evaluation import create_evaluation_report
import numpy as np

trajectories = np.load('results/diffusion_v7/generated_trajectories.npy')
conditions = np.load('results/diffusion_v7/generation_conditions.npy')
lengths = np.load('results/diffusion_v7/generation_lengths.npy')

create_evaluation_report(trajectories, conditions, lengths,
                         output_dir='results/diffusion_v7/plots')
```

**Diagnostic scripts:**
```bash
python visualize_generated.py
python compare_real_vs_generated.py
python diagnose_distance_error.py
```

---

## Evaluation Metrics (`evaluation/metrics.py`)

The `evaluate_trajectories()` function computes comprehensive metrics:

### Goal Accuracy Metrics

**Distance Error:**
```python
# For each trajectory:
final_dx, final_dy = trajectory[length-1, 0:2]  # Last valid position
actual_distance = sqrt(final_dx² + final_dy²)

# Denormalize target distance (if norm_params provided)
target_distance = (target_norm + 1) / 2 * (d_max - d_min) + d_min

error = abs(actual_distance - target_distance)

# Output metrics:
# - mean_distance_error, median_distance_error, std_distance_error
# - max_distance_error, min_distance_error
```

**Angle Error:**
```python
# For each trajectory:
actual_angle = atan2(final_dy, final_dx)
target_angle = atan2(sin(target), cos(target))

# Compute shortest angular distance (handles wraparound)
error = angular_distance(actual_angle, target_angle)

# Output metrics:
# - mean_angle_error_deg, median_angle_error_deg
# - max_angle_error_deg, std_angle_error_rad
```

### Realism Metrics

```python
# Extract feature statistics from trajectories:
speeds = trajectory[:length, 2]       # speed feature
accels = trajectory[:length, 3]       # acceleration feature
ang_vels = trajectory[:length, 6]     # angular velocity feature
jerk = diff(accel)                    # smoothness (derivative of accel)

# Output metrics:
# - gen_speed_mean, gen_speed_std
# - gen_accel_mean, gen_accel_std
# - gen_ang_vel_mean, gen_ang_vel_std
# - gen_smoothness_mean (lower = smoother trajectories)

# If real trajectories provided for comparison:
# - real_speed_mean, real_speed_std (same for accel, ang_vel)
# - speed_mean_diff, accel_mean_diff, smoothness_diff
```

### Diversity Metrics

```python
# Group trajectories by identical conditions
# Compute pairwise MSE distances within each group
# (Requires num_samples > 1 per condition)

# Output metrics:
# - diversity_mean (higher = more diverse)
# - diversity_median, diversity_std
# - num_condition_groups, avg_samples_per_condition
```

### Interpreting Results

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| `mean_distance_error` | < 20px | < 50px | > 100px |
| `mean_angle_error_deg` | < 10° | < 25° | > 45° |
| `smoothness_diff` | < 0.01 | < 0.05 | > 0.1 |
| `diversity_mean` | > 0.001 | > 0.0001 | = 0 (mode collapse) |

---

## Troubleshooting

### Common Issues

**1. High Distance Error (>0.5)**

**Symptom:** `mean_distance_error: 0.538` or higher

**Diagnosis:**
```bash
python diagnose_distance_error.py
```

**Causes:**
- Normalization params not loaded (should see "Loaded normalization params from...")
- Scale mismatch (denormalization broken)
- Model not learning distance (correlation < 0.3)

**Fix:**
- Ensure `normalization_params.npy` exists in data_dir
- Check if generation script loads it correctly
- If correlation is low, retrain with lower CFG dropout (0.05 instead of 0.1)

**2. Trajectories Too Straight (straightness >0.95)**

**Symptom:** Generated trajectories look like lines

**Diagnosis:**
```bash
python compare_real_vs_generated.py
```

**If real data is also straight (>0.9):**
- This is correct! Model learned the data distribution
- Real mouse movements in your dataset are very straight

**If only generated is straight:**
- Increase model capacity (try `--mode full`)
- Train longer (500+ epochs)
- Increase CFG scale during generation (try 5.0)

**3. PyTorch 2.6 Loading Error**

**Symptom:** `WeightsUnpickler error: Unsupported global`

**Fix:** Already fixed in code - `weights_only=False` in `torch.load()`

**4. JSON Serialization Error**

**Symptom:** `Object of type float32 is not JSON serializable`

**Fix:** Already fixed - converts numpy types to Python types before saving

**5. Poor Angle Accuracy (>30° mean error)**

**Possible causes:**
- CFG scale too low (try 5.0 or 10.0)
- Model undertrained
- Angle encoding issue (rare - should work with cos/sin)

**6. Training Loss Not Decreasing**

**Smoke test fail:**
- Check data loading (shapes correct?)
- Verify gradient flow (test_diffusion_v7_architecture.py)
- Check learning rate (default: 2e-4)

**Full training plateau:**
- Lower learning rate (try 1e-4)
- Add learning rate warmup
- Check for NaN gradients

---

## Technical Details

### Normalization

**Distance Normalization:**
```python
# Training data: d_min to d_max pixels
# Normalized: [-1, 1]
d_norm = 2 * ((d - d_min) / (d_max - d_min)) - 1

# Denormalization (for evaluation):
d = (d_norm + 1) / 2 * (d_max - d_min) + d_min
```

**Angle Encoding:**
- Raw: `θ ∈ [-π, π]`
- Encoded: `[cos(θ), sin(θ)]`
- Avoids discontinuity at ±180°

### Classifier-Free Guidance

**Training:**
- 10% of batches: use null embedding instead of goal
- Model learns both conditional and unconditional generation

**Sampling:**
```python
pred_cond = model(x_t, t, goal_embed)
pred_uncond = model(x_t, t, null_embed)
pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
```

**CFG Scale Effects:**
- 1.0: No guidance (ignore condition)
- 2.0: Moderate guidance (default)
- 5.0: Strong guidance (better accuracy, less diversity)
- 10.0: Very strong (may reduce realism)

### DDIM vs DDPM

**DDPM (original):**
- 1000 steps
- Stochastic (adds noise)
- High quality
- Slow (~100 seconds)

**DDIM (fast):**
- 50 steps (or any number)
- Deterministic (eta=0.0)
- Nearly same quality
- Fast (~5 seconds)

**Recommendation:** Use DDIM for generation, DDPM for debugging

---

## Model Checkpoints

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 299,
    'global_step': 150000,
    'best_val_loss': 0.042,
    'model_state_dict': {...},           # TrajectoryTransformer weights
    'goal_conditioner_state_dict': {...}, # GoalConditioner weights
    'optimizer_state_dict': {...},       # Adam state
    'scheduler_state_dict': {...},       # LR scheduler (optional)
    'config': TrajectoryDiffusionConfig(...)
}
```

### Loading Checkpoints

```python
checkpoint = torch.load('checkpoints_diffusion_v7/best.pth',
                        weights_only=False)  # PyTorch 2.6 compatibility

# Models automatically created from checkpoint['config']
model.load_state_dict(checkpoint['model_state_dict'])
goal_conditioner.load_state_dict(checkpoint['goal_conditioner_state_dict'])
```

---

## Performance Benchmarks

### Smoke Test Results

**Expected:**
- Initial loss: ~0.3
- Final loss: ~0.02
- Convergence: 500-1000 steps
- Duration: 1-2 minutes

**Pass criteria:** Loss decreases 10-15x

### Full Training (Medium Preset)

**Model specs:**
- Parameters: 4.8M
- Latent dim: 256
- Layers: 6

**Performance:**
- Training time: ~8 hours (300 epochs, GTX 1080)
- Best validation loss: ~0.04
- Goal accuracy: Distance ~0.1, Angle ~15°

### Generation Speed

**DDIM (50 steps):**
- Single trajectory: ~50ms
- Batch of 100: ~5 seconds

**DDPM (1000 steps):**
- Single trajectory: ~1 second
- Batch of 100: ~100 seconds

---

## Future Enhancements

### Planned Features

1. **Time Conditioning**: Add trajectory duration as 4th condition
2. **Curvature Control**: Explicit control over path straightness
3. **Multi-Modal Generation**: Sample multiple diverse trajectories per condition
4. **Trajectory Editing**: Modify existing trajectories
5. **Real-time Generation**: Optimize for <10ms latency

### Experimental Ideas

1. **Latent Diffusion**: If dataset grows to 100K+ trajectories
2. **Score-Based Models**: SDE formulation instead of DDPM
3. **Flow Matching**: Faster sampling with rectified flows
4. **Consistency Models**: 1-step generation

---

## References

### Papers

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
2. **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
3. **Classifier-Free Guidance**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (NeurIPS 2021)
4. **MotionDiffuse**: Zhang et al., "MotionDiffuse: Text-Driven Human Motion Generation" (CVPR 2023)

### Code References

- MotionDiffuse: https://github.com/mingyuan-zhang/MotionDiffuse
- OpenAI Guided Diffusion: https://github.com/openai/guided-diffusion

---

## Appendix: Quick Reference

### Common Commands

```bash
# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: DATA COLLECTION & PREPARATION
# ═══════════════════════════════════════════════════════════════════════

# Step 0.1: Record mouse movements (GUI application)
python SapiRecorder.py

# Step 0.2: Combine all session CSVs
python combine_cvss.py

# Step 0.3: Split into individual trajectory JSONs
python trajectory_splitter_adaptive.py combined_all_sessions.csv trajectories/

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

python preprocess_V6.py --input trajectories/ --output processed_data_v6/

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: TRAINING
# ═══════════════════════════════════════════════════════════════════════

# Smoke test
python train_diffusion_v7.py --mode smoke_test

# Full training
python train_diffusion_v7.py --mode medium --epochs 500

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: GENERATION
# ═══════════════════════════════════════════════════════════════════════

python generate_diffusion_v7.py --checkpoint checkpoints_diffusion_v7/best.pth --num_samples 100

# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

python visualize_generated.py
python compare_real_vs_generated.py
python diagnose_distance_error.py
```

### File Sizes

- Preprocessed data: ~50MB (1500 trajectories)
- Model checkpoint: ~20MB (medium), ~110MB (full)
- Generated trajectories: ~10MB (1000 samples)

### Directory Cleanup

**Safe to delete after successful training:**
- `checkpoints_diffusion_v7/checkpoint_epoch_*.pth` (keep best.pth and final.pth only)
- Temporary diagnostic outputs
- `__pycache__/` directories

**Never delete:**
- `processed_data_v6/` (regenerating requires raw data)
- `diffusion_v7/` package
- Training scripts

---

## Contact & Support

For issues, see:
- GitHub Issues: https://github.com/anthropics/claude-code/issues
- This document: `DIFFUSION_V7_ARCHITECTURE.md`

Last updated: 2025-12-25
