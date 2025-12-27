# Diffusion V7/V8 - Master Architecture Document

**Pure Diffusion Model for Mouse Trajectory Generation**

> ⚠️ **This document covers both V7 (broken) and V8 (fixed) architectures.**
> V7 used Classifier-Free Guidance for endpoint conditioning—this is fundamentally flawed.
> V8 uses **inpainting** instead. See [Critical Design Issue (V7)](#critical-design-issue-v7).

---

## Table of Contents

1. [Overview](#overview)
2. [Critical Design Issue (V7)](#critical-design-issue-v7) ⚠️ **READ THIS**
3. [V8 Architecture](#v8-architecture) ✅ **THE SOLUTION**
4. [Complete Pipeline](#complete-pipeline)
5. [Directory Structure](#directory-structure)
6. [Core Components](#core-components)
7. [Data Flow](#data-flow)
8. [File Classification](#file-classification)
9. [Usage Guide](#usage-guide)
10. [Troubleshooting](#troubleshooting)
11. [Appendix A: Migrating from V7](#appendix-a-migrating-from-v7)
12. [Appendix B: Design Rationale](#appendix-b-design-rationale)
13. [References](#references)

---

## Overview

### What is This Project?

A **pure diffusion model** (DDPM/DDIM) that generates realistic mouse cursor trajectories conditioned on endpoint position. Unlike latent diffusion, it operates directly in trajectory space.

### V7 vs V8 Summary

| Aspect | V7 (Broken) | V8 (Fixed) |
|--------|-------------|------------|
| Conditioning | CFG with goal embedding | **Inpainting** |
| Features | 8D (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt) | **2D (x, y)** |
| Training | 10% CFG dropout on goal condition | **No goal conditioning** |
| Inference | CFG formula: `uncond + scale × (cond - uncond)` | **Replace endpoints each step** |
| Direction control | Goal condition (unreliable) | **Endpoint position (guaranteed)** |

### Key Features (V8)

- **Pure Diffusion**: Works directly on 2D trajectory positions (x, y)
- **Inpainting Conditioning**: Endpoints fixed at each denoising step
- **No Goal Embedding**: Direction emerges from endpoint position
- **Variable Length**: Handles trajectories from 11-200 timesteps
- **DDIM Sampling**: Fast generation (50 steps vs 1000 DDPM steps)

### Why Pure Diffusion?

- V6 GAN has 50:1 compression (9,600D → 192D pooled latent)
- Risk of losing temporal details in aggressive compression
- Pure diffusion preserves full trajectory dynamics
- Reference: MotionDiffuse, MDM use pure diffusion for motion sequences

---

## Critical Design Issue (V7)

> ⚠️ **V7's CFG-based conditioning cannot reliably control trajectory direction.**
> This section documents why V7 is broken. Skip to [V8 Architecture](#v8-architecture) for the solution.

### The Problem: CFG + Directional Endpoints

V7 used Classifier-Free Guidance (CFG) for goal conditioning:

```
Training:  10% of batches → replace goal with null embedding
Inference: pred = uncond + scale × (cond - uncond)
```

CFG works brilliantly for **semantic** conditions (art style, text prompts) but **fails catastrophically** for **geometric/directional** conditions like trajectory endpoints.

### Why CFG Fails: The Averaging Problem

CFG requires a meaningful "unconditional" distribution. For mouse trajectories:

| Direction | dx trend | dy trend | heading |
|-----------|----------|----------|---------|
| East →    | positive | ~0       | 0°      |
| West ←    | negative | ~0       | 180°    |
| North ↑   | ~0       | negative | -90°    |
| South ↓   | ~0       | positive | +90°    |

**The unconditional distribution averages ALL directions:**

```
East trajectories:  dx → +100, dy → 0
West trajectories:  dx → -100, dy → 0
                    ─────────────────
Average:            dx → 0,    dy → 0   ← Goes NOWHERE
```

The "unconditional" prediction is a trajectory that doesn't move. CFG guidance from this meaningless baseline produces meaningless results.

### Triple Direction Encoding (V7)

V7's 8D features encoded direction **three separate times**:

| Encoding | Features | What it represents |
|----------|----------|-------------------|
| 1. Position | dx, dy | Cumulative displacement → direction |
| 2. Heading | sin(h), cos(h) | Instantaneous direction |
| 3. Condition | cos(θ), sin(θ) | Target direction |

During denoising, **nothing enforced consistency** between these encodings. The model could output:
- dx/dy implying the trajectory goes East
- sin/cos(heading) implying it points North
- Goal condition specifying Southeast

Result: internally contradictory outputs.

### Observable Symptoms of V7's Failure

If you trained V7 and saw any of these, now you know why:

- Trajectories curve unexpectedly or spiral
- Endpoint accuracy degrades as distance increases
- Increasing CFG scale doesn't improve direction accuracy
- Generated trajectories "average out" to short, directionless movements
- High variance in endpoint error despite high CFG scale

---

## V8 Architecture

> ✅ **The correct approach: Inpainting for geometric constraints.**
> Based on Diffuser (Janner et al., 2022) and MDM (Tevet et al., 2022).

### Core Principle: Separate Training from Constraint Satisfaction

| Phase | What happens | Goal conditioning? |
|-------|--------------|-------------------|
| **Training** | Model learns "what valid trajectories look like" | **NO** |
| **Inference** | Constraints applied via inpainting | Endpoint position fixed each step |

The model learns an **unconditional** distribution of valid mouse trajectories. Direction is not learned—it's enforced during generation by fixing the endpoint.

### Feature Representation (Simplified)

**V7 Features (8D) — Problematic:**
```
[dx, dy, speed, accel, sin(h), cos(h), angular_vel, dt]
 └──┬──┘              └────────┬────────┘
    │                          │
    └── Encodes direction      └── ALSO encodes direction → CONFLICT
```

**V8 Features (2D) — Clean:**
```
[x, y]   ← Position at each timestep, relative to start (0, 0)
```

**Why remove the other 6 features?**

| Removed Feature | Reason |
|-----------------|--------|
| speed | Derivable from position differences and dt |
| acceleration | Derivable from speed differences |
| sin(heading), cos(heading) | Derivable from dx/dy; creates consistency conflicts |
| angular_velocity | Derivable from heading differences |
| dt | Constant at 8ms; if variable, add back as 3rd dimension |

Derived features cause **denoising inconsistency**: the model might output (x, y) implying heading=45° while simultaneously outputting sin/cos(heading) implying heading=60°. Nothing enforces physical consistency.

With 2D (x, y), the trajectory is self-consistent by construction.

### Training Procedure (V8)

```python
# V8: NO goal conditioning, NO CFG
for batch in dataloader:
    x = batch['positions']        # (B, L, 2)
    lengths = batch['lengths']    # (B,)

    t = sample_timesteps(B)
    noise = randn_like(x)
    x_noisy = sqrt(alpha[t]) * x + sqrt(1 - alpha[t]) * noise

    noise_pred = model(x_noisy, t, lengths)  # No goal input!
    loss = mse(noise_pred, noise, mask=length_mask)
```

The model learns trajectory dynamics (smoothness, acceleration profiles, natural curvature) without any directional bias.

### Inference Procedure: Inpainting

```python
def generate(start_pos, end_pos, length, steps=50):
    x = randn(1, max_len, 2)                    # Start from noise

    for t in reversed(timesteps):
        noise_pred = model(x, t, length)
        x_0_pred = predict_x0(x, t, noise_pred)

        # INPAINTING: Fix endpoints
        x_0_pred[0, 0] = start_pos              # Usually (0, 0)
        x_0_pred[0, length-1] = end_pos         # Target position

        if t > 0:
            x = add_noise(x_0_pred, t-1)        # Re-noise for next step
        else:
            x = x_0_pred                        # Final output

    return x[0, :length]
```

**Key details:**
- Inpaint at **actual length** (`length-1`), not padded length (199)
- Fix both start and end at every denoising step
- Direction emerges naturally from the endpoint position

> 💡 **If endpoint artifacts appear** (kinks, sudden direction changes near the end), consider inpainting the first/last 2-3 positions with interpolated constraints. But start with just endpoints.

### Why Not Canonical Frame?

We considered rotating all trajectories to face East during preprocessing, then rotating outputs back at inference.

**Rejected** because human biomechanics differ by direction:

| Direction | Primary Movement |
|-----------|-----------------|
| East (→)  | Wrist abduction, forearm rotation |
| West (←)  | Wrist adduction |
| North (↑) | Wrist extension, shoulder involvement |
| South (↓) | Wrist flexion |

A North trajectory rotated to face East has East-direction **positions** but North-direction **dynamics** (the original acceleration and jerk profiles). This corrupts training data with physically inconsistent examples.

**Decision**: Keep world coordinates. Inpainting handles direction implicitly.

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

> **V8 change**: Outputs 2D positions `(x, y)` instead of 8D features. No conditions file.

```
Raw Data (trajectories/*.json)
    ↓
preprocess_V8.py
    │
    ├── Step 1: Load raw trajectories (x, y, t arrays)
    │
    ├── Step 2: Filter bad recordings
    │   ├── Length > 195 points (would be truncated)
    │   ├── Distance ratio > 3.0x (erratic/distracted path)
    │   └── Ideal distance < 20px (too short)
    │
    ├── Step 3: Convert to relative positions
    │   └── x_rel = x - x[0], y_rel = y - y[0]
    │
    ├── Step 4: Normalize to [-1, 1]
    │   └── Based on max displacement in training set
    │
    ├── Step 5: Pad to fixed length (200)
    │   └── Post-padding: [real, real, ..., 0, 0, 0]
    │
    ├── Step 6: Store endpoint positions (for evaluation)
    │   └── endpoints[i] = (x[-1], y[-1]) before normalization
    │
    └── Step 7: Create train/val/test splits (60/20/20)
    ↓
processed_data_v8/
    ├── X_train.npy              (positions: N × 200 × 2)
    ├── L_train.npy              (lengths: N)
    ├── endpoints_train.npy      (original endpoints: N × 2, for evaluation)
    ├── X_val.npy, L_val.npy, endpoints_val.npy
    ├── X_test.npy, L_test.npy, endpoints_test.npy
    └── normalization_params.npy (position_scale only)
```

**V8 Features (2D):**
| Feature | Description |
|---------|-------------|
| `x` | Relative x position from start (normalized to [-1, 1]) |
| `y` | Relative y position from start (normalized to [-1, 1]) |

**No Conditions File**: Training is unconditional. Endpoints are stored separately for evaluation only.

<details>
<summary>V7 Preprocessing (deprecated)</summary>

V7 used 8D features and 3D conditions:
- Features: `[dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt]`
- Conditions: `[distance_norm, cos(angle), sin(angle)]`
- Output: `X_*.npy` (N × 200 × 8), `C_*.npy` (N × 3)

This is **deprecated** due to the CFG problem described above.
</details>

### Phase 2: Model Training

> **V8 change**: No goal conditioning. No CFG dropout. Model learns unconditional trajectory distribution.

```
processed_data_v8/
    ↓
train_diffusion_v8.py --mode [smoke_test|medium|full]
    ↓
checkpoints_diffusion_v8/
    ├── best.pth              (lowest validation loss)
    ├── final.pth             (last epoch)
    └── checkpoint_epoch_N.pth
```

**Training Process:**
1. **Smoke Test**: Single-batch overfitting (64 samples, 1000 steps)
   - Validates architecture works
   - Loss should decrease 10-15x
2. **Full Training**: Train on full dataset
   - 300+ epochs typical
   - Saves best.pth when validation loss improves

**What the model learns**: Valid trajectory dynamics—smoothness, acceleration profiles, natural curvature. No directional bias because no direction is specified during training.

<details>
<summary>V7 Training (deprecated)</summary>

V7 used CFG with 10% dropout on goal conditions. This is deprecated.
</details>

### Phase 3: Generation with Inpainting

> **V8 change**: Direction controlled by inpainting endpoint at each denoising step.

```
checkpoints_diffusion_v8/best.pth + target endpoint (x, y)
    ↓
generate_diffusion_v8.py --target_x 0.5 --target_y 0.3 --length 100
    ↓
results/diffusion_v8/
    ├── generated_trajectories.npy   (N × max_len × 2)
    ├── generation_endpoints.npy     (N × 2)
    ├── generation_lengths.npy       (N,)
    ├── evaluation_metrics.json
    └── generation_info.json
```

**Inpainting Process** (at each of 50 DDIM steps):
1. Predict clean trajectory from noise
2. Replace position[0] with start (0, 0)
3. Replace position[length-1] with target endpoint
4. Re-noise for next step

**Generation Methods:**
- **DDIM** (default): 50 steps, deterministic
- **DDPM**: 1000 steps, stochastic

**Evaluation Metrics:**
- **Endpoint Accuracy**: Euclidean distance to target (should be ~0 with inpainting)
- **Realism**: Speed/acceleration distributions, smoothness (jerk)
- **Diversity**: Pairwise trajectory distances for same endpoint

### Phase 4: Visualization

```
results/diffusion_v8/
    ↓
visualize_generated.py OR create_evaluation_report()
    ↓
Diagnostic plots for analysis
```

---

## Directory Structure

> **V8 changes**: New `diffusion_v8/` package. Removes `goal_conditioner.py`, adds `inpainting_sampler.py`.

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
│   └── ...
│
├── # ─────────────────────────────────────────────────────────────────
├── # DIFFUSION MODEL V8 (Phase 1-4)
├── # ─────────────────────────────────────────────────────────────────
├── diffusion_v8/                    # V8 package (inpainting-based)
│   ├── __init__.py
│   ├── config_trajectory.py         # Configuration (no CFG params)
│   │
│   ├── models/                      # Model architectures
│   │   ├── __init__.py
│   │   ├── trajectory_transformer.py # Denoising model (time-only conditioning)
│   │   └── gaussian_diffusion.py    # DDPM/DDIM (from MotionDiffuse)
│   │   # NOTE: No goal_conditioner.py - removed in V8
│   │
│   ├── datasets/                    # Data loading
│   │   ├── __init__.py
│   │   └── trajectory_dataset.py    # Loads positions (X) and lengths (L)
│   │
│   ├── trainers/                    # Training loops
│   │   ├── __init__.py
│   │   └── trajectory_trainer.py    # Unconditional training
│   │
│   ├── sampling/                    # Generation
│   │   ├── __init__.py
│   │   └── inpainting_sampler.py    # DDIM/DDPM with endpoint inpainting
│   │   # NOTE: Replaces trajectory_sampler.py (CFG-based)
│   │
│   └── evaluation/                  # Metrics & visualization
│       ├── __init__.py
│       ├── metrics.py               # Evaluation metrics
│       └── visualize.py             # Plotting utilities
│
├── diffusion_v7/                    # V7 package (DEPRECATED - CFG-based)
│   └── ...                          # Keep for reference, do not use
│
├── processed_data_v8/               # V8 preprocessed data (2D positions)
│   ├── X_train.npy, L_train.npy, endpoints_train.npy
│   ├── X_val.npy, L_val.npy, endpoints_val.npy
│   ├── X_test.npy, L_test.npy, endpoints_test.npy
│   └── normalization_params.npy
│
├── processed_data_v6/               # V7 preprocessed data (DEPRECATED)
│   └── ...                          # 8D features + conditions
│
├── checkpoints_diffusion_v8/        # V8 saved models
│   ├── best.pth
│   ├── final.pth
│   └── checkpoint_epoch_*.pth
│
├── results/                         # Generated outputs
│   └── diffusion_v8/
│       ├── generated_trajectories.npy
│       ├── generation_endpoints.npy
│       ├── evaluation_metrics.json
│       └── plots/
│
├── preprocess_V8.py                 # V8 preprocessing (2D positions)
├── train_diffusion_v8.py            # V8 training CLI
├── generate_diffusion_v8.py         # V8 generation CLI (inpainting)
├── inpainting_reference.py          # Reference implementation (see Appendix)
│
├── # DEPRECATED V7 scripts (keep for reference):
├── preprocess_V6.py
├── train_diffusion_v7.py
├── generate_diffusion_v7.py
│
└── Diagnostic Scripts:
    ├── visualize_generated.py
    ├── compare_real_vs_generated.py
    └── check_norm_params.py
```

---

## Core Components

> **V8 changes**: Simplified configuration (no CFG), removed GoalConditioner, simplified transformer (time-only conditioning), new InpaintingSampler.

### 1. Configuration (`config_trajectory.py`)

**V8 Configuration:**

```python
@dataclass
class TrajectoryDiffusionConfig:
    # ═══════════════════════════════════════════════════════════════════
    # DATA CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    input_dim: int = 2              # V8: (x, y) positions only
    max_seq_len: int = 200          # Maximum trajectory length
    # NOTE: condition_dim removed - no goal conditioning in V8

    # ═══════════════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════
    latent_dim: int = 512           # Transformer hidden dimension
    num_layers: int = 9             # Number of transformer blocks
    num_heads: int = 8              # Attention heads
    ff_size: int = 1024             # Feedforward expansion size
    dropout: float = 0.1            # Dropout rate
    activation: str = 'gelu'        # Activation function

    # ═══════════════════════════════════════════════════════════════════
    # CFG REMOVED IN V8 - These parameters no longer exist
    # ═══════════════════════════════════════════════════════════════════
    # use_cfg: bool                 # REMOVED
    # cfg_dropout: float            # REMOVED
    # cfg_guidance_scale: float     # REMOVED

    # ═══════════════════════════════════════════════════════════════════
    # DIFFUSION PROCESS
    # ═══════════════════════════════════════════════════════════════════
    diffusion_steps: int = 1000     # Number of diffusion timesteps (T)
    noise_schedule: str = 'linear'  # Beta schedule: 'linear', 'cosine'
    beta_start: float = 1e-4
    beta_end: float = 0.02
    model_mean_type: str = 'epsilon'    # Predict noise
    model_var_type: str = 'fixed_small'
    loss_type: str = 'mse'

    # ═══════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 0.5
    num_epochs: int = 1000
    optimizer: str = 'adam'

    # ═══════════════════════════════════════════════════════════════════
    # SAMPLING (INPAINTING)
    # ═══════════════════════════════════════════════════════════════════
    sampling_method: str = 'ddim'
    ddim_steps: int = 50
    ddim_eta: float = 0.0           # 0 = deterministic
    inpaint_start: bool = True      # Fix start position during generation
    inpaint_end: bool = True        # Fix end position during generation

    # ═══════════════════════════════════════════════════════════════════
    # DATA & CHECKPOINTS
    # ═══════════════════════════════════════════════════════════════════
    data_dir: str = 'processed_data_v8'
    save_dir: str = 'checkpoints_diffusion_v8'
    log_dir: str = 'logs_diffusion_v8'
```

**Model Presets:**

| Preset | latent_dim | num_layers | num_heads | ff_size | ~Parameters |
|--------|------------|------------|-----------|---------|-------------|
| `smoke_test()` | 128 | 4 | 4 | 256 | ~500K |
| `medium()` | 256 | 6 | 8 | 512 | ~3M |
| `full()` | 512 | 9 | 8 | 1024 | ~20M |

*Note: V8 models are smaller than V7 because input_dim=2 instead of 8.*

<details>
<summary>V7 Configuration (deprecated)</summary>

V7 included CFG parameters that are now removed:
- `condition_dim: int = 3` - for goal conditioning
- `use_cfg: bool = True`
- `cfg_dropout: float = 0.1`
- `cfg_guidance_scale: float = 2.0`

These are deprecated due to the CFG problem.
</details>

### 2. Goal Conditioner — REMOVED IN V8

> ⚠️ `goal_conditioner.py` does not exist in V8. Direction is controlled via inpainting, not learned embeddings.

<details>
<summary>V7 Goal Conditioner (deprecated)</summary>

V7 embedded the 3D goal condition `[distance_norm, cos(θ), sin(θ)]` into latent space and used CFG during sampling. This is deprecated because CFG fails for directional conditioning.
</details>

### 3. Trajectory Transformer (`trajectory_transformer.py`)

> **V8 change**: Time-only conditioning. No goal embedding input.

**V8 Architecture:**
```
Input: x_noisy (batch, seq_len, 2), t (batch,), lengths (batch,)
       └── Note: 2D positions, no goal_embed

┌────────────────────────────────────────────────────────────────────┐
│  INPUT EMBEDDING                                                    │
│  x_noisy (batch, seq_len, 2)                                       │
│      ↓ Linear(2 → latent_dim)                                      │
│  h (batch, seq_len, latent_dim)                                    │
│      + pos_encoding (learnable)                                    │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TIME EMBEDDING ONLY (no goal embedding)                            │
│  t (batch,) → sinusoidal → MLP → t_emb (batch, latent_dim)         │
│  cond_emb = t_emb  ← V8: Just time, not concat with goal           │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TRANSFORMER BLOCKS (×num_layers)                                   │
│  TemporalSelfAttention(h, cond_emb, mask)                          │
│  FeedForward(h, cond_emb) with FiLM conditioning                   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Linear(latent_dim → 2)                                     │
│  noise_pred (batch, seq_len, 2)                                    │
└────────────────────────────────────────────────────────────────────┘
```

**V8 Forward Signature:**
```python
def forward(self, x, t, lengths) -> noise_pred:
    # x: (batch, seq_len, 2) - noisy positions
    # t: (batch,) - diffusion timestep
    # lengths: (batch,) - actual sequence lengths
    # Returns: (batch, seq_len, 2) - predicted noise
```

### 4. Gaussian Diffusion (`gaussian_diffusion.py`)

Unchanged from MotionDiffuse (OpenAI's DDPM implementation).

**Features:**
- Beta schedules: linear, cosine, sqrt
- Model mean types: epsilon, x0, previous_x
- DDIM sampling support

### 5. Inpainting Sampler (`inpainting_sampler.py`)

> **V8**: Replaces `trajectory_sampler.py`. No CFG—direction controlled via endpoint inpainting.

```python
class InpaintingSampler:
    def generate(self, start_pos, end_pos, length,
                 method='ddim', ddim_steps=50, ddim_eta=0.0)
```

**Inpainting Algorithm (pseudocode):**
```python
x = randn(1, max_len, 2)              # Start from noise
for t in reversed(timesteps):
    noise_pred = model(x, t, length)
    x_0_pred = predict_x0(x, t, noise_pred)

    # INPAINTING: Replace endpoints
    x_0_pred[0, 0] = start_pos        # Fix start
    x_0_pred[0, length-1] = end_pos   # Fix end

    x = add_noise(x_0_pred, t-1) if t > 0 else x_0_pred
return x[0, :length]
```

**Key difference from V7**: No CFG formula. Direction comes from the endpoint position, not from a learned embedding.

**Convenience Method:**
```python
trajectory = sampler.generate_single(
    target_x=0.5,    # Normalized endpoint x
    target_y=0.3,    # Normalized endpoint y
    length=100
)  # Returns: (length, 2)
```

<details>
<summary>V7 Trajectory Sampler (deprecated)</summary>

V7 used CFG-based sampling with goal embeddings. This is deprecated.
</details>

---

## Data Flow

> **V8 changes**: No goal conditioning during training. Inpainting during generation.

### Training Data Flow (V8)

```
JSON files → preprocess_V8.py → .npy files (2D positions)
    ↓
TrajectoryDataset loads (X, L)  ← No conditions
    ↓
TrajectoryDiffusionTrainer:
    1. Sample timestep t
    2. Forward diffusion: x_t = √(α_t)·x_0 + √(1-α_t)·ε
    3. TrajectoryTransformer: predict ε from (x_t, t)  ← No goal!
    4. Loss: MSE(predicted_ε, actual_ε)
    5. Backprop & optimize
```

### Generation Data Flow (V8)

```
Load checkpoint + specify target endpoint (target_x, target_y)
    ↓
InpaintingSampler:
    1. Start from pure noise: x_T ~ N(0, I)
    2. For t = T...0:
        a. noise_pred = model(x_t, t)  ← No goal embedding
        b. x_0_pred = predict_clean(x_t, noise_pred)
        c. x_0_pred[0] = (0, 0)           ← INPAINT START
        d. x_0_pred[L-1] = (tx, ty)       ← INPAINT END
        e. x_{t-1} = add_noise(x_0_pred) if t > 0
    3. Return x_0[:L]
    ↓
Evaluate metrics
```

<details>
<summary>V7 Data Flow (deprecated)</summary>

V7 used goal conditioning during training (10% dropout) and CFG during generation. This is deprecated.
</details>

---

## File Classification

> **V8 changes**: New package `diffusion_v8/`, new scripts. V7 files kept for reference.

### ✅ ESSENTIAL (V8) - DO NOT DELETE

**Data Collection & Preparation (Phase 0):**
- `SapiRecorder.py` - GUI for recording mouse movements (125Hz sampling)
- `combine_cvss.py` - Combine multiple recording session CSVs
- `trajectory_splitter_adaptive.py` - Split combined CSV into individual trajectory JSONs

**V8 Core Package:**
- `diffusion_v8/` (entire directory)
  - `config_trajectory.py` - V8 configuration (no CFG)
  - `models/trajectory_transformer.py` - Time-only conditioning
  - `models/gaussian_diffusion.py` - DDPM/DDIM
  - `datasets/trajectory_dataset.py` - Loads 2D positions
  - `trainers/trajectory_trainer.py` - Unconditional training
  - `sampling/inpainting_sampler.py` - Endpoint inpainting
  - `evaluation/metrics.py`, `evaluation/visualize.py`
  - **NOTE**: No `goal_conditioner.py` in V8

**V8 Scripts:**
- `preprocess_V8.py` - V8 preprocessing (2D positions)
- `train_diffusion_v8.py` - V8 training CLI
- `generate_diffusion_v8.py` - V8 generation CLI (inpainting)
- `inpainting_reference.py` - Reference implementation

**V8 Data:**
- `processed_data_v8/` - 2D position data
- `checkpoints_diffusion_v8/` - V8 trained models

### ⚠️ DEPRECATED (V7) - Keep for Reference

**V7 Package (CFG-based, broken):**
- `diffusion_v7/` - Keep for reference, do not use
- `preprocess_V6.py`, `train_diffusion_v7.py`, `generate_diffusion_v7.py`
- `processed_data_v6/`, `checkpoints_diffusion_v7/`

### 🔧 USEFUL - Keep for Development

**Testing & Validation:**
- `test_diffusion_v8_architecture.py` - Validates V8 model works

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

**V6 GAN files** (if still in use):
- Keep: `embedder_v6.py`, `recovery_v6.py`, `model_v6.py`, `pooler_v6.py`

**MotionDiffuse legacy files** (in `diffusion_v7/`):
- `gaussian_diffusion.py` is copied to V8 and works correctly
- Other MotionDiffuse files have unmet dependencies and are not used

**Active V8 code (what you should edit):**
- `diffusion_v8/config_trajectory.py`
- `diffusion_v8/models/trajectory_transformer.py`
- `diffusion_v8/models/gaussian_diffusion.py`
- `diffusion_v8/datasets/trajectory_dataset.py`
- `diffusion_v8/trainers/trajectory_trainer.py`
- `diffusion_v8/sampling/inpainting_sampler.py`
- `diffusion_v8/evaluation/metrics.py`, `evaluation/visualize.py`

---

## Usage Guide

> **V8 commands shown below.** V7 commands are deprecated.

### 0. Data Collection & Preparation (Phase 0)

**Step 0.1: Record Mouse Movements**
```bash
python SapiRecorder.py
```
- GUI window opens with two dots (A=start, B=target)
- Click dot A → Move mouse → Click dot B
- Each session: 192 trajectories (24 distances × 8 directions)
- Output: `D:\V6\user0001\session_*.csv`

**Step 0.2: Combine Recording Sessions**
```bash
python combine_cvss.py
```
Output: `combined_all_sessions.csv`

**Step 0.3: Split into Individual Trajectories**
```bash
python trajectory_splitter_adaptive.py combined_all_sessions.csv trajectories/
```
Output: `trajectories/trajectory_NNNN.json`

### 1. Preprocessing (V8)

```bash
python preprocess_V8.py --input trajectories/ --output processed_data_v8/
```

**Output:**
- `processed_data_v8/X_{train,val,test}.npy` - 2D positions (N × 200 × 2)
- `processed_data_v8/L_{train,val,test}.npy` - Lengths
- `processed_data_v8/endpoints_{train,val,test}.npy` - For evaluation
- `processed_data_v8/normalization_params.npy`

**Note**: No `C_*.npy` files—V8 doesn't use goal conditioning during training.

### 2. Training (V8)

**Smoke Test:**
```bash
python train_diffusion_v8.py --mode smoke_test
```
Expected: Loss decreases from ~0.3 to ~0.02 in 1000 steps

**Full Training:**
```bash
python train_diffusion_v8.py --mode medium --epochs 500
```

**Checkpoints:**
- `checkpoints_diffusion_v8/best.pth` - Lowest validation loss
- `checkpoints_diffusion_v8/final.pth` - Last epoch

### 3. Generation with Inpainting (V8)

**Generate to specific endpoint:**
```bash
python generate_diffusion_v8.py \
    --checkpoint checkpoints_diffusion_v8/best.pth \
    --target_x 0.5 --target_y 0.3 \
    --length 100
```

**Batch generation from test endpoints:**
```bash
python generate_diffusion_v8.py \
    --checkpoint checkpoints_diffusion_v8/best.pth \
    --use_test_endpoints \
    --num_samples 100
```

**Output:**
- `results/diffusion_v8/generated_trajectories.npy`
- `results/diffusion_v8/generation_endpoints.npy`
- `results/diffusion_v8/evaluation_metrics.json`

**Note**: No `--cfg_scale` parameter—V8 uses inpainting, not CFG.

### 4. Visualization

```python
from diffusion_v8.evaluation import create_evaluation_report
import numpy as np

trajectories = np.load('results/diffusion_v8/generated_trajectories.npy')
endpoints = np.load('results/diffusion_v8/generation_endpoints.npy')
lengths = np.load('results/diffusion_v8/generation_lengths.npy')

create_evaluation_report(trajectories, endpoints, lengths,
                         output_dir='results/diffusion_v8/plots')
```

**Diagnostic scripts:**
```bash
python visualize_generated.py
python compare_real_vs_generated.py
```

---

## Evaluation Metrics (`evaluation/metrics.py`)

> **V8 change**: With inpainting, endpoint accuracy should be nearly perfect. Focus on realism metrics.

### Endpoint Accuracy (V8)

```python
# With inpainting, endpoint is fixed at each step
# Error should be ~0 (numerical precision only)
final_x, final_y = trajectory[length-1, 0:2]
target_x, target_y = target_endpoint
error = sqrt((final_x - target_x)² + (final_y - target_y)²)
# Expected: < 0.001 (essentially zero)
```

### Realism Metrics

```python
# Derive features from 2D positions:
positions = trajectory[:length]           # (L, 2)
velocities = diff(positions, axis=0)      # (L-1, 2)
speeds = norm(velocities, axis=1)         # (L-1,)
accels = diff(speeds)                     # (L-2,)
jerk = diff(accels)                       # Smoothness metric

# Output metrics:
# - gen_speed_mean, gen_speed_std
# - gen_smoothness_mean (lower = smoother)
```

### Diversity Metrics

```python
# Generate multiple trajectories to same endpoint
# Compute pairwise distances
# Output: diversity_mean (higher = more diverse)
```

### Interpreting Results (V8)

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| `endpoint_error` | ~0 | < 0.01 | > 0.1 (inpainting broken) |
| `smoothness` | < 0.01 | < 0.05 | > 0.1 |
| `diversity_mean` | > 0.001 | > 0.0001 | = 0 (mode collapse) |

---

## Troubleshooting

> **V8**: Most CFG-related issues are eliminated. Focus on inpainting and realism.

### V8-Specific Issues

**1. Endpoint Not Reached Exactly**

With proper inpainting, this shouldn't happen. Check:
- Inpainting applied at correct position (`length-1`, not `max_length-1`)
- Inpainting applied at **every** denoising step
- Final step returns `x_0_pred` without re-noising

**2. Kink/Artifact at Endpoint**

**Symptom:** Trajectory has sudden direction change near the end

**Cause:** Model struggles to smoothly approach the fixed endpoint

**Fix options:**
- Train longer to learn smoother dynamics
- Consider inpainting last 2-3 positions with interpolated values
- Check if real data has similar artifacts (may be learning real behavior)

**3. Trajectories Too Straight**

**Symptom:** Generated trajectories are nearly linear

**Diagnosis:**
```bash
python compare_real_vs_generated.py
```

**If real data is also straight:** This is correct—model learned the distribution.

**If only generated is straight:**
- Increase model capacity (`--mode full`)
- Train longer (500+ epochs)
- Check diversity metrics for mode collapse

### General Issues

**4. Training Loss Not Decreasing**

- Check data loading (shapes should be `(N, 200, 2)`)
- Verify gradient flow
- Check learning rate (default: 1e-4)

**5. PyTorch 2.6 Loading Error**

**Fix:** Use `weights_only=False` in `torch.load()`

**6. Mode Collapse (diversity_mean ≈ 0)**

All trajectories look identical despite different endpoints:
- Check model capacity
- Verify training data has variety
- Ensure noise is being added properly during generation

---

## Technical Details

### Normalization (V8)

**Position Normalization:**
```python
# All positions relative to start (0, 0)
# Normalized to [-1, 1] based on max displacement
x_norm = x / position_scale
y_norm = y / position_scale

# Denormalization:
x = x_norm * position_scale
y = y_norm * position_scale
```

### Inpainting (V8)

**How it works:**
At each denoising step, the model predicts what the clean trajectory should look like. We then **replace** the start and end positions with our constraints before adding noise for the next step.

```python
x_0_pred = denoise(x_t, t)          # Model's prediction
x_0_pred[0] = start_position         # Override start
x_0_pred[length-1] = end_position    # Override end
x_t_minus_1 = add_noise(x_0_pred)    # Noise for next step
```

Over many steps, the model learns to generate trajectories consistent with these fixed endpoints.

<details>
<summary>V7 CFG (deprecated)</summary>

V7 used Classifier-Free Guidance which is inappropriate for directional conditioning. See "Critical Design Issue (V7)" section.
</details>

### DDIM vs DDPM

**DDPM (original):**
- 1000 steps, stochastic, slow (~100 seconds)

**DDIM (fast):**
- 50 steps, deterministic (eta=0), fast (~5 seconds)
- Nearly same quality

**Recommendation:** Use DDIM for generation

---

## Model Checkpoints

### V8 Checkpoint Contents

```python
checkpoint = {
    'epoch': 299,
    'global_step': 150000,
    'best_val_loss': 0.042,
    'model_state_dict': {...},           # TrajectoryTransformer weights
    # NOTE: No goal_conditioner in V8
    'optimizer_state_dict': {...},
    'config': TrajectoryDiffusionConfig(...)
}
```

### Loading V8 Checkpoints

```python
checkpoint = torch.load('checkpoints_diffusion_v8/best.pth',
                        weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
# No goal_conditioner to load in V8
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

### Core Papers

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
2. **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
3. **Diffuser**: Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" (ICML 2022)
   - **Key insight**: Inpainting for trajectory endpoint constraints
   - This paper established the approach V8 uses
4. **MDM**: Tevet et al., "Human Motion Diffusion Model" (ICLR 2023)
   - Uses inpainting for keyframe constraints in human motion
5. **CFG**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (NeurIPS 2021)
   - Useful for semantic conditions, NOT geometric/directional constraints

### Background

- **MotionDiffuse**: Zhang et al., "MotionDiffuse: Text-Driven Human Motion Generation" (CVPR 2023)
  - Source of `gaussian_diffusion.py` and transformer architecture

### Code References

- Diffuser: https://github.com/jannerm/diffuser
- MotionDiffuse: https://github.com/mingyuan-zhang/MotionDiffuse

---

## Appendix A: Quick Reference (V8)

### Common Commands

```bash
# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════

python SapiRecorder.py                                              # Record
python combine_cvss.py                                              # Combine
python trajectory_splitter_adaptive.py combined.csv trajectories/   # Split

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1-3: V8 PIPELINE
# ═══════════════════════════════════════════════════════════════════════

python preprocess_V8.py --input trajectories/ --output processed_data_v8/
python train_diffusion_v8.py --mode medium --epochs 500
python generate_diffusion_v8.py --checkpoint best.pth --target_x 0.5 --target_y 0.3

# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

python visualize_generated.py
python compare_real_vs_generated.py
```

### File Sizes

- Preprocessed data: ~20MB (1500 trajectories, 2D vs 8D)
- Model checkpoint: ~15MB (medium), ~80MB (full)
- Generated trajectories: ~5MB (1000 samples)

---

## Appendix B: Migrating from V7 to V8

### Code Changes Required

1. **Remove GoalConditioner**
   - Delete `diffusion_v7/models/goal_conditioner.py`
   - Remove all imports and usages

2. **Update TrajectoryTransformer**
   - Change `input_dim: 8 → 2`
   - Remove `goal_embed` from forward signature
   - `cond_emb = time_emb` only (no goal concatenation)

3. **Replace TrajectorySampler with InpaintingSampler**
   - Remove CFG logic
   - Add endpoint replacement at each denoising step

4. **Update Preprocessing**
   - Output `(x, y)` only, not 8D features
   - Remove condition arrays (`C_*.npy`)

5. **Update Training Script**
   - Remove CFG dropout logic
   - Remove goal conditioning

### Checkpoint Incompatibility

V7 checkpoints are **NOT compatible** with V8:
- Different input dimensions (8 vs 2)
- Different conditioning (goal_embed vs none)

**You must retrain from scratch.**

---

## Appendix C: Design Rationale

### Why Inpainting Over CFG?

| Approach | Use Case | Problem with Trajectories |
|----------|----------|--------------------------|
| CFG | Semantic conditions (style, mood) | Unconditional = average of all directions = meaningless |
| Inpainting | Geometric constraints (endpoints) | ✓ Works perfectly |

### Why 2D Features Over 8D?

| Feature Set | Pros | Cons |
|-------------|------|------|
| 8D (V7) | Rich information | Denoising inconsistency, redundant encoding |
| 2D (V8) | Self-consistent, simple | Must derive speed/heading post-hoc |

The 2D representation cannot have internal contradictions—the trajectory is defined solely by positions.

### Why Not Canonical Frame?

Human biomechanics differ by direction. A North trajectory rotated to face East has correct positions but wrong dynamics (acceleration profiles, muscle engagement patterns).

---

## Contact & Support

- This document: `DIFFUSION_V7_ARCHITECTURE.md`
- Reference implementation: `inpainting_reference.py`

Last updated: 2025-12-27
