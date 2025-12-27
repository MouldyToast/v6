# Diffusion V8 Architecture

**Pure Diffusion Model for Mouse Trajectory Generation using Inpainting**

---

## Table of Contents

1. [Overview](#overview)
2. [Critical Design Issue (V7)](#critical-design-issue-v7) - Why V7 failed
3. [V8 Architecture](#v8-architecture) - The solution
4. [Pipeline](#pipeline)
5. [Core Components](#core-components)
6. [Quick Reference](#quick-reference)
7. [Troubleshooting](#troubleshooting)
8. [Migration from V7](#migration-from-v7)
9. [References](#references)

---

## Overview

A **pure diffusion model** (DDPM/DDIM) generating mouse trajectories conditioned on endpoint position via **inpainting**.

### V7 vs V8

| Aspect | V7 (Broken) | V8 (Fixed) |
|--------|-------------|------------|
| Conditioning | CFG with goal embedding | **Inpainting** |
| Features | 8D | **2D (x, y)** |
| Training | 10% CFG dropout | **No goal conditioning** |
| Direction control | Goal condition (unreliable) | **Endpoint position (guaranteed)** |

---

## Critical Design Issue (V7)

> ⚠️ **V7's CFG cannot reliably control trajectory direction.**

### The Problem

CFG requires a meaningful "unconditional" distribution. For trajectories going in all directions:

```
East trajectories:  dx → +100
West trajectories:  dx → -100
─────────────────────────────
Average (unconditional): dx → 0  ← Goes NOWHERE
```

The unconditional prediction is meaningless. CFG guidance from this baseline produces meaningless results.

### Triple Direction Encoding

V7 encoded direction three times (features dx/dy, features sin/cos heading, condition cos/sin angle). Nothing enforced consistency—the model could output contradictory directions.

---

## V8 Architecture

> ✅ **Inpainting for geometric constraints** (Diffuser, Janner et al. 2022)

### Core Principle

| Phase | Goal conditioning? |
|-------|-------------------|
| **Training** | NO - learns "what valid trajectories look like" |
| **Inference** | Endpoints fixed via inpainting each step |

### Features: 2D Only

```
V7: [dx, dy, speed, accel, sin(h), cos(h), angular_vel, dt]  ← Conflicts
V8: [x, y]  ← Self-consistent positions
```

Derived features (speed, heading) cause denoising inconsistency. 2D positions cannot contradict themselves.

### Training (Pseudocode)

```python
for batch in dataloader:
    x = batch['positions']        # (B, L, 2)
    t = sample_timesteps(B)
    noise = randn_like(x)
    x_noisy = sqrt(alpha[t]) * x + sqrt(1 - alpha[t]) * noise
    noise_pred = model(x_noisy, t, lengths)  # No goal!
    loss = mse(noise_pred, noise)
```

### Inference: Inpainting (Pseudocode)

```python
def generate(end_pos, length, steps=50):
    x = randn(1, max_len, 2)
    for t in reversed(timesteps):
        x_0_pred = denoise(x, t)
        x_0_pred[0, 0] = (0, 0)           # Fix start
        x_0_pred[0, length-1] = end_pos   # Fix end
        x = add_noise(x_0_pred, t-1) if t > 0 else x_0_pred
    return x[0, :length]
```

> 💡 If endpoint artifacts appear, consider inpainting last 2-3 positions with interpolation.

### Why Not Canonical Frame?

Human biomechanics differ by direction. A North trajectory rotated to East has correct positions but wrong dynamics.

---

## Pipeline

### Phase 0: Data Collection
```
SapiRecorder.py → session CSVs
combine_cvss.py → combined CSV
trajectory_splitter_adaptive.py → trajectory JSONs
```

### Phase 1: Preprocessing
```
trajectories/*.json → preprocess_V8.py → processed_data_v8/
    X_train.npy  (N × 200 × 2)
    L_train.npy  (N,)
    endpoints_train.npy  (N × 2, for evaluation)
    normalization_params.npy
```

### Phase 2: Training
```
train_diffusion_v8.py --mode medium --epochs 500
    → checkpoints_diffusion_v8/best.pth
```

### Phase 3: Generation
```
generate_diffusion_v8.py --target_x 0.5 --target_y 0.3 --length 100
    → results/diffusion_v8/generated_trajectories.npy
```

---

## Core Components

### Configuration

```python
@dataclass
class TrajectoryDiffusionConfig:
    input_dim: int = 2              # (x, y) only
    max_seq_len: int = 200
    latent_dim: int = 512
    num_layers: int = 9
    num_heads: int = 8
    ff_size: int = 1024
    diffusion_steps: int = 1000
    ddim_steps: int = 50
    # No CFG parameters
```

| Preset | latent_dim | layers | ~Params |
|--------|------------|--------|---------|
| smoke_test | 128 | 4 | ~500K |
| medium | 256 | 6 | ~3M |
| full | 512 | 9 | ~20M |

### Trajectory Transformer

```
Input: x_noisy (B, L, 2), t (B,), lengths (B,)

Linear(2 → latent_dim) + pos_encoding
    ↓
Time embedding (sinusoidal → MLP)  ← No goal embedding
    ↓
Transformer blocks × num_layers (attention + FiLM)
    ↓
Linear(latent_dim → 2) → noise_pred
```

### Inpainting Sampler

Replaces CFG-based sampler. At each DDIM step:
1. Predict clean trajectory
2. Replace position[0] = start, position[length-1] = end
3. Re-noise for next step

### Directory Structure

```
diffusion_v8/
├── config_trajectory.py
├── models/
│   ├── trajectory_transformer.py  (time-only conditioning)
│   └── gaussian_diffusion.py
├── datasets/trajectory_dataset.py
├── trainers/trajectory_trainer.py  (no CFG)
├── sampling/inpainting_sampler.py  (replaces CFG sampler)
└── evaluation/metrics.py, visualize.py
```

---

## Quick Reference

```bash
# Data collection
python SapiRecorder.py
python combine_cvss.py
python trajectory_splitter_adaptive.py combined.csv trajectories/

# V8 pipeline
python preprocess_V8.py --input trajectories/ --output processed_data_v8/
python train_diffusion_v8.py --mode medium --epochs 500
python generate_diffusion_v8.py --checkpoint best.pth --target_x 0.5 --target_y 0.3
```

---

## Troubleshooting

**Endpoint not reached exactly**: Check inpainting at `length-1` (not `max_length-1`), applied every step, final step doesn't re-noise.

**Kink at endpoint**: Train longer, or inpaint last 2-3 positions with interpolation.

**Trajectories too straight**: Check if real data is also straight. If not, increase model capacity or train longer.

**Mode collapse**: All trajectories identical—check model capacity and noise addition.

---

## Migration from V7

1. Remove `goal_conditioner.py`
2. Change `input_dim: 8 → 2` in transformer
3. Remove `goal_embed` from forward signature
4. Replace CFG sampler with inpainting sampler
5. Update preprocessing to output 2D positions only

**V7 checkpoints are NOT compatible.** Must retrain.

---

## References

1. **Diffuser**: Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" (ICML 2022) — **inpainting approach**
2. **MDM**: Tevet et al., "Human Motion Diffusion Model" (ICLR 2023) — keyframe inpainting
3. **DDPM/DDIM**: Ho et al. (2020), Song et al. (2021)
4. **CFG**: Ho & Salimans (2021) — useful for semantic, NOT geometric conditions

Code: [Diffuser](https://github.com/jannerm/diffuser), [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse)

---

Reference implementation: `inpainting_reference.py`

Last updated: 2025-12-27
