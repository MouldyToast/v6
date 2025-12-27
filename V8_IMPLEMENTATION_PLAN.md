# V8 Implementation Plan

## Overview

This document details the migration from V7 (CFG-based) to V8 (inpainting-based) trajectory diffusion.

**Core Change**: Replace Classifier-Free Guidance with endpoint inpainting for direction control.

---

## Phase 1: Directory Structure

Create the V8 package structure:

```
diffusion_v8/
├── __init__.py
├── config_trajectory.py          # Modified from V7
├── models/
│   ├── __init__.py
│   ├── trajectory_transformer.py  # Modified (2D input, no goal_embed)
│   └── gaussian_diffusion.py      # Copied unchanged from V7
├── datasets/
│   ├── __init__.py
│   └── trajectory_dataset.py      # Modified (2D positions, no conditions)
├── trainers/
│   ├── __init__.py
│   └── trajectory_trainer.py      # Modified (no CFG, no GoalConditioner)
├── sampling/
│   ├── __init__.py
│   └── inpainting_sampler.py      # NEW (from inpainting_reference.py)
└── evaluation/
    ├── __init__.py
    ├── metrics.py                  # Copied from V7
    └── visualize.py                # Copied from V7
```

**Note**: No `goal_conditioner.py` in V8.

---

## Phase 2: File-by-File Changes

### 2.1 config_trajectory.py

**Source**: `diffusion_v7/config_trajectory.py` (333 lines)

| Line | Current V7 | V8 Change |
|------|------------|-----------|
| 19 | `input_dim: int = 8` | → `input_dim: int = 2` |
| 20 | `condition_dim: int = 3` | **DELETE** |
| 36-38 | CFG block | **DELETE** |
| 109 | `assert self.input_dim == 8` | → `assert self.input_dim == 2` |
| 110 | `assert self.condition_dim == 3` | **DELETE** |
| 114-115 | CFG assertions | **DELETE** |
| 236-239 | Print CFG section | **DELETE** |
| 287 | `condition_embed = ...` | **DELETE** |

**New parameters to add:**
```python
# Inpainting Configuration
inpaint_start: bool = True      # Fix start position
inpaint_end: bool = True        # Fix end position
```

---

### 2.2 trajectory_transformer.py

**Source**: `diffusion_v7/models/trajectory_transformer.py` (504 lines)

| Line | Current V7 | V8 Change |
|------|------------|-----------|
| 305 | `input_dim: int = 8` | → `input_dim: int = 2` |
| 313 | `goal_latent_dim: int = 512` | **DELETE parameter** |
| 326 | `self.cond_embed_dim = self.time_embed_dim + goal_latent_dim` | → `self.cond_embed_dim = self.time_embed_dim` |
| 357-363 | `forward(self, x, t, goal_embed, lengths)` | → `forward(self, x, t, lengths)` |
| 390 | `cond_emb = torch.cat([t_emb, goal_embed], dim=-1)` | → `cond_emb = t_emb` |
| 355 | `self.output_proj = ... Linear(latent_dim, input_dim)` | Will auto-adjust to 2 |

**Unchanged components:**
- `timestep_embedding()` function
- `StylizationBlock` class
- `TemporalSelfAttention` class
- `FeedForward` class
- `TransformerBlock` class
- `generate_src_mask()` method

---

### 2.3 trajectory_trainer.py

**Source**: `diffusion_v7/trainers/trajectory_trainer.py` (578 lines)

| Line | Current V7 | V8 Change |
|------|------------|-----------|
| 22 | `from ..models.goal_conditioner import GoalConditioner` | **DELETE** |
| 50-55 | `self.goal_conditioner = GoalConditioner(...)` | **DELETE** |
| 86 | `all_params = ... + list(self.goal_conditioner.parameters())` | → `all_params = list(self.model.parameters())` |
| 167-169 | Print conditioner params | **DELETE** |
| 187-191 | Print CFG section | **DELETE** |
| 215 | `self.goal_conditioner.train()` | **DELETE** |
| 218-220 | `condition = batch['condition'].to(self.device)` | **DELETE** |
| 227-234 | CFG mask computation | **DELETE** |
| 234 | `goal_embed = self.goal_conditioner(...)` | **DELETE** |
| 239-241 | `model_fn` with goal_embed | → `return self.model(x, timesteps, lengths)` |
| 266 | Grad clip includes goal_conditioner | → Only model params |
| 410 | `self.goal_conditioner.train()` | **DELETE** |
| 446 | `self.goal_conditioner.eval()` | **DELETE** |
| 457-458 | Load condition from batch | **DELETE** |
| 465 | `goal_embed = self.goal_conditioner(...)` | **DELETE** |
| 468-469 | `model_fn` with goal_embed | → `return self.model(x, timesteps, lengths)` |
| 521 | `'goal_conditioner_state_dict': ...` | **DELETE** |
| 546 | `self.goal_conditioner.load_state_dict(...)` | **DELETE** |

---

### 2.4 trajectory_dataset.py

**Source**: `diffusion_v7/datasets/trajectory_dataset.py` (325 lines)

| Line | Current V7 | V8 Change |
|------|------------|-----------|
| 7-9 | Comment: 8D features, 3D conditions | Update to 2D |
| 53 | `self.X, self.C, self.L = self._load_data()` | → `self.X, self.L = ...` |
| 58 | Print C shape | **DELETE** |
| 65 | `C_path = ...` | **DELETE** |
| 69-71 | Check C_path exists | **DELETE** |
| 75 | `C = np.load(C_path)` | **DELETE** |
| 81 | `assert X.shape[2] == 8` | → `assert X.shape[2] == 2` |
| 82-83 | Assert C shape | **DELETE** |
| 85 | `assert len(X) == len(C) == len(L)` | → `assert len(X) == len(L)` |
| 92 | `return X, C, L` | → `return X, L` |
| 107-110 | Return dict with 'condition' | Remove 'condition' key |

**V8 __getitem__ return:**
```python
return {
    'motion': torch.FloatTensor(self.X[idx]),     # (max_seq_len, 2)
    'length': torch.LongTensor([self.L[idx]])[0]  # scalar
}
```

---

### 2.5 inpainting_sampler.py (NEW)

**Source**: `inpainting_reference.py` lines 28-196

This file already exists and can be copied with minor adjustments.

Key class: `InpaintingSampler`
- `generate(start_pos, end_pos, length, ddim_steps, eta)`
- `generate_batch(endpoints, lengths, ddim_steps)`
- `_get_ddim_timesteps(num_steps)`
- `_predict_x0(x_t, t, noise_pred)`
- `_add_noise(x_0, t, eta)`

---

### 2.6 gaussian_diffusion.py (COPY UNCHANGED)

**Source**: `diffusion_v7/models/gaussian_diffusion.py`

Copy exactly as-is. This file is from MotionDiffuse/OpenAI and handles:
- Beta schedules (linear, cosine, sqrt)
- Forward diffusion (q_sample)
- Training losses (training_losses)
- DDPM sampling (p_sample_loop)

---

### 2.7 preprocess_V8.py (NEW)

**Based on**: `inpainting_reference.py` lines 274-324 and `preprocess_V6.py` structure

**Key changes from V6 preprocessing:**
1. Output 2D positions `(x, y)` instead of 8D features
2. No condition array `C_*.npy`
3. Store endpoints separately for evaluation

**Output files:**
```
processed_data_v8/
├── X_train.npy              # (N, 200, 2) - positions
├── X_val.npy
├── X_test.npy
├── L_train.npy              # (N,) - lengths
├── L_val.npy
├── L_test.npy
├── endpoints_train.npy      # (N, 2) - for evaluation only
├── endpoints_val.npy
├── endpoints_test.npy
└── normalization_params.npy # {position_scale: float}
```

---

## Phase 3: CLI Scripts

### 3.1 train_diffusion_v8.py

```python
# Usage:
# python train_diffusion_v8.py --mode smoke_test
# python train_diffusion_v8.py --mode medium --epochs 500
# python train_diffusion_v8.py --mode full --epochs 1000
```

### 3.2 generate_diffusion_v8.py

```python
# Usage:
# python generate_diffusion_v8.py --checkpoint best.pth --target_x 0.5 --target_y 0.3 --length 100
# python generate_diffusion_v8.py --checkpoint best.pth --use_test_endpoints --num_samples 100
```

---

## Phase 4: Testing

### 4.1 Smoke Test Criteria

- Loss should decrease from ~0.3 to ~0.02 in 1000 steps
- Architecture validation: can overfit single batch

### 4.2 Inpainting Validation

- Endpoint error should be ~0 (numerical precision)
- Generated trajectories should be smooth
- Diverse outputs for same endpoint (with eta > 0)

---

## Implementation Order

1. **Create directory structure** - `diffusion_v8/` with all subdirs
2. **Copy gaussian_diffusion.py** - unchanged
3. **Create config_trajectory.py** - remove CFG, input_dim 8→2
4. **Create trajectory_transformer.py** - remove goal_embed
5. **Create trajectory_dataset.py** - 2D, no conditions
6. **Create trajectory_trainer.py** - remove CFG/GoalConditioner
7. **Create inpainting_sampler.py** - from reference
8. **Create __init__.py files** - package structure
9. **Create preprocess_V8.py** - new preprocessing
10. **Create train_diffusion_v8.py** - CLI
11. **Create generate_diffusion_v8.py** - CLI
12. **Run smoke test** - validate architecture

---

## Risk Mitigation

1. **Keep V7 intact** - Don't modify existing V7 files
2. **Test incrementally** - Validate each component before integration
3. **Use reference implementation** - `inpainting_reference.py` is already tested
4. **Document differences** - This plan serves as migration guide

---

## Checkpoint Incompatibility

V7 checkpoints **cannot** be loaded into V8:
- Different input dimensions (8 vs 2)
- Different conditioning (goal_embed vs none)
- Different cond_embed_dim (1024 vs 512)

**Must retrain from scratch.**
