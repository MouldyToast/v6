# Diffusion V7: Pure Diffusion for Trajectory Generation
## Implementation Plan Based on MotionDiffuse Architecture

**Date**: 2025-12-19
**Status**: Planning Phase
**Reference**: MotionDiffuse (3 years old, adapted for mouse trajectory generation)

---

## Executive Summary

We will adapt the MotionDiffuse architecture to generate mouse trajectories with goal conditioning. The model will learn to generate variable-length trajectories that reach specified endpoints (distance + angle).

**Key Decision**: We will use **3D conditioning** initially: `[distance_norm, cos(angle), sin(angle)]`
Time conditioning can be added later if needed (see Section 4).

---

## 1. Current Data Understanding

### 1.1 Input Data Format (from preprocess_V6.py)

**Trajectory Features** (8 dimensions):
```python
feature_names = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']
```

**Feature Details**:
- `dx`, `dy`: Relative position from trajectory start (position-independent)
- `speed`: Movement speed (px/s)
- `accel`: Acceleration (px/s²)
- `sin_h`, `cos_h`: Heading direction (smooth encoding)
- `ang_vel`: Angular velocity (turning rate)
- `dt`: Time delta between samples

**Shapes**:
- `X`: (n_samples, max_seq_len, 8) - trajectories, padded to 200 timesteps
- `C`: (n_samples, 3) - conditions: [distance_norm, cos(goal_angle), sin(goal_angle)]
- `L`: (n_samples,) - original lengths before padding

**Data Location**: `D:\V6\processed_data_v6\`
- Training set: 1,536 trajectories (test), 15,000+ (production)

### 1.2 Goal Conditioning (from preprocess_V6.py:595-674)

The conditioning is computed as:
```python
# Goal vector: start → end
dx_goal = x[-1] - x[0]
dy_goal = y[-1] - y[0]

# Distance: Euclidean distance from start to end
distance = sqrt(dx_goal² + dy_goal²)  # This is ideal_distance

# Angle: Direction from start to end
angle = arctan2(dy_goal, dx_goal)  # Range: [-π, π]

# 3D condition vector
condition = [
    normalize(distance),  # Normalized to [-1, 1]
    cos(angle),          # X direction component [-1, 1]
    sin(angle)           # Y direction component [-1, 1]
]
```

**This is perfect for endpoint conditioning!** The model learns: "Generate trajectory that ends at this distance and angle from the start."

---

## 2. MotionDiffuse Architecture Analysis

### 2.1 Key Components (from diffusion_v7/)

**File Structure**:
```
diffusion_v7/
├── models/
│   ├── transformer.py          # MotionTransformer (denoiser network)
│   └── gaussian_diffusion.py   # DDPM implementation (from OpenAI)
├── trainers/
│   └── ddpm_trainer.py         # Training loop
├── datasets/
│   ├── dataset.py              # Text2Motion dataset
│   └── dataloader.py           # Data loading utilities
├── tools/
│   ├── train.py                # Training script
│   └── evaluation.py           # Evaluation metrics
└── utils/
    └── get_opt.py              # Config/options parsing
```

### 2.2 MotionDiffuse vs Our Task

| Aspect | MotionDiffuse (Original) | Our Trajectory Diffusion |
|--------|-------------------------|--------------------------|
| **Input** | Human motion sequences (263D joints) | Mouse trajectories (8D features) |
| **Conditioning** | Text (CLIP embeddings, 512D) | Goal (3D: distance + angle) |
| **Sequence Length** | Up to 196 frames | Up to 200 timesteps |
| **Normalization** | Z-normalization (mean/std per feature) | Already normalized to [-1, 1] in V6 |
| **Variable Length** | Yes (with masking) | Yes (with masking) |
| **Diffusion Steps** | 1000 (DDPM) | 1000 (DDPM, 50 DDIM for sampling) |
| **Architecture** | Transformer with temporal self-attention | Same (adapt for 8D) |

### 2.3 Core Diffusion Process (gaussian_diffusion.py)

MotionDiffuse uses standard DDPM:

**Forward Process** (add noise):
```python
# At timestep t, add noise to clean data
q(x_t | x_0) = N(x_t; sqrt(α̅_t) * x_0, (1 - α̅_t) * I)

# In code:
x_noisy = sqrt_alpha_bar[t] * x_clean + sqrt_one_minus_alpha_bar[t] * noise
```

**Reverse Process** (denoise):
```python
# Learn to predict the noise
noise_pred = model(x_noisy, t, condition)
loss = MSE(noise_pred, true_noise)
```

**Sampling** (generation):
```python
# Start from pure noise
x = randn(batch, seq_len, 8)

# Iteratively denoise
for t in reversed(range(1000)):
    noise_pred = model(x, t, condition)
    x = denoise_step(x, t, noise_pred)  # DDPM or DDIM step

return x  # Generated trajectory
```

### 2.4 Transformer Architecture (transformer.py)

**MotionTransformer** components:
1. **Input Embedding**: Projects motion features to latent_dim
2. **Timestep Embedding**: Sinusoidal embedding for diffusion timestep
3. **Condition Embedding**: Embeds text via CLIP (we'll replace with goal MLP)
4. **Temporal Attention Blocks**: Self-attention over sequence + feedforward
5. **Output Projection**: Projects latent back to motion features

**Key Features**:
- `StylizationBlock`: FiLM-style conditioning (scale & shift from timestep/condition)
- `LinearTemporalSelfAttention`: Masked attention for variable-length sequences
- Masking: Pads shorter sequences and masks them during attention

---

## 3. Required Adaptations

### 3.1 Replace Text Conditioning with Goal Conditioning

**Original** (MotionDiffuse):
```python
# Text conditioning via CLIP
text_embedding = clip.encode_text(caption)  # (batch, 512)
```

**Our Adaptation**:
```python
class GoalConditioner(nn.Module):
    """Embed 3D goal condition to match transformer hidden dimension"""
    def __init__(self, condition_dim=3, hidden_dim=512):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim)
        )

    def forward(self, condition):
        # condition: (batch, 3) - [distance_norm, cos(angle), sin(angle)]
        return self.embed(condition)  # (batch, hidden_dim)
```

### 3.2 Adapt Input/Output Dimensions

**Original**: 263D motion features (human skeleton)
**Ours**: 8D trajectory features

**Changes needed**:
```python
# In MotionTransformer.__init__():
# OLD:
self.input_proj = nn.Linear(263, latent_dim)
self.output_proj = nn.Linear(latent_dim, 263)

# NEW:
self.input_proj = nn.Linear(8, latent_dim)  # 8D trajectory features
self.output_proj = nn.Linear(latent_dim, 8)
```

### 3.3 Data Normalization Strategy

**MotionDiffuse approach**: Z-normalization (subtract mean, divide by std)
```python
motion_normalized = (motion - mean) / std
```

**Our current approach** (V6): Min-max normalization to [-1, 1]
```python
feature_normalized = 2 * ((feature - min) / (max - min)) - 1
```

**Decision**: Two options:

**Option A: Keep V6 normalization** (Recommended)
- Pro: Data already preprocessed
- Pro: [-1, 1] range is good for diffusion
- Con: Different from MotionDiffuse (but shouldn't matter)

**Option B: Switch to Z-normalization**
- Pro: Matches MotionDiffuse exactly
- Pro: More standard in ML
- Con: Need to recompute stats and re-normalize data

**Recommendation**: Start with Option A (keep current V6 normalization). The diffusion model doesn't care about the normalization method as long as it's consistent.

### 3.4 Dataset Adapter

Create a new dataset class that wraps V6 preprocessed data:

```python
class TrajectoryDiffusionDataset(torch.utils.data.Dataset):
    """
    Adapter for V6 preprocessed data → MotionDiffuse format
    """
    def __init__(self, data_dir, split='train'):
        # Load V6 preprocessed data
        self.X = np.load(f'{data_dir}/X_{split}.npy')  # (n, 200, 8)
        self.C = np.load(f'{data_dir}/C_{split}.npy')  # (n, 3)
        self.L = np.load(f'{data_dir}/L_{split}.npy')  # (n,)

    def __getitem__(self, idx):
        trajectory = self.X[idx]  # (200, 8)
        condition = self.C[idx]   # (3,)
        length = self.L[idx]      # scalar

        # Return in MotionDiffuse format
        return {
            'motion': torch.FloatTensor(trajectory),
            'condition': torch.FloatTensor(condition),
            'length': length
        }
```

---

## 4. Time Conditioning: Should We Add It?

**Question**: Should we add trajectory duration as a 4th conditioning dimension?

### 4.1 Current Conditioning (3D)
```python
condition = [distance_norm, cos(angle), sin(angle)]
```

This specifies **WHERE** the trajectory should end, but not **HOW LONG** it should take.

### 4.2 With Time Conditioning (4D)
```python
condition = [distance_norm, cos(angle), sin(angle), time_norm]
```

This adds control over trajectory **duration** (total time from start to end).

### 4.3 Analysis

**Pros of adding time**:
- More control: Can generate "fast" vs "slow" trajectories to same endpoint
- Separates movement speed from path shape
- Useful if you want specific timing (e.g., "reach this point in 0.5s")

**Cons of adding time**:
- More complex: 4D conditioning space instead of 3D
- Redundant?: Speed is already encoded in the trajectory features
- Limited data: Your 1,536 trajectories may not cover all (distance, angle, time) combinations

### 4.4 How Time is Currently Encoded

Looking at your features:
- `dt`: Time delta between samples (varies per timestep)
- `speed`: Movement speed (varies per timestep)
- Total time = sum of all `dt` values in the trajectory

The trajectory itself implicitly encodes timing through the `dt` and `speed` features. The diffusion model can learn to generate appropriate timing.

### 4.5 Recommendation

**Start without time conditioning (3D only)**:
1. Simpler to implement and debug
2. The model will learn timing patterns from the `dt` and `speed` features in trajectories
3. Your data naturally varies in timing for the same endpoint

**Add time conditioning later if needed (4D)**:
- If you find generated trajectories have unrealistic timing
- If you want explicit control over duration
- Easy to add: just append normalized time to condition vector

**How to add time later**:
```python
# Compute total time for each trajectory
total_time = sum(dt values)  # or t[-1] - t[0]

# Normalize
time_norm = 2 * ((total_time - min_time) / (max_time - min_time)) - 1

# Extend condition
condition = [distance_norm, cos(angle), sin(angle), time_norm]  # Now 4D

# Update GoalConditioner
self.embed = nn.Sequential(
    nn.Linear(4, 128),  # Changed from 3 to 4
    ...
)
```

**Decision**: Use **3D conditioning** for initial implementation. Monitor generated trajectory timing during evaluation. Add time if needed.

---

## 5. Feature Transformation: Do We Need to Change the 8 Features?

**Your question**: *"The 8 features may not be ideal for diffusion"*

### 5.1 Current Features Analysis

```python
features = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']
```

**Good for diffusion**:
- ✅ Position-independent (dx, dy are relative to start)
- ✅ Smooth, continuous values (good for noise addition/removal)
- ✅ Normalized to [-1, 1]
- ✅ No discontinuities (sin/cos for angles)
- ✅ Physically meaningful (speed, acceleration, heading)

**Potential concerns**:
- ❓ `dx`, `dy` grow larger throughout trajectory (not stationary)
- ❓ Mixed units (position, velocity, acceleration, time)
- ❓ Some features may be redundant or derived

### 5.2 MotionDiffuse's Features

MotionDiffuse uses:
- Root position (x, y, z)
- Root velocity
- Root rotation velocity
- Joint rotations (for skeleton)
- Joint velocities
- Foot contacts

**Key difference**: MotionDiffuse also has position drift (root position), similar to our `dx`, `dy`.

### 5.3 Alternative Feature Representations

**Option A: Keep Current Features** (Recommended)
```python
features = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']
```
- Pro: Already preprocessed and working in V6
- Pro: Physically meaningful
- Pro: Similar to MotionDiffuse (which also has position drift)
- Con: dx, dy grow over time (not stationary)

**Option B: Velocity-Based (Stationary)**
```python
features = ['vx', 'vy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']
```
Where `vx = dx[t] - dx[t-1]`, `vy = dy[t] - dy[t-1]` (velocities instead of positions)
- Pro: Stationary (doesn't drift)
- Pro: Easier for diffusion to model (similar variance throughout sequence)
- Con: Need to reprocess data
- Con: Lose absolute position (need to integrate during generation)

**Option C: Relative Displacements + Cumulative Position**
```python
features = ['dx', 'dy', 'step_dx', 'step_dy', 'sin_h', 'cos_h', 'ang_vel', 'dt']
```
- `dx`, `dy`: Cumulative position (current)
- `step_dx`, `step_dy`: Step-wise displacement (stationary)
- Pro: Best of both worlds
- Con: 8 dimensions becomes less clean

### 5.4 Recommendation

**Keep current features (Option A)** for initial implementation:

**Rationale**:
1. **MotionDiffuse precedent**: They also use position features that drift (root position), and it works fine
2. **Physical meaning**: dx, dy directly relate to the goal (distance, angle)
3. **Goal conditioning helps**: The model knows the target endpoint, which constrains dx, dy growth
4. **Already preprocessed**: V6 data is ready to use
5. **V6 autoencoder works**: If features were problematic, V6 wouldn't reconstruct well (MSE=0.0052)

**Mitigation for non-stationarity**:
The diffusion model learns to denoise in the context of:
- Goal condition (where to end up)
- Sequence length (how many steps available)
- Attention mechanism (can look at future/past timesteps)

This should be sufficient for the model to learn proper dx, dy trajectories.

**Fallback**: If training shows instability or poor results, we can:
1. Try Option B (velocity-based) by reprocessing data
2. Add positional encoding to help model understand "where in sequence"
3. Use curriculum learning (train on short trajectories first)

---

## 6. Implementation Roadmap

### Phase 1: Setup & Architecture (Week 1)

**Goals**:
- Adapt MotionDiffuse architecture for 8D trajectories + 3D goal conditioning
- Create dataset adapter for V6 preprocessed data
- Verify architecture with forward/backward passes

**Tasks**:

**1.1 Create adapted Transformer (2 days)**
```
File: diffusion_v7/models/trajectory_transformer.py
```
- Copy `transformer.py` from MotionDiffuse
- Change input/output dims: 263 → 8
- Replace text conditioning with GoalConditioner
- Keep temporal attention + masking
- Add comments explaining adaptations

**1.2 Create Dataset Adapter (1 day)**
```
File: diffusion_v7/datasets/trajectory_dataset.py
```
- Load V6 .npy files (X, C, L)
- Return format matching MotionDiffuse: (motion, condition, length)
- Handle train/val/test splits
- Verify shapes and data ranges

**1.3 Setup Training Pipeline (2 days)**
```
File: diffusion_v7/trainers/trajectory_trainer.py
```
- Adapt `ddpm_trainer.py` for goal conditioning
- Replace text encoding with GoalConditioner
- Keep diffusion training loop (already good)
- Add logging (TensorBoard)

**1.4 Testing (2 days)**
- Unit test: GoalConditioner forward pass
- Unit test: TrajectoryTransformer forward pass with masking
- Integration test: Full training step (no actual training, just verify no errors)
- Verify gradient flow

**Deliverables**:
- `trajectory_transformer.py` - Adapted transformer
- `trajectory_dataset.py` - Dataset adapter
- `trajectory_trainer.py` - Training loop
- `test_architecture.py` - Unit tests
- Architecture verified, no training yet

---

### Phase 2: Training & Debugging (Week 2)

**Goals**:
- Train diffusion model on 1,536 trajectory dataset
- Monitor training curves and generated samples
- Debug any instabilities or quality issues

**Tasks**:

**2.1 Initial Training (3 days)**
- Train on full dataset (1,536 samples)
- Monitor losses:
  - MSE loss (noise prediction)
  - Per-timestep diffusion loss
  - Validation loss
- Save checkpoints every epoch
- Generate samples every 10 epochs

**2.2 Debugging & Iteration (3 days)**
Common issues to watch for:
- **NaN losses**: Check gradients, learning rate, normalization
- **Mode collapse**: Generated trajectories all look similar
- **Goal mismatch**: Generated trajectories don't reach target
- **Unrealistic motion**: Jerky, unsmooth, or physically impossible

Fixes to try:
- Adjust learning rate (start: 1e-4)
- Gradient clipping (already in MotionDiffuse)
- Increase model capacity (more layers/heads)
- Add guidance scale for conditioning
- Check data preprocessing

**2.3 Qualitative Evaluation (1 day)**
Visualize generated trajectories:
- Plot trajectories vs goal (do they reach target?)
- Check smoothness (acceleration, jerk)
- Compare to real trajectories from test set
- Test different conditions (distance, angle combinations)

**Deliverables**:
- Trained model checkpoint
- Training curves (TensorBoard)
- Sample generated trajectories (visualizations)
- Debugging notes and fixes applied

---

### Phase 3: Sampling & Evaluation (Week 3)

**Goals**:
- Implement DDIM sampling (50 steps instead of 1000)
- Quantitative evaluation metrics
- Compare to V6 GAN baseline
- Generate final test samples

**Tasks**:

**3.1 DDIM Sampling (2 days)**
```
File: diffusion_v7/sampling/ddim_sampler.py
```
- Implement DDIM (deterministic sampling)
- 50 steps instead of 1000 (20x faster)
- Verify quality vs DDPM
- Add generation script

**3.2 Evaluation Metrics (2 days)**
```
File: diffusion_v7/evaluation/metrics.py
```

**Goal Accuracy**:
- Endpoint distance error: |generated_endpoint_dist - target_dist|
- Endpoint angle error: |generated_endpoint_angle - target_angle|

**Realism**:
- Speed distribution (compare to real data)
- Acceleration distribution
- Smoothness (jerk metric)
- Path efficiency (actual_dist / ideal_dist ratio)

**Diversity**:
- For same condition, generate N samples
- Compute pairwise trajectory distances
- Measure variance

**3.3 Comparison to V6 GAN (2 days)**
- Load V6 GAN generator
- Generate samples with same conditions
- Compare metrics: accuracy, realism, diversity
- Create comparison visualizations

**3.4 Documentation (1 day)**
- Write evaluation report
- Document hyperparameters
- Save final model and generation script
- Create usage examples

**Deliverables**:
- DDIM sampler (fast generation)
- Evaluation metrics computed
- Comparison report (Diffusion V7 vs V6 GAN)
- Final trained model + generation script
- Usage documentation

---

## 7. Architecture Details

### 7.1 Model Configuration

Based on MotionDiffuse defaults, adapted for trajectories:

```python
model_config = {
    # Input/Output
    'input_dim': 8,              # trajectory features
    'latent_dim': 512,           # transformer hidden dimension
    'condition_dim': 3,          # goal conditioning (distance + angle)

    # Transformer architecture
    'num_layers': 9,             # depth
    'num_heads': 8,              # attention heads
    'ff_size': 1024,             # feedforward layer size
    'dropout': 0.1,
    'activation': 'gelu',

    # Sequence
    'max_seq_len': 200,          # maximum trajectory length

    # Diffusion
    'diffusion_steps': 1000,     # training timesteps
    'noise_schedule': 'linear',  # beta schedule
    'beta_start': 1e-4,
    'beta_end': 0.02,

    # Training
    'batch_size': 64,            # adjust based on GPU memory
    'learning_rate': 1e-4,
    'weight_decay': 0.0,
    'grad_clip': 0.5,
    'epochs': 1000,              # may need less with small dataset

    # Sampling
    'ddim_steps': 50,            # fast sampling
    'ddim_eta': 0.0,             # deterministic
}
```

### 7.2 Network Architecture

```
TrajectoryDiffusionModel
│
├── Input Embedding
│   └── Linear(8 → 512)
│
├── Timestep Embedding
│   └── SinusoidalEmbedding(→ 512)
│
├── Condition Embedding
│   └── GoalConditioner(3 → 512)
│
├── Transformer Blocks (x9)
│   ├── Temporal Self-Attention (masked)
│   │   ├── LayerNorm
│   │   ├── Multi-Head Attention (8 heads)
│   │   └── StylizationBlock (timestep + condition)
│   │
│   └── FeedForward
│       ├── LayerNorm
│       ├── Linear(512 → 1024) + GELU
│       ├── Linear(1024 → 512)
│       └── StylizationBlock
│
└── Output Projection
    └── Linear(512 → 8)
```

**Key mechanisms**:
- **Masking**: Attention mask zeros out padding positions
- **FiLM conditioning**: Timestep & goal modulate via scale/shift
- **Residual connections**: Skip connections around each block

---

## 8. Data Pipeline

### 8.1 Data Flow

```
Raw trajectories (JSON)
    ↓
[preprocess_V6.py] - Extract features, compute goal conditions
    ↓
Processed data (.npy files)
├── X_train.npy: (n_train, 200, 8)    - Trajectories
├── C_train.npy: (n_train, 3)         - Goal conditions
├── L_train.npy: (n_train,)           - Lengths
└── normalization_params.npy          - Min/max for denormalization
    ↓
[TrajectoryDiffusionDataset] - PyTorch Dataset
    ↓
DataLoader (batch_size=64, shuffle=True)
    ↓
Training Loop
```

### 8.2 Batch Format

```python
batch = {
    'motion': Tensor(64, 200, 8),     # Padded trajectories
    'condition': Tensor(64, 3),       # Goal conditions
    'length': Tensor(64,)             # Original lengths
}

# During training:
# 1. Sample diffusion timestep t ~ Uniform(0, 999)
# 2. Add noise: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
# 3. Predict noise: ε_pred = model(x_t, t, condition, length)
# 4. Compute loss: MSE(ε_pred, ε) with masking
# 5. Backprop
```

---

## 9. Open Questions & Decisions Needed

### ✅ Resolved
1. **Conditioning format**: 3D [distance, cos(angle), sin(angle)] ✓
2. **Feature representation**: Keep current 8 features ✓
3. **Normalization**: Keep V6 [-1, 1] normalization ✓

### ❓ To Decide

1. **Time conditioning**: Start without, add later if needed?
   - **User response needed**: Do you want explicit time control?

2. **Model capacity**:
   - MotionDiffuse uses 9 layers, 512 dim for 263D motion
   - Our data is 8D (much simpler)
   - Should we use smaller model for faster training?
   - Recommendation: Start with full size, can reduce if needed

3. **Batch size**:
   - MotionDiffuse uses 64
   - With 1,536 samples, 64 = 24 batches/epoch
   - Recommendation: Use 64, good for stable gradients

4. **Augmentation**:
   - Should we augment training data?
   - Possible: Rotate trajectories, flip, add small noise
   - Recommendation: No augmentation initially (data is already position-independent)

5. **Classifier-free guidance**:
   - MotionDiffuse uses this for better conditioning
   - Trains model to work with/without condition
   - During sampling, combines conditional + unconditional predictions
   - Recommendation: Implement in Phase 3 if needed

---

## 10. Success Criteria

### Minimum Viable Model (End of Week 2)
- ✅ Training converges (loss decreases)
- ✅ Generates trajectories that reach target endpoint (within 10% error)
- ✅ Trajectories are smooth (no NaN, no extreme jitter)
- ✅ Model works with variable lengths

### Target Performance (End of Week 3)
- ✅ Goal distance error < 5% of target
- ✅ Goal angle error < 10 degrees
- ✅ Smoothness comparable to real data
- ✅ Diversity: 10 samples for same condition are visibly different
- ✅ Generation speed: <1s for batch of 64 trajectories (DDIM)

### Comparison to V6 GAN
- ✅ Better or equal goal accuracy
- ✅ Better diversity (diffusion should excel here)
- ✅ Comparable realism metrics

---

## 11. Risk Mitigation

### Risk 1: Small Dataset (1,536 samples)
**Mitigation**:
- Diffusion models can work with small datasets (better than GANs)
- Monitor for overfitting (validation loss)
- Use weight decay, dropout
- Plan to scale to 15,000 samples

### Risk 2: Non-Stationary Features (dx, dy drift)
**Mitigation**:
- Goal conditioning should constrain drift
- MotionDiffuse handles similar drift
- Fallback: Switch to velocity-based features

### Risk 3: Training Instability
**Mitigation**:
- Gradient clipping (already in MotionDiffuse)
- Learning rate scheduling
- Batch normalization in goal conditioner
- Monitor for NaN/exploding gradients

### Risk 4: Poor Conditioning Control
**Mitigation**:
- Verify goal embedding is learned (check attention weights)
- Try classifier-free guidance
- Increase conditioning signal strength

---

## 12. Next Steps

**Immediate** (Today):
1. ✅ Review this plan with user
2. ❓ Get confirmation on decisions (time conditioning, model size)
3. ❓ Verify data location and accessibility

**Week 1 Start** (After Approval):
1. Create `diffusion_v7/models/trajectory_transformer.py`
2. Create `diffusion_v7/datasets/trajectory_dataset.py`
3. Create `diffusion_v7/trainers/trajectory_trainer.py`
4. Write unit tests

**Questions for User**:
1. Should we add time as 4th conditioning dimension? (Recommendation: No, add later if needed)
2. Should we use smaller model (fewer layers/dims) given 8D input? (Recommendation: Start full size)
3. Is data at `D:\V6\processed_data_v6\` accessible in training environment?
4. Should we implement data augmentation? (Recommendation: No initially)

---

## 13. File Structure (After Implementation)

```
v6/
├── diffusion_v7/                      # MotionDiffuse reference code
│   ├── models/
│   │   ├── gaussian_diffusion.py     # (Keep from MotionDiffuse)
│   │   ├── transformer.py            # (Reference)
│   │   └── trajectory_transformer.py # NEW: Our adapted model
│   ├── datasets/
│   │   ├── dataset.py                # (Reference)
│   │   └── trajectory_dataset.py     # NEW: V6 data adapter
│   ├── trainers/
│   │   ├── ddpm_trainer.py           # (Reference)
│   │   └── trajectory_trainer.py     # NEW: Our training loop
│   ├── sampling/
│   │   └── ddim_sampler.py           # NEW: Fast sampling
│   ├── evaluation/
│   │   └── metrics.py                # NEW: Evaluation
│   └── utils/
│       └── conditioning.py           # NEW: GoalConditioner
│
├── run_diffusion_v7.py               # NEW: Training script
├── generate_diffusion_v7.py          # NEW: Generation script
├── evaluate_diffusion_v7.py          # NEW: Evaluation script
├── test_diffusion_v7.py              # NEW: Unit tests
│
└── preprocess_V6.py                  # (Existing - no changes needed)
```

---

## 14. Conclusion

We have a clear path to adapt MotionDiffuse for trajectory generation:
1. Replace text conditioning with 3D goal conditioning
2. Adapt input/output dims (263D → 8D)
3. Use existing V6 preprocessed data (no reprocessing needed)
4. Keep transformer architecture (proven to work)
5. 3-week implementation: Setup (Week 1) → Training (Week 2) → Evaluation (Week 3)

The approach is conservative (minimal changes from MotionDiffuse) while being appropriate for our task (goal conditioning, 8D trajectories).

**Ready to proceed upon user approval and answers to open questions.**
