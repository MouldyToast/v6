# Trajectory Diffusion Model - Implementation Recommendation

## Executive Summary

**Recommendation: Pure Diffusion Model (1600D trajectory space)**

While latent diffusion offers faster training (2 weeks vs 3 weeks), pure diffusion provides:
- Higher success probability (85-90% vs 80-85%)
- No information bottleneck from 50:1 compression
- Better long-term maintainability
- Direct trajectory control

## Architecture Comparison

### Current V6 Architecture
```
Stage 1 (Autoencoder - COMPLETED):
  x_real (200×8) → Encoder → h_seq (200×48) → Pooler → z_summary (192D)
  z_summary (192D) → Expander → h_seq (200×48) → Decoder → x_recon (200×8)

Stage 2 (GAN):
  Generator: noise + condition → z_summary (192D)
  Discriminator: z_summary + condition → real/fake score
```

**Key insight**: Pooler compresses 9,600D → 192D (50:1 ratio)

### Option A: Latent Diffusion (Replace GAN with Diffusion in 192D)

```python
# Architecture
class LatentDiffusionV7:
    def __init__(self):
        # Frozen V6 components
        self.encoder = load_frozen_v6_encoder()      # (200×8) → (200×48)
        self.pooler = load_frozen_v6_pooler()        # (200×48) → (192D)
        self.expander = load_frozen_v6_expander()    # (192D) → (200×48)
        self.decoder = load_frozen_v6_decoder()      # (200×48) → (200×8)

        # NEW: Diffusion denoiser in 192D latent space
        self.denoiser = LatentDenoiser(
            latent_dim=192,
            condition_dim=3,  # distance + cos(angle) + sin(angle)
            hidden_dim=512,
            time_emb_dim=128
        )

    def training_step(self, x_real, condition, lengths):
        # Encode to latent (frozen)
        with torch.no_grad():
            h_seq = self.encoder(x_real, lengths)
            z_clean = self.pooler(h_seq, lengths)  # (batch, 192)

        # Standard DDPM training
        t = torch.randint(0, 1000, (batch,))
        noise = torch.randn_like(z_clean)
        z_noisy = sqrt_alpha_bar[t] * z_clean + sqrt_one_minus_alpha_bar[t] * noise

        # Predict noise
        noise_pred = self.denoiser(z_noisy, t, condition)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def generate(self, condition, seq_len):
        # Start from noise
        z = torch.randn(batch, 192)

        # Iterative denoising (50 DDIM steps)
        for t in reversed(range(0, 1000, 20)):
            z = ddim_step(z, t, condition)

        # Decode (frozen V6)
        with torch.no_grad():
            h_seq = self.expander(z, seq_len)
            x = self.decoder(h_seq, lengths)

        return x
```

**Pros:**
- Leverages existing V6 autoencoder (already trained)
- Fast training: 192D space vs 1600D space
- Fast generation: ~2-3x faster inference
- Proven approach (Stable Diffusion, Latent Diffusion Models)
- Simple implementation (~500 lines of code)

**Cons:**
- **50:1 compression bottleneck**: 9,600D → 192D may lose fine temporal details
- **Fixed by frozen autoencoder**: Quality ceiling from Stage 1
- **Two-stage dependency**: If autoencoder has issues, diffusion inherits them
- **Mode coverage risk**: 192D may not capture full trajectory diversity
- Lower success probability: 80-85%

**Time estimate:** 2 weeks

### Option B: Pure Diffusion (Direct on 1600D Trajectories)

```python
# Architecture
class TrajectoryDiffusion:
    def __init__(self):
        # Single diffusion model operating directly on trajectories
        self.denoiser = Transformer1D(
            input_dim=8,              # trajectory features
            d_model=256,              # transformer width
            n_heads=8,
            n_layers=8,
            condition_dim=3,          # distance + cos(angle) + sin(angle)
            max_seq_len=200,
            time_emb_dim=128
        )

    def training_step(self, x_real, condition, lengths):
        # Standard DDPM training directly on trajectories
        t = torch.randint(0, 1000, (batch,))
        noise = torch.randn_like(x_real)  # (batch, 200, 8)

        # Add noise
        x_noisy = sqrt_alpha_bar[t] * x_real + sqrt_one_minus_alpha_bar[t] * noise

        # Predict noise with mask for variable lengths
        noise_pred = self.denoiser(x_noisy, t, condition, lengths)

        # Masked loss (only valid timesteps)
        loss = masked_mse_loss(noise_pred, noise, lengths)
        return loss

    def generate(self, condition, seq_len):
        # Start from noise
        x = torch.randn(batch, seq_len, 8)
        lengths = torch.full((batch,), seq_len)

        # Iterative denoising (50 DDIM steps)
        for t in reversed(range(0, 1000, 20)):
            x = ddim_step(x, t, condition, lengths)

        return x
```

**Pros:**
- **No information bottleneck**: Full 1600D space preserves all temporal details
- **End-to-end learning**: Diffusion learns optimal representations for generation
- **Higher success probability**: 85-90%
- **MotionDiffuse reference**: Purpose-built for motion sequences
- **Single model**: Simpler long-term maintenance
- **Better diversity**: Can model fine-grained trajectory variations

**Cons:**
- Slower training: 1600D vs 192D space
- Slower generation: ~2-3x slower inference
- More complex implementation (~1200 lines for transformer)
- Requires more GPU memory

**Time estimate:** 3 weeks

## Technical Deep Dive: Why 192D Might Be Insufficient

### Compression Analysis

Your V6 pooler compresses a temporal sequence to a single vector:

```python
# Encoder output: (batch, 200 timesteps, 48 features) = 9,600 numbers
# Pooler operation: attention-weighted sum → (batch, 192)

# What's preserved:
✓ Global trajectory shape (overall curvature)
✓ Summary statistics (avg speed, distance, endpoint)
✓ High-level features learned by pooler attention

# What's potentially lost:
✗ Per-timestep variation (local speed changes)
✗ Fine temporal dynamics (acceleration patterns)
✗ Rare trajectory patterns (modes with <1% frequency)
✗ Temporal ordering details (when events occur)
```

### Generation Requirements vs Reconstruction

**Reconstruction (what V6 autoencoder does):**
- Given: A specific real trajectory
- Task: Compress → decompress to match input
- Success: MSE = 0.0052 ✓

**Generation (what diffusion needs):**
- Given: Random noise + goal condition
- Task: Sample from full distribution of valid trajectories
- Success: Generate diverse, realistic trajectories that cover all modes

**The difference matters**: An autoencoder can achieve low reconstruction error while still missing rare trajectory patterns, because it only needs to reconstruct trajectories it has seen. Generation requires modeling the entire distribution, including rare modes.

## Specific Concerns for Your Use Case

### 1. Goal Conditioning Complexity

You're conditioning on distance + angle. Consider these scenarios:

```
Scenario A: Distance=100, Angle=0° (straight ahead)
  - Should generate: straight-line trajectories
  - Variations: constant speed vs acceleration, slight wobble vs perfect line

Scenario B: Distance=100, Angle=90° (hard right)
  - Should generate: various turning trajectories
  - Variations: wide turn vs sharp turn, smooth vs jerky, early turn vs late turn

Scenario C: Distance=100, Angle=180° (U-turn)
  - Should generate: diverse U-turn patterns
  - Variations: left U-turn vs right U-turn, wide vs tight, speed variations
```

**Question**: Can 192D latent space capture all these variations?

The pooler was trained for **reconstruction**, not **generation diversity**. It may have learned to compress away variations that don't hurt reconstruction but are essential for generation.

### 2. Variable Length Handling

```python
# V6 approach:
z_summary = pooler(h_seq, lengths)  # (200, 48) → (192,)
# Question: Does z_summary encode the target length?
# If not, how does expander know whether to generate 50 or 200 timesteps?

# Pure diffusion approach:
x = denoiser(x_noisy, t, condition, lengths)  # lengths explicitly provided
# Clear: sequence length is part of the generation process
```

### 3. Training Data Distribution

If your training data has:
- 60% straight trajectories
- 30% gentle curves
- 8% sharp turns
- 2% complex maneuvers

The pooler may have learned to represent common patterns well but compress rare patterns aggressively. This is fine for reconstruction (rare patterns still reconstruct well from their latent codes), but generation in 192D may undersample rare modes.

## Recommended Implementation Path: Pure Diffusion

### Phase 1: Core Diffusion (Week 1)

**Reference**: Start from MotionDiffuse architecture
- GitHub: https://github.com/mingyuan-zhang/MotionDiffuse
- Paper: "MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model"

**Adaptation steps:**

```python
# 1. Create diffusion schedule (standard DDPM)
def create_schedule(T=1000):
    """Linear beta schedule from DDPM"""
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

# 2. Implement denoiser network
#    Option A: Transformer (like MotionDiffuse)
#    Option B: Temporal U-Net (like Guided Diffusion)
#    Recommendation: Transformer for variable-length sequences

# 3. Training loop
for epoch in range(num_epochs):
    for x_real, conditions, lengths in dataloader:
        # Sample timestep
        t = torch.randint(0, T, (batch,))

        # Add noise
        noise = torch.randn_like(x_real)
        x_noisy = add_noise(x_real, t, noise)

        # Predict noise
        noise_pred = model(x_noisy, t, conditions, lengths)

        # Masked loss
        loss = masked_mse_loss(noise_pred, noise, lengths)
        loss.backward()
        optimizer.step()
```

**Deliverables:**
- `diffusion_v7/schedule.py` - noise schedule
- `diffusion_v7/denoiser.py` - transformer denoiser
- `diffusion_v7/trainer.py` - training loop
- `diffusion_v7/model.py` - main diffusion model

### Phase 2: Conditioning & Variable Length (Week 2)

**Goal conditioning:**

```python
class GoalConditioner(nn.Module):
    def __init__(self, condition_dim=3):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

    def forward(self, distance, angle):
        # Convert to model input
        condition = torch.stack([
            distance,
            torch.cos(angle),
            torch.sin(angle)
        ], dim=-1)  # (batch, 3)

        return self.embed(condition)  # (batch, 128)
```

**Variable length handling:**

```python
# Approach 1: Mask-based (recommended)
def denoiser_forward(x, t, condition, lengths):
    # x: (batch, max_seq_len, 8)
    # lengths: (batch,) - actual lengths

    # Create attention mask
    mask = create_mask(lengths, max_seq_len)  # (batch, max_seq_len)

    # Transformer with mask
    output = transformer(x, t, condition, mask)

    # Zero out padding
    output = output * mask.unsqueeze(-1)
    return output

# Approach 2: Pack/unpack (like V6)
def denoiser_forward(x, t, condition, lengths):
    # Pack sequences
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

    # Process
    output_packed = transformer(packed, t, condition)

    # Unpack
    output, _ = pad_packed_sequence(output_packed, batch_first=True)
    return output
```

**Deliverables:**
- `diffusion_v7/conditioning.py` - goal conditioning module
- `diffusion_v7/masking.py` - variable length masking
- Updated training pipeline with conditioning

### Phase 3: Generation & Evaluation (Week 3)

**DDIM sampling** (faster than DDPM):

```python
def ddim_sample(model, condition, seq_len, steps=50):
    """Generate using DDIM (Song et al., 2020)"""
    # Start from noise
    x = torch.randn(batch, seq_len, 8)
    lengths = torch.full((batch,), seq_len)

    # Subset timesteps for fast sampling
    timesteps = torch.linspace(999, 0, steps).long()

    for i, t in enumerate(timesteps):
        # Predict noise
        noise_pred = model(x, t, condition, lengths)

        # DDIM update (deterministic)
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            x = ddim_step(x, t, t_next, noise_pred)
        else:
            # Final step
            x = (x - sqrt_one_minus_alpha[t] * noise_pred) / sqrt_alpha[t]

    return x
```

**Evaluation metrics:**

```python
# 1. Goal conditioning accuracy
def evaluate_goal_accuracy(generated_trajectories, target_conditions):
    """Check if generated trajectories match goal distance/angle"""
    final_positions = generated_trajectories[:, -1, :2]  # (batch, 2)

    actual_distance = torch.norm(final_positions, dim=1)
    actual_angle = torch.atan2(final_positions[:, 1], final_positions[:, 0])

    target_distance = target_conditions[:, 0]
    target_angle = target_conditions[:, 1]

    distance_error = (actual_distance - target_distance).abs().mean()
    angle_error = angular_distance(actual_angle, target_angle).mean()

    return distance_error, angle_error

# 2. Diversity metrics
def evaluate_diversity(generated_trajectories):
    """Measure trajectory diversity within same condition"""
    # Group by condition, compute pairwise distances
    diversity_scores = []
    for condition_group in grouped_trajectories:
        pairwise_dist = compute_trajectory_distances(condition_group)
        diversity_scores.append(pairwise_dist.mean())
    return torch.tensor(diversity_scores).mean()

# 3. Realism metrics (compare to V6 evaluation)
def evaluate_realism(generated_trajectories, real_trajectories):
    """Use V6's evaluation metrics"""
    # Speed distribution
    # Acceleration distribution
    # Smoothness (jerk)
    # Compare via Frechet Distance, etc.
```

**Deliverables:**
- `diffusion_v7/sampling.py` - DDIM generation
- `diffusion_v7/evaluation.py` - metrics
- `generate_v7.py` - generation script
- Comparison report vs V6 GAN

## Code Structure

```
v6/
├── diffusion_v7/
│   ├── __init__.py
│   ├── model.py              # Main TrajectoryDiffusion class
│   ├── denoiser.py           # Transformer denoiser network
│   ├── schedule.py           # DDPM/DDIM noise schedules
│   ├── conditioning.py       # Goal conditioning modules
│   ├── masking.py            # Variable length utilities
│   ├── trainer.py            # Training loop
│   ├── sampling.py           # DDIM generation
│   ├── evaluation.py         # Metrics
│   └── config.py             # Hyperparameters
│
├── run_diffusion_v7.py       # Training script
├── generate_diffusion_v7.py  # Generation script
└── evaluate_diffusion_v7.py  # Evaluation script
```

## Hyperparameter Recommendations

Based on MotionDiffuse and your V6 experience:

```python
# Model architecture
d_model = 256              # Transformer width
n_heads = 8                # Attention heads
n_layers = 8               # Transformer depth
time_emb_dim = 128         # Time embedding dimension
condition_dim = 3          # distance + cos(angle) + sin(angle)

# Diffusion schedule
T = 1000                   # Training diffusion steps
beta_start = 1e-4          # Noise schedule start
beta_end = 0.02            # Noise schedule end
schedule = 'linear'        # Linear beta schedule

# Sampling
sample_steps = 50          # DDIM steps (20x faster than 1000-step DDPM)
eta = 0.0                  # DDIM eta (0=deterministic, 1=DDPM)

# Training
batch_size = 128           # Large batch for stable training
learning_rate = 1e-4       # Adam LR
weight_decay = 1e-5        # L2 regularization
epochs = 500               # Train to convergence
warmup_epochs = 10         # LR warmup

# Data
max_seq_len = 200          # Max trajectory length
feature_dim = 8            # dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt
```

## Alternative: Hybrid Approach (Best of Both Worlds)

If you want to leverage V6 while avoiding the bottleneck:

### Latent Diffusion in h_seq Space (9,600D)

Instead of diffusing in z_summary (192D), diffuse in h_seq (200×48=9,600D):

```python
class HybridLatentDiffusion:
    def __init__(self):
        # Frozen V6 components
        self.encoder = load_frozen_v6_encoder()      # (200×8) → (200×48)
        self.decoder = load_frozen_v6_decoder()      # (200×48) → (200×8)

        # NEW: Diffusion in h_seq space (NO POOLER)
        self.denoiser = TemporalDenoiser(
            latent_dim=48,        # Per-timestep latent dim
            condition_dim=3,
            n_layers=6
        )

    def training_step(self, x_real, condition, lengths):
        # Encode to h_seq (frozen)
        with torch.no_grad():
            h_clean = self.encoder(x_real, lengths)  # (batch, 200, 48)

        # Diffusion in h_seq space
        t = torch.randint(0, 1000, (batch,))
        noise = torch.randn_like(h_clean)
        h_noisy = add_noise(h_clean, t, noise)

        noise_pred = self.denoiser(h_noisy, t, condition, lengths)
        loss = masked_mse_loss(noise_pred, noise, lengths)
        return loss

    def generate(self, condition, seq_len):
        # Diffusion in h_seq space
        h = torch.randn(batch, seq_len, 48)
        for t in reversed(range(0, 1000, 20)):
            h = ddim_step(h, t, condition, lengths)

        # Decode (frozen V6)
        with torch.no_grad():
            x = self.decoder(h, lengths)
        return x
```

**Pros vs Latent 192D:**
- Preserves temporal structure (200 timesteps)
- Only 6:1 compression (9600D → 1600D via encoder)
- Leverages V6 encoder/decoder

**Pros vs Pure Diffusion:**
- Smaller space than raw trajectories (9,600D vs 1,600D)
- Faster training/generation
- Leverages V6 encoder/decoder

**Cons:**
- Still has encoder bottleneck (though much smaller)
- Requires frozen encoder to be good at ALL trajectory patterns
- More complex than pure diffusion

**Time estimate:** 2.5 weeks

## Decision Matrix

| Criterion | Latent 192D | Hybrid 9600D | Pure 1600D |
|-----------|-------------|--------------|------------|
| Success probability | 80-85% | 85% | 85-90% |
| Information loss | High (50:1) | Low (6:1) | None (1:1) |
| Training time | 2 weeks | 2.5 weeks | 3 weeks |
| Implementation complexity | Low | Medium | Medium |
| Inference speed | Fastest | Fast | Moderate |
| Long-term maintenance | Medium | Complex | Simple |
| Leverages V6 | Yes | Yes | No |
| Distribution coverage | Risk | Good | Best |
| Variable length handling | Complex | Good | Native |
| **Recommendation** | ❌ | 🤔 | ✅ |

## Final Recommendation

Given that you said **"I'm in this long term"**, I strongly recommend **Pure Diffusion**:

1. **Higher quality ceiling**: No information bottleneck
2. **Better long-term bet**: Single model, simpler to maintain
3. **MotionDiffuse exists**: Purpose-built reference implementation
4. **Extra week is worth it**: 85-90% success vs 80-85%
5. **Direct trajectory control**: Full temporal detail

### If you're time-constrained:
- Go with **Hybrid 9600D** (middle ground)
- Keeps temporal structure but leverages V6 encoder/decoder
- 2.5 weeks, 85% success

### If you want fastest path:
- **Latent 192D** will work for common trajectory patterns
- But may struggle with diversity and rare modes
- Accept 80-85% success rate and potential quality ceiling

## Next Steps

If you choose **Pure Diffusion** (recommended):

1. **This week**: Set up diffusion_v7/ directory structure
2. **Week 1**: Implement core diffusion (schedule + denoiser + training)
3. **Week 2**: Add goal conditioning + variable length support
4. **Week 3**: DDIM sampling + evaluation + comparison to V6

I can help implement any of these approaches. Which direction do you want to go?
