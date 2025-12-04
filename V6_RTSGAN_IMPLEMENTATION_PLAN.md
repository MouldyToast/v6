# TimeGAN V6: RTSGAN-Style Two-Stage Training

## Executive Summary

V6 fundamentally restructures the training paradigm from TimeGAN's joint training to RTSGAN's **strictly decoupled two-stage approach**. The key insight: by training the autoencoder to convergence first, then training a WGAN generator in the **frozen latent space**, we eliminate the encoder-generator semantic gap that plagued V4's Phase 3.

### Why This Approach

The core problem in V4 Phase 3:
- Embedder learns to reconstruct paths from real data (rich semantic signal)
- Generator receives only weak adversarial signal through the decoder
- The two networks have **no shared learning objective** - they're asymmetric

RTSGAN's solution:
1. **Stage 1**: Train autoencoder until latent space is stable and meaningful
2. **Stage 2**: Freeze encoder/decoder, train WGAN to generate in that **fixed** latent space
3. The discriminator operates on latent vectors, not decoded sequences
4. This gives the generator **direct, dense feedback** about latent space quality

---

## Architecture Overview

### Key Architectural Decisions

**Decision 1: Sequence-Level vs Single-Vector Latent Space**

RTSGAN pools the entire time series to a single latent vector. For trajectories, we have two options:

| Approach | Pros | Cons |
|----------|------|------|
| **Single Vector (RTSGAN-style)** | Simpler GAN, faster training, easier distribution matching | Loses fine-grained temporal detail |
| **Sequence of Vectors (TimeGAN-style)** | Preserves temporal structure | More complex GAN, harder to train |

**Recommendation**: **Hybrid approach**
- Pool encoder output to a **trajectory summary vector** (global features)
- Keep **sequence-level generation** but condition on summary + noise
- Discriminator operates on **pooled latent representations**

This captures the best of both: global trajectory characteristics in summary vector, temporal dynamics in sequence.

**Decision 2: Condition Integration**

Trajectory distance condition must be integrated properly:
- **Stage 1**: Autoencoder is unconditional (learns pure latent representation)
- **Stage 2**: Generator is conditioned on distance, produces latent sequences
- Discriminator receives (latent_summary, condition) pairs

---

## Detailed Architecture

### Component Diagram

```
STAGE 1: Autoencoder Training (Frozen after convergence)
═══════════════════════════════════════════════════════

x_real ──→ [Encoder] ──→ h_seq ──→ [Pool] ──→ z_summary
              │                                    │
              │            ┌───────────────────────┘
              ▼            ▼
         [Decoder] ←── [Unpool]
              │
              ▼
          x_recon

Loss: L_recon = MSE(x_real, x_recon)


STAGE 2: WGAN-GP in Frozen Latent Space
═══════════════════════════════════════

                    ┌─────────────────────────┐
                    │   Real Path (frozen)    │
                    │                         │
x_real ──→ [Encoder*] ──→ h_real ──→ [Pool*] ──→ z_real
                                                   │
                                                   ▼
                                            ┌──────────┐
              ┌─────────────────────────────│Discrim(D)│◄───┐
              │                             └──────────┘    │
              │                                   │         │
              │     noise + condition             │         │
              │           │                       ▼         │
              │           ▼               WGAN-GP Loss      │
              │     ┌───────────┐                           │
              │     │Generator  │                           │
              │     │    (G)    │                           │
              │     └───────────┘                           │
              │           │                                 │
              │           ▼                                 │
              │      z_fake (latent summary)────────────────┘
              │           │
              │           ▼
              │    ┌────────────┐
              │    │Seq Expand  │  (MLP to sequence of latent vectors)
              │    └────────────┘
              │           │
              │           ▼
              │       h_fake_seq
              │           │
              │           ▼
              └────→ [Decoder*] ──→ x_fake
                    (* = FROZEN)

Key: Discriminator operates on LATENT SPACE (z_real vs z_fake)
     Not on decoded feature space
```

### New Components for V6

#### 1. LatentPooler (Sequence → Summary Vector)

```python
class LatentPooler(nn.Module):
    """
    Pool sequence of latent vectors to single trajectory summary.
    
    Options:
    - 'last': Use last hidden state
    - 'mean': Average pool
    - 'attention': Learned attention pooling
    - 'hybrid': Concat last + mean + max
    """
    
    def __init__(self, latent_dim, pool_type='attention', output_dim=None):
        super().__init__()
        self.pool_type = pool_type
        output_dim = output_dim or latent_dim
        
        if pool_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.Tanh(),
                nn.Linear(latent_dim // 2, 1)
            )
            self.project = nn.Linear(latent_dim, output_dim)
            
        elif pool_type == 'hybrid':
            # Concat last + mean + max → 3x latent_dim → output_dim
            self.project = nn.Linear(latent_dim * 3, output_dim)
    
    def forward(self, h_seq, lengths):
        """
        Args:
            h_seq: (batch, seq_len, latent_dim)
            lengths: (batch,)
        Returns:
            z_summary: (batch, output_dim)
        """
        if self.pool_type == 'attention':
            # Compute attention weights
            attn_scores = self.attention(h_seq)  # (batch, seq_len, 1)
            
            # Mask padding
            mask = create_mask(lengths, h_seq.size(1))
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = (h_seq * attn_weights).sum(dim=1)
            return self.project(pooled)
            
        elif self.pool_type == 'hybrid':
            # Get last valid timestep for each sequence
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(h_seq.size(0))
            h_last = h_seq[batch_idx, last_idx]
            
            # Mean pool (masked)
            mask = create_mask(lengths, h_seq.size(1)).unsqueeze(-1)
            h_mean = (h_seq * mask).sum(dim=1) / lengths.unsqueeze(1).float()
            
            # Max pool (masked) 
            h_max = (h_seq * mask + (~mask) * -1e9).max(dim=1)[0]
            
            return self.project(torch.cat([h_last, h_mean, h_max], dim=-1))
```

#### 2. LatentExpander (Summary Vector → Sequence)

```python
class LatentExpander(nn.Module):
    """
    Expand latent summary back to sequence of latent vectors.
    
    The generator produces z_summary. This expands it to h_fake_seq
    that can be fed to the frozen decoder.
    """
    
    def __init__(self, summary_dim, latent_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.summary_dim = summary_dim
        self.latent_dim = latent_dim
        
        # Transform summary to initial LSTM hidden state
        self.init_hidden = nn.Linear(summary_dim, hidden_dim * num_layers)
        self.init_cell = nn.Linear(summary_dim, hidden_dim * num_layers)
        
        # LSTM to generate sequence autoregressively
        self.lstm = nn.LSTM(
            input_size=latent_dim,  # Previous output
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Project to latent dim
        self.output = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z_summary, seq_len):
        """
        Args:
            z_summary: (batch, summary_dim)
            seq_len: int, max sequence length to generate
        Returns:
            h_seq: (batch, seq_len, latent_dim)
        """
        batch_size = z_summary.size(0)
        device = z_summary.device
        
        # Initialize LSTM state from summary
        h0 = self.init_hidden(z_summary).view(batch_size, self.lstm.num_layers, -1).transpose(0, 1).contiguous()
        c0 = self.init_cell(z_summary).view(batch_size, self.lstm.num_layers, -1).transpose(0, 1).contiguous()
        
        # Start token (zeros)
        h_prev = torch.zeros(batch_size, 1, self.latent_dim, device=device)
        
        outputs = []
        hidden = (h0, c0)
        
        for t in range(seq_len):
            out, hidden = self.lstm(h_prev, hidden)
            h_t = self.output(out)
            h_t = torch.tanh(h_t)  # Bound to [-1, 1]
            outputs.append(h_t)
            h_prev = h_t
            
        return torch.cat(outputs, dim=1)
```

#### 3. LatentGenerator (WGAN Generator)

```python
class LatentGenerator(nn.Module):
    """
    MLP generator that produces latent summary vectors.
    Much simpler than TimeGAN's sequence generator.
    
    Operates entirely on fixed-dimension vectors:
    noise (z_dim) + condition (cond_dim) → latent summary (summary_dim)
    """
    
    def __init__(self, noise_dim=128, condition_dim=1, summary_dim=96, 
                 hidden_dims=[256, 256, 256]):
        super().__init__()
        
        self.noise_dim = noise_dim
        
        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64)
        )
        
        # Main generator network
        layers = []
        in_dim = noise_dim + 64  # noise + condition embedding
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, summary_dim))
        layers.append(nn.Tanh())  # Bound to [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z, condition):
        """
        Args:
            z: (batch, noise_dim) - random noise
            condition: (batch, condition_dim) - distance condition
        Returns:
            z_summary: (batch, summary_dim) - generated latent summary
        """
        cond_emb = self.cond_embed(condition)
        x = torch.cat([z, cond_emb], dim=-1)
        return self.network(x)
```

#### 4. LatentDiscriminator (WGAN Critic)

```python
class LatentDiscriminator(nn.Module):
    """
    MLP discriminator/critic for WGAN-GP.
    Operates on latent summary vectors, NOT decoded sequences.
    
    This is the key insight: discriminating in latent space
    gives generator direct feedback about latent quality.
    """
    
    def __init__(self, summary_dim=96, condition_dim=1, 
                 hidden_dims=[256, 256, 256]):
        super().__init__()
        
        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64)
        )
        
        # Main discriminator network
        layers = []
        in_dim = summary_dim + 64  # latent + condition embedding
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),  # LayerNorm for WGAN-GP
                nn.LeakyReLU(0.2),
            ])
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        # No sigmoid - WGAN uses raw scores
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z_summary, condition):
        """
        Args:
            z_summary: (batch, summary_dim) - latent summary
            condition: (batch, condition_dim) - distance condition
        Returns:
            score: (batch, 1) - Wasserstein distance estimate
        """
        cond_emb = self.cond_embed(condition)
        x = torch.cat([z_summary, cond_emb], dim=-1)
        return self.network(x)
```

---

## Training Pipeline

### Stage 1: Autoencoder Training (To Convergence)

**Goal**: Learn a stable, meaningful latent space that captures trajectory structure.

```python
def train_stage1_autoencoder(model, train_loader, config):
    """
    Train Encoder + Pooler + Expander + Decoder until convergence.
    
    This stage learns:
    1. Encoder: x → h_seq (sequence of latent vectors)
    2. Pooler: h_seq → z_summary (trajectory summary)
    3. Expander: z_summary → h_seq_recon (back to sequence)
    4. Decoder: h_seq → x_recon (back to features)
    
    The bottleneck through z_summary forces learning of
    global trajectory characteristics.
    """
    
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.pooler.parameters()},
        {'params': model.expander.parameters()},
        {'params': model.decoder.parameters()},
    ], lr=config['lr_autoencoder'])
    
    for iteration in range(config['stage1_iterations']):
        x_real, condition, lengths = next(train_loader)
        
        # Forward through autoencoder
        h_seq = model.encoder(x_real, lengths)
        z_summary = model.pooler(h_seq, lengths)
        h_seq_recon = model.expander(z_summary, x_real.size(1))
        x_recon = model.decoder(h_seq_recon, lengths)
        
        # Reconstruction loss
        loss_recon = masked_mse_loss(x_real, x_recon, lengths)
        
        # Optional: Latent consistency loss
        # Encourage h_seq_recon to match h_seq
        loss_latent = masked_mse_loss(h_seq, h_seq_recon, lengths)
        
        # Total loss
        loss = loss_recon + config['lambda_latent'] * loss_latent
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check convergence
        if loss_recon < config['stage1_threshold']:
            print(f"Stage 1 converged at iteration {iteration}")
            break
    
    # FREEZE all autoencoder components
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.pooler.parameters():
        param.requires_grad = False
    for param in model.expander.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
```

**Key Hyperparameters for Stage 1**:
- `lr_autoencoder`: 1e-3 (standard)
- `stage1_iterations`: 10,000-20,000 (until convergence)
- `lambda_latent`: 0.5 (latent consistency weight)
- `stage1_threshold`: 0.05 (reconstruction MSE to consider converged)

### Stage 2: WGAN-GP in Frozen Latent Space

**Goal**: Train generator to produce latent summaries matching real distribution.

```python
def train_stage2_wgan(model, train_loader, config):
    """
    Train LatentGenerator and LatentDiscriminator with WGAN-GP.
    
    Key insight: Discriminator operates on latent summaries (z_summary),
    NOT on decoded sequences. This gives generator direct feedback.
    
    The frozen encoder provides the target distribution:
    z_real = Pooler(Encoder(x_real))
    
    Generator learns to match: z_fake ≈ z_real
    """
    
    opt_G = torch.optim.Adam(
        model.generator.parameters(),
        lr=config['lr_generator'],
        betas=(0.0, 0.9)  # WGAN-GP recommended
    )
    opt_D = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=config['lr_discriminator'],
        betas=(0.0, 0.9)
    )
    
    for iteration in range(config['stage2_iterations']):
        
        # ─── Discriminator Update ─────────────────────────
        for _ in range(config['n_critic']):
            x_real, condition, lengths = next(train_loader)
            batch_size = x_real.size(0)
            
            # Get real latent summaries (frozen encoder)
            with torch.no_grad():
                h_seq_real = model.encoder(x_real, lengths)
                z_real = model.pooler(h_seq_real, lengths)
            
            # Generate fake latent summaries
            noise = torch.randn(batch_size, config['noise_dim'], device=device)
            z_fake = model.generator(noise, condition)
            
            # Discriminator scores
            d_real = model.discriminator(z_real, condition)
            d_fake = model.discriminator(z_fake.detach(), condition)
            
            # Gradient penalty
            gp = compute_gradient_penalty_latent(
                model.discriminator, z_real, z_fake, condition
            )
            
            # WGAN-GP discriminator loss
            loss_D = d_fake.mean() - d_real.mean() + config['lambda_gp'] * gp
            
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        
        # ─── Generator Update ─────────────────────────────
        noise = torch.randn(batch_size, config['noise_dim'], device=device)
        z_fake = model.generator(noise, condition)
        
        d_fake = model.discriminator(z_fake, condition)
        
        # WGAN generator loss (maximize critic score)
        loss_G = -d_fake.mean()
        
        # Optional: Feature matching loss
        # Match first moments of latent distributions
        with torch.no_grad():
            h_seq_real = model.encoder(x_real, lengths)
            z_real = model.pooler(h_seq_real, lengths)
        
        loss_fm = F.mse_loss(z_fake.mean(dim=0), z_real.mean(dim=0))
        loss_fm += F.mse_loss(z_fake.std(dim=0), z_real.std(dim=0))
        
        loss_G_total = loss_G + config['lambda_fm'] * loss_fm
        
        opt_G.zero_grad()
        loss_G_total.backward()
        opt_G.step()
```

**Key Hyperparameters for Stage 2**:
- `lr_generator`: 1e-4
- `lr_discriminator`: 1e-4
- `n_critic`: 5 (discriminator updates per generator update)
- `lambda_gp`: 10.0 (gradient penalty weight)
- `lambda_fm`: 1.0 (feature matching weight)
- `stage2_iterations`: 20,000-50,000

---

## Generation Pipeline

```python
def generate_trajectories(model, n_samples, conditions, max_len, device):
    """
    Generate trajectories using trained V6 model.
    
    Pipeline:
    1. Sample noise z ~ N(0, 1)
    2. Generator produces latent summary: z_summary = G(noise, condition)
    3. Expander creates sequence: h_seq = Expander(z_summary, max_len)
    4. Decoder reconstructs features: x_fake = Decoder(h_seq)
    """
    model.eval()
    
    with torch.no_grad():
        # Sample noise
        noise = torch.randn(n_samples, model.noise_dim, device=device)
        
        # Generate latent summaries
        z_summary = model.generator(noise, conditions)
        
        # Expand to sequence
        h_seq = model.expander(z_summary, max_len)
        
        # Decode to features
        lengths = torch.full((n_samples,), max_len, device=device)
        x_fake = model.decoder(h_seq, lengths)
        
        # Clamp to valid range
        x_fake = torch.clamp(x_fake, -1, 1)
    
    return x_fake
```

---

## File Structure

```
timegan_v6/
├── config_model_v6.py        # V6 configuration
├── timegan_v6.py             # Main model wrapper
├── embedder_v6.py            # Encoder (reuse from V4 or enhance)
├── recovery_v6.py            # Decoder (reuse from V4 or enhance)
├── pooler_v6.py              # NEW: Latent pooler
├── expander_v6.py            # NEW: Latent expander
├── latent_generator_v6.py    # NEW: MLP generator for latent space
├── latent_discriminator_v6.py# NEW: MLP discriminator for latent space
├── trainer_v6.py             # Two-stage training pipeline
├── losses_v6.py              # Loss functions
└── utils_v6.py               # Utilities
```

---

## Implementation Phases

### Phase 1: Core Components (Days 1-2)

1. **LatentPooler** (`pooler_v6.py`)
   - Implement attention-based pooling
   - Add mask handling for variable lengths
   - Test with random inputs

2. **LatentExpander** (`expander_v6.py`)
   - Implement LSTM-based expansion
   - Initialize from summary vector
   - Test expansion/decoding

3. **LatentGenerator** (`latent_generator_v6.py`)
   - Simple MLP with condition embedding
   - Layer normalization for stability
   - Tanh output activation

4. **LatentDiscriminator** (`latent_discriminator_v6.py`)
   - MLP critic for WGAN-GP
   - No sigmoid (raw scores)
   - Layer normalization

### Phase 2: Model Integration (Day 3)

5. **TimeGANV6** (`timegan_v6.py`)
   - Compose all components
   - Implement freeze/unfreeze methods
   - Multiple forward modes

6. **Configuration** (`config_model_v6.py`)
   - Stage-specific hyperparameters
   - Architecture dimensions
   - Loss weights

### Phase 3: Training Pipeline (Days 4-5)

7. **Stage 1 Trainer**
   - Autoencoder training loop
   - Convergence detection
   - Latent space visualization

8. **Stage 2 Trainer**
   - WGAN-GP in latent space
   - Feature matching loss
   - Generation validation

9. **Full Pipeline** (`trainer_v6.py`)
   - Orchestrate both stages
   - Checkpointing between stages
   - TensorBoard logging

### Phase 4: Evaluation & Refinement (Days 6-7)

10. **Evaluation Suite**
    - Trajectory quality metrics
    - Latent space analysis
    - Comparison with V4

11. **Hyperparameter Tuning**
    - Grid search key parameters
    - Ablation studies

---

## Expected Improvements Over V4

| Aspect | V4 Issue | V6 Solution | Expected Improvement |
|--------|----------|-------------|---------------------|
| **Semantic Gap** | Generator learns from weak adversarial signal | Direct latent space discrimination | Major |
| **Training Stability** | Phase 3 often destabilizes Phase 1 learning | Frozen autoencoder in Stage 2 | Major |
| **Discriminator Dominance** | Feature-space D overwhelms G | Simpler MLP D in latent space | Moderate |
| **Condition Fidelity** | Weak condition signal through sequence | Direct condition in latent summary | Moderate |
| **Training Speed** | Complex sequence-level GAN | Simple vector-level WGAN | 2-3x faster |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Summary vector loses temporal detail** | Keep sequence generation in Expander; add temporal consistency loss |
| **Expander learns trivial mapping** | Pre-train Expander jointly with autoencoder; latent consistency loss |
| **Mode collapse in latent space** | Feature matching loss; larger noise dimension |
| **Condition ignored by generator** | Condition reconstruction loss; strong condition embedding |

---

## Success Criteria

1. **Stage 1 Convergence**: Reconstruction MSE < 0.05 within 10k iterations
2. **Latent Distribution Match**: MMD(z_real, z_fake) < 0.1
3. **Generation Quality**: 
   - Discriminative score < 0.3 (50% would be random)
   - Predictive score improved over V4
   - Visual trajectory inspection passes
4. **Condition Fidelity**: Distance error < 10% vs target

---

## Next Steps

1. Review this plan and discuss any concerns
2. Start with Phase 1: Core components
3. Test each component independently before integration
4. Compare early Stage 1 results with V4's autoencoder

Would you like me to start implementing any specific component?
