"""
Configuration for Trajectory Diffusion Model

Provides configurable hyperparameters for model architecture, training, and sampling.
Supports both smoke testing (small model) and full training (large model).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryDiffusionConfig:
    """Configuration for trajectory diffusion model and training."""

    # =========================================================================
    # Data Configuration
    # =========================================================================
    input_dim: int = 8              # Trajectory features (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
    condition_dim: int = 3          # Goal conditioning (distance_norm, cos(angle), sin(angle))
    max_seq_len: int = 200          # Maximum trajectory length

    # =========================================================================
    # Model Architecture - CONFIGURABLE for smoke testing vs full training
    # =========================================================================
    latent_dim: int = 512           # Transformer hidden dimension (reduce for smoke test)
    num_layers: int = 9             # Transformer depth (reduce for smoke test)
    num_heads: int = 8              # Attention heads (reduce for smoke test)
    ff_size: int = 1024             # Feedforward layer size (reduce for smoke test)
    dropout: float = 0.1            # Dropout rate
    activation: str = 'gelu'        # Activation function

    # =========================================================================
    # Classifier-Free Guidance (CFG)
    # =========================================================================
    use_cfg: bool = True                    # Enable classifier-free guidance
    cfg_dropout: float = 0.1                # Probability of dropping condition during training
    cfg_guidance_scale: float = 2.0         # Guidance scale during sampling (>1 = stronger conditioning)

    # =========================================================================
    # Diffusion Configuration
    # =========================================================================
    diffusion_steps: int = 1000     # Number of diffusion timesteps (T)
    noise_schedule: str = 'linear'  # Beta schedule: 'linear', 'cosine', 'sqrt'
    beta_start: float = 1e-4        # Starting beta value
    beta_end: float = 0.02          # Ending beta value

    # Model output type
    model_mean_type: str = 'epsilon'    # 'epsilon' (predict noise), 'x0' (predict clean), 'v' (velocity)
    model_var_type: str = 'fixed_small' # 'fixed_small', 'fixed_large', 'learned'
    loss_type: str = 'mse'              # 'mse', 'l1'

    # =========================================================================
    # Training Configuration
    # =========================================================================
    batch_size: int = 64            # Training batch size
    learning_rate: float = 1e-4     # Adam learning rate
    weight_decay: float = 0.0       # L2 regularization
    grad_clip: float = 0.5          # Gradient clipping threshold
    num_epochs: int = 1000          # Total training epochs

    # Optimizer
    optimizer: str = 'adam'         # 'adam', 'adamw'
    betas: tuple = (0.9, 0.999)     # Adam betas

    # Learning rate schedule
    use_lr_scheduler: bool = False  # Enable LR scheduling
    lr_warmup_steps: int = 500      # Warmup steps
    lr_decay_steps: int = 10000     # Decay steps

    # =========================================================================
    # Sampling Configuration
    # =========================================================================
    sampling_method: str = 'ddim'   # 'ddpm', 'ddim'
    ddim_steps: int = 50            # DDIM sampling steps (faster)
    ddim_eta: float = 0.0           # DDIM eta (0=deterministic, 1=DDPM)
    clip_denoised: bool = False     # Clip denoised samples to [-1, 1]

    # =========================================================================
    # Data Loading
    # =========================================================================
    data_dir: str = 'processed_data_v6'  # Directory with preprocessed V6 data
    num_workers: int = 4                 # DataLoader workers
    pin_memory: bool = True              # Pin memory for GPU

    # =========================================================================
    # Logging & Checkpointing
    # =========================================================================
    log_interval: int = 10          # Log every N steps
    sample_interval: int = 100      # Generate samples every N steps
    checkpoint_interval: int = 500  # Save checkpoint every N steps
    save_dir: str = 'checkpoints_diffusion_v7'
    log_dir: str = 'logs_diffusion_v7'

    # =========================================================================
    # Device
    # =========================================================================
    device: str = 'cuda'            # 'cuda' or 'cpu'

    # =========================================================================
    # Debug / Testing
    # =========================================================================
    smoke_test: bool = False        # Enable smoke testing mode (single batch overfitting)
    smoke_test_steps: int = 1000    # Steps for single-batch overfitting test
    smoke_test_batch_size: int = 64 # Batch size for smoke test

    def __post_init__(self):
        """Validate configuration."""
        assert self.input_dim == 8, "Input dim must be 8 (trajectory features)"
        assert self.condition_dim == 3, "Condition dim must be 3 (distance + angle)"
        assert self.num_heads > 0 and self.latent_dim % self.num_heads == 0, \
            f"latent_dim ({self.latent_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.diffusion_steps > 0, "Diffusion steps must be > 0"
        assert 0 <= self.cfg_dropout <= 1, "CFG dropout must be in [0, 1]"
        assert self.cfg_guidance_scale >= 1.0, "CFG guidance scale must be >= 1.0"


# =============================================================================
# Preset Configurations
# =============================================================================

def get_smoke_test_config() -> TrajectoryDiffusionConfig:
    """
    Small model configuration for smoke testing (single-batch overfitting).

    Goal: Verify architecture works by overfitting on 64 samples.
    Should reach near-zero loss within 1000 steps.
    """
    return TrajectoryDiffusionConfig(
        # Small model for fast testing
        latent_dim=128,             # Reduced from 512
        num_layers=4,               # Reduced from 9
        num_heads=4,                # Reduced from 8
        ff_size=256,                # Reduced from 1024

        # Smoke test settings
        smoke_test=True,
        smoke_test_steps=1000,
        smoke_test_batch_size=64,

        # Higher LR for faster convergence on small batch
        learning_rate=2e-4,

        # Disable some features for simplicity
        use_lr_scheduler=False,

        # Logging
        log_interval=50,
        sample_interval=200,

        # CFG enabled (test this too)
        use_cfg=True,
        cfg_dropout=0.1,
    )


def get_full_config() -> TrajectoryDiffusionConfig:
    """
    Full model configuration for production training.

    Uses MotionDiffuse-scale architecture (but tunable).
    """
    return TrajectoryDiffusionConfig(
        # Full model size
        latent_dim=512,
        num_layers=9,
        num_heads=8,
        ff_size=1024,

        # Full training settings
        smoke_test=False,
        num_epochs=1000,
        batch_size=64,
        learning_rate=1e-4,

        # Enable LR scheduler for long training
        use_lr_scheduler=True,
        lr_warmup_steps=500,
        lr_decay_steps=10000,

        # CFG enabled
        use_cfg=True,
        cfg_dropout=0.1,
        cfg_guidance_scale=2.0,
    )


def get_medium_config() -> TrajectoryDiffusionConfig:
    """
    Medium model configuration for faster experimentation.

    Balance between smoke test and full model.
    """
    return TrajectoryDiffusionConfig(
        # Medium model size
        latent_dim=256,
        num_layers=6,
        num_heads=8,
        ff_size=512,

        # Training settings
        smoke_test=False,
        num_epochs=500,
        batch_size=64,
        learning_rate=1e-4,

        # CFG enabled
        use_cfg=True,
        cfg_dropout=0.1,
        cfg_guidance_scale=2.0,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def print_config(config: TrajectoryDiffusionConfig):
    """Pretty print configuration."""
    print("=" * 70)
    print("Trajectory Diffusion Configuration")
    print("=" * 70)

    print("\n[Data]")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Condition dim: {config.condition_dim}")
    print(f"  Max sequence length: {config.max_seq_len}")

    print("\n[Model Architecture]")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  FF size: {config.ff_size}")
    print(f"  Dropout: {config.dropout}")

    print("\n[Classifier-Free Guidance]")
    print(f"  Enabled: {config.use_cfg}")
    print(f"  Dropout rate: {config.cfg_dropout}")
    print(f"  Guidance scale: {config.cfg_guidance_scale}")

    print("\n[Diffusion]")
    print(f"  Steps: {config.diffusion_steps}")
    print(f"  Noise schedule: {config.noise_schedule}")
    print(f"  Beta range: [{config.beta_start}, {config.beta_end}]")

    print("\n[Training]")
    if config.smoke_test:
        print(f"  MODE: SMOKE TEST (single batch overfitting)")
        print(f"  Smoke test steps: {config.smoke_test_steps}")
        print(f"  Smoke test batch size: {config.smoke_test_batch_size}")
    else:
        print(f"  MODE: Full training")
        print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Grad clip: {config.grad_clip}")

    print("\n[Sampling]")
    print(f"  Method: {config.sampling_method}")
    print(f"  DDIM steps: {config.ddim_steps}")
    print(f"  DDIM eta: {config.ddim_eta}")

    print("\n[Paths]")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Save dir: {config.save_dir}")
    print(f"  Log dir: {config.log_dir}")

    # Model size estimate
    params = estimate_params(config)
    print(f"\n[Model Size]")
    print(f"  Estimated parameters: {params:,}")
    print("=" * 70)


def estimate_params(config: TrajectoryDiffusionConfig) -> int:
    """Estimate number of model parameters."""
    d = config.latent_dim
    L = config.num_layers
    h = config.num_heads
    ff = config.ff_size

    # Input/output projections
    input_proj = config.input_dim * d
    output_proj = d * config.input_dim

    # Condition embedding
    condition_embed = config.condition_dim * 128 + 128 * 256 + 256 * d

    # Timestep embedding (sinusoidal, no params)
    timestep_embed = 0

    # Transformer layers
    per_layer = (
        # Self-attention
        4 * d * d +  # Q, K, V, O projections
        # LayerNorm
        2 * d +
        # Feedforward
        d * ff + ff * d +
        # LayerNorm
        2 * d +
        # StylizationBlocks (2 per layer)
        2 * (d * 2 * d + 2 * d)
    )
    transformer = L * per_layer

    total = input_proj + output_proj + condition_embed + transformer
    return total


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SMOKE TEST CONFIG")
    print("="*70)
    smoke_config = get_smoke_test_config()
    print_config(smoke_config)

    print("\n\n" + "="*70)
    print("MEDIUM CONFIG")
    print("="*70)
    medium_config = get_medium_config()
    print_config(medium_config)

    print("\n\n" + "="*70)
    print("FULL CONFIG")
    print("="*70)
    full_config = get_full_config()
    print_config(full_config)
