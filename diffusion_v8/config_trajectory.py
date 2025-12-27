"""
Configuration for Trajectory Diffusion Model V8

V8 Key Changes from V7:
- input_dim: 8 -> 2 (x, y positions only)
- Removed condition_dim (no goal conditioning)
- Removed CFG parameters (use_cfg, cfg_dropout, cfg_guidance_scale)
- Added inpainting parameters

Direction control is now via endpoint inpainting, not learned embeddings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryDiffusionConfig:
    """Configuration for V8 trajectory diffusion model and training."""

    # =========================================================================
    # Data Configuration (V8: 2D positions only)
    # =========================================================================
    input_dim: int = 2              # V8: (x, y) positions only (was 8 in V7)
    max_seq_len: int = 200          # Maximum trajectory length
    # NOTE: condition_dim removed - V8 uses inpainting, not goal conditioning

    # =========================================================================
    # Model Architecture
    # =========================================================================
    latent_dim: int = 512           # Transformer hidden dimension
    num_layers: int = 9             # Transformer depth
    num_heads: int = 8              # Attention heads
    ff_size: int = 1024             # Feedforward layer size
    dropout: float = 0.1            # Dropout rate
    activation: str = 'gelu'        # Activation function

    # =========================================================================
    # CFG REMOVED IN V8
    # =========================================================================
    # The following V7 parameters are REMOVED:
    # - use_cfg: bool = True
    # - cfg_dropout: float = 0.1
    # - cfg_guidance_scale: float = 2.0
    # Direction is now controlled via inpainting, not CFG.

    # =========================================================================
    # Diffusion Configuration
    # =========================================================================
    diffusion_steps: int = 1000     # Number of diffusion timesteps (T)
    noise_schedule: str = 'linear'  # Beta schedule: 'linear', 'cosine', 'sqrt'
    beta_start: float = 1e-4        # Starting beta value
    beta_end: float = 0.02          # Ending beta value

    # Model output type
    model_mean_type: str = 'epsilon'    # 'epsilon' (predict noise), 'x0', 'v'
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
    # Sampling Configuration (V8: Inpainting)
    # =========================================================================
    sampling_method: str = 'ddim'   # 'ddpm', 'ddim'
    ddim_steps: int = 50            # DDIM sampling steps (faster)
    ddim_eta: float = 1.0           # DDIM eta (1.0 required for inpainting re-noising)
    clip_denoised: bool = False     # Clip denoised samples to [-1, 1]

    # Inpainting configuration (V8 specific)
    inpaint_start: bool = True      # Fix start position during generation
    inpaint_end: bool = True        # Fix end position during generation

    # =========================================================================
    # Data Loading
    # =========================================================================
    data_dir: str = 'processed_data_v8'  # V8 preprocessed data
    num_workers: int = 4                 # DataLoader workers
    pin_memory: bool = True              # Pin memory for GPU

    # =========================================================================
    # Logging & Checkpointing
    # =========================================================================
    log_interval: int = 10          # Log every N steps
    sample_interval: int = 100      # Generate samples every N steps
    checkpoint_interval: int = 500  # Save checkpoint every N steps
    save_dir: str = 'checkpoints_diffusion_v8'
    log_dir: str = 'logs_diffusion_v8'

    # =========================================================================
    # Device
    # =========================================================================
    device: str = 'cuda'            # 'cuda' or 'cpu'

    # =========================================================================
    # Debug / Testing
    # =========================================================================
    smoke_test: bool = False        # Enable smoke testing mode
    smoke_test_steps: int = 1000    # Steps for single-batch overfitting
    smoke_test_batch_size: int = 64 # Batch size for smoke test

    def __post_init__(self):
        """Validate configuration."""
        assert self.input_dim == 2, "V8 input_dim must be 2 (x, y positions)"
        assert self.num_heads > 0 and self.latent_dim % self.num_heads == 0, \
            f"latent_dim ({self.latent_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.diffusion_steps > 0, "Diffusion steps must be > 0"


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
        latent_dim=128,
        num_layers=4,
        num_heads=4,
        ff_size=256,

        # Smoke test settings
        smoke_test=True,
        smoke_test_steps=1000,
        smoke_test_batch_size=64,

        # Higher LR for faster convergence
        learning_rate=2e-4,

        # Disable LR scheduler
        use_lr_scheduler=False,

        # Logging
        log_interval=50,
        sample_interval=200,
    )


def get_medium_config() -> TrajectoryDiffusionConfig:
    """
    Medium model configuration for faster experimentation.
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
    )


def get_full_config() -> TrajectoryDiffusionConfig:
    """
    Full model configuration for production training.
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
    )


# =============================================================================
# Helper Functions
# =============================================================================

def print_config(config: TrajectoryDiffusionConfig):
    """Pretty print configuration."""
    print("=" * 70)
    print("Trajectory Diffusion V8 Configuration")
    print("=" * 70)

    print("\n[Data]")
    print(f"  Input dim: {config.input_dim} (x, y positions)")
    print(f"  Max sequence length: {config.max_seq_len}")

    print("\n[Model Architecture]")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  FF size: {config.ff_size}")
    print(f"  Dropout: {config.dropout}")

    print("\n[Conditioning]")
    print(f"  Method: Inpainting (V8)")
    print(f"  Inpaint start: {config.inpaint_start}")
    print(f"  Inpaint end: {config.inpaint_end}")

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
    ff = config.ff_size

    # Input/output projections (V8: input_dim=2)
    input_proj = config.input_dim * d
    output_proj = d * config.input_dim

    # Timestep embedding (no goal conditioner in V8)
    timestep_embed = d * d + d * d  # Two linear layers in time_embed

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
        # V8: cond_embed_dim = time_embed_dim = latent_dim (not 2*latent_dim)
        2 * (d * 2 * d + 2 * d)
    )
    transformer = L * per_layer

    total = input_proj + output_proj + timestep_embed + transformer
    return total


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("V8 SMOKE TEST CONFIG")
    print("=" * 70)
    smoke_config = get_smoke_test_config()
    print_config(smoke_config)

    print("\n\n" + "=" * 70)
    print("V8 MEDIUM CONFIG")
    print("=" * 70)
    medium_config = get_medium_config()
    print_config(medium_config)

    print("\n\n" + "=" * 70)
    print("V8 FULL CONFIG")
    print("=" * 70)
    full_config = get_full_config()
    print_config(full_config)
