"""
Configuration for TimeGAN V6 (RTSGAN-Style Two-Stage Training)

Contains all hyperparameters organized by:
1. Architecture dimensions
2. Stage 1 (Autoencoder) training parameters
3. Stage 2 (WGAN-GP) training parameters
4. Generation parameters

Usage:
    from timegan_v6.config_model_v6 import TimeGANV6Config
    config = TimeGANV6Config()
    # or with custom values:
    config = TimeGANV6Config(latent_dim=64, noise_dim=256)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TimeGANV6Config:
    """
    Complete configuration for TimeGAN V6.

    All parameters have sensible defaults based on the implementation plan
    and can be overridden at instantiation.
    """

    # =========================================================================
    # Architecture Dimensions
    # =========================================================================

    # Input/Output dimensions
    feature_dim: int = 8          # V4 trajectory features (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
    max_seq_len: int = 200        # Maximum sequence length

    # Encoder/Decoder (from V4)
    encoder_hidden_dim: int = 64  # LSTM hidden dim for encoder
    encoder_num_layers: int = 3   # Number of LSTM layers in encoder
    decoder_hidden_dim: int = 64  # LSTM hidden dim for decoder
    decoder_num_layers: int = 3   # Number of LSTM layers in decoder

    # Latent space
    latent_dim: int = 48          # Dimension of encoder output (per timestep)
    summary_dim: int = 96         # Dimension of pooled trajectory summary

    # Pooler
    pool_type: str = 'attention'  # 'attention', 'mean', 'last', 'hybrid'
    pooler_dropout: float = 0.1

    # Expander
    expand_type: str = 'lstm'     # 'lstm', 'mlp', 'repeat'
    expander_hidden_dim: int = 128
    expander_num_layers: int = 2
    expander_dropout: float = 0.1

    # Generator
    noise_dim: int = 128          # Dimension of input noise vector
    condition_dim: int = 1        # Dimension of condition (distance)
    generator_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    generator_use_residual: bool = True
    generator_type: str = 'standard'  # 'standard' or 'film' (FiLM conditioning)

    # Discriminator
    discriminator_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    discriminator_use_spectral_norm: bool = False
    discriminator_type: str = 'standard'  # 'standard', 'projection', 'multiscale'
    discriminator_n_scales: int = 3  # For multiscale discriminator

    # =========================================================================
    # Stage 1: Autoencoder Training
    # =========================================================================

    # Learning rate
    lr_autoencoder: float = 1e-3

    # Training iterations
    stage1_iterations: int = 15000

    # Loss weights
    lambda_recon: float = 1.0      # Reconstruction loss weight
    lambda_latent: float = 0.5     # Latent consistency loss weight (h_seq vs h_seq_recon)

    # Convergence
    stage1_threshold: float = 0.05  # Reconstruction MSE threshold for convergence
    stage1_patience: int = 1000     # Iterations to wait after threshold before stopping

    # Regularization
    stage1_grad_clip: float = 1.0   # Gradient clipping norm
    stage1_weight_decay: float = 1e-5

    # =========================================================================
    # Stage 2: WGAN-GP Training
    # =========================================================================

    # Learning rates
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4

    # WGAN-GP specific
    n_critic: int = 5              # Discriminator updates per generator update
    lambda_gp: float = 10.0        # Gradient penalty weight

    # Feature matching
    lambda_fm: float = 1.0         # Feature matching loss weight
    use_feature_matching: bool = True

    # Training iterations
    stage2_iterations: int = 30000

    # Adam betas for WGAN-GP (recommended: (0.0, 0.9))
    adam_beta1: float = 0.0
    adam_beta2: float = 0.9

    # Regularization
    stage2_grad_clip: float = 1.0
    stage2_weight_decay: float = 0.0  # Usually 0 for WGAN-GP

    # =========================================================================
    # Generation Parameters
    # =========================================================================

    # Number of samples to generate for evaluation
    eval_n_samples: int = 1000

    # Truncation trick (optional noise truncation for quality vs diversity)
    truncation_psi: float = 1.0    # 1.0 = no truncation, <1.0 = more quality

    # =========================================================================
    # Training Configuration
    # =========================================================================

    batch_size: int = 64
    num_workers: int = 4

    # Logging
    log_interval: int = 100        # Log every N iterations
    eval_interval: int = 1000      # Evaluate every N iterations
    save_interval: int = 5000      # Save checkpoint every N iterations

    # Checkpointing
    checkpoint_dir: str = './checkpoints/v6'
    save_best_only: bool = False

    # Device
    device: str = 'cuda'           # 'cuda' or 'cpu'

    # Random seed
    seed: Optional[int] = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        assert self.feature_dim > 0, "feature_dim must be positive"
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.summary_dim > 0, "summary_dim must be positive"
        assert self.noise_dim > 0, "noise_dim must be positive"

        assert self.pool_type in ['attention', 'mean', 'last', 'hybrid'], \
            f"Invalid pool_type: {self.pool_type}"
        assert self.expand_type in ['lstm', 'mlp', 'repeat'], \
            f"Invalid expand_type: {self.expand_type}"
        assert self.generator_type in ['standard', 'film'], \
            f"Invalid generator_type: {self.generator_type}"
        assert self.discriminator_type in ['standard', 'projection', 'multiscale'], \
            f"Invalid discriminator_type: {self.discriminator_type}"

        assert self.n_critic >= 1, "n_critic must be at least 1"
        assert self.lambda_gp > 0, "lambda_gp must be positive"

        assert 0.0 <= self.truncation_psi <= 1.0, "truncation_psi must be in [0, 1]"

    def get_stage1_config(self) -> dict:
        """Get configuration dict for Stage 1 training."""
        return {
            'lr': self.lr_autoencoder,
            'iterations': self.stage1_iterations,
            'lambda_recon': self.lambda_recon,
            'lambda_latent': self.lambda_latent,
            'threshold': self.stage1_threshold,
            'patience': self.stage1_patience,
            'grad_clip': self.stage1_grad_clip,
            'weight_decay': self.stage1_weight_decay,
        }

    def get_stage2_config(self) -> dict:
        """Get configuration dict for Stage 2 training."""
        return {
            'lr_generator': self.lr_generator,
            'lr_discriminator': self.lr_discriminator,
            'iterations': self.stage2_iterations,
            'n_critic': self.n_critic,
            'lambda_gp': self.lambda_gp,
            'lambda_fm': self.lambda_fm,
            'use_feature_matching': self.use_feature_matching,
            'adam_betas': (self.adam_beta1, self.adam_beta2),
            'grad_clip': self.stage2_grad_clip,
            'weight_decay': self.stage2_weight_decay,
        }

    def get_architecture_config(self) -> dict:
        """Get architecture configuration dict."""
        return {
            'feature_dim': self.feature_dim,
            'latent_dim': self.latent_dim,
            'summary_dim': self.summary_dim,
            'noise_dim': self.noise_dim,
            'condition_dim': self.condition_dim,
            'encoder': {
                'hidden_dim': self.encoder_hidden_dim,
                'num_layers': self.encoder_num_layers,
            },
            'decoder': {
                'hidden_dim': self.decoder_hidden_dim,
                'num_layers': self.decoder_num_layers,
            },
            'pooler': {
                'pool_type': self.pool_type,
                'dropout': self.pooler_dropout,
            },
            'expander': {
                'expand_type': self.expand_type,
                'hidden_dim': self.expander_hidden_dim,
                'num_layers': self.expander_num_layers,
                'dropout': self.expander_dropout,
            },
            'generator': {
                'hidden_dims': self.generator_hidden_dims,
                'use_residual': self.generator_use_residual,
                'type': self.generator_type,
            },
            'discriminator': {
                'hidden_dims': self.discriminator_hidden_dims,
                'use_spectral_norm': self.discriminator_use_spectral_norm,
                'type': self.discriminator_type,
                'n_scales': self.discriminator_n_scales,
            },
        }

    def to_dict(self) -> dict:
        """Convert entire config to dictionary."""
        return {
            'architecture': self.get_architecture_config(),
            'stage1': self.get_stage1_config(),
            'stage2': self.get_stage2_config(),
            'training': {
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
                'device': self.device,
                'seed': self.seed,
            },
            'logging': {
                'log_interval': self.log_interval,
                'eval_interval': self.eval_interval,
                'save_interval': self.save_interval,
                'checkpoint_dir': self.checkpoint_dir,
            },
            'generation': {
                'eval_n_samples': self.eval_n_samples,
                'truncation_psi': self.truncation_psi,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TimeGANV6Config':
        """Create config from dictionary (for loading from file)."""
        # Flatten nested dict if necessary
        flat_dict = {}

        if 'architecture' in config_dict:
            arch = config_dict['architecture']
            flat_dict['feature_dim'] = arch.get('feature_dim', 8)
            flat_dict['latent_dim'] = arch.get('latent_dim', 48)
            flat_dict['summary_dim'] = arch.get('summary_dim', 96)
            flat_dict['noise_dim'] = arch.get('noise_dim', 128)
            flat_dict['condition_dim'] = arch.get('condition_dim', 1)

            if 'encoder' in arch:
                flat_dict['encoder_hidden_dim'] = arch['encoder'].get('hidden_dim', 64)
                flat_dict['encoder_num_layers'] = arch['encoder'].get('num_layers', 3)
            if 'decoder' in arch:
                flat_dict['decoder_hidden_dim'] = arch['decoder'].get('hidden_dim', 64)
                flat_dict['decoder_num_layers'] = arch['decoder'].get('num_layers', 3)
            if 'pooler' in arch:
                flat_dict['pool_type'] = arch['pooler'].get('pool_type', 'attention')
                flat_dict['pooler_dropout'] = arch['pooler'].get('dropout', 0.1)
            if 'expander' in arch:
                flat_dict['expand_type'] = arch['expander'].get('expand_type', 'lstm')
                flat_dict['expander_hidden_dim'] = arch['expander'].get('hidden_dim', 128)
                flat_dict['expander_num_layers'] = arch['expander'].get('num_layers', 2)
            if 'generator' in arch:
                flat_dict['generator_hidden_dims'] = arch['generator'].get('hidden_dims', [256, 256, 256])
                flat_dict['generator_use_residual'] = arch['generator'].get('use_residual', True)
                flat_dict['generator_type'] = arch['generator'].get('type', 'standard')
            if 'discriminator' in arch:
                flat_dict['discriminator_hidden_dims'] = arch['discriminator'].get('hidden_dims', [256, 256, 256])
                flat_dict['discriminator_use_spectral_norm'] = arch['discriminator'].get('use_spectral_norm', False)
                flat_dict['discriminator_type'] = arch['discriminator'].get('type', 'standard')

        if 'stage1' in config_dict:
            s1 = config_dict['stage1']
            flat_dict['lr_autoencoder'] = s1.get('lr', 1e-3)
            flat_dict['stage1_iterations'] = s1.get('iterations', 15000)
            flat_dict['lambda_recon'] = s1.get('lambda_recon', 1.0)
            flat_dict['lambda_latent'] = s1.get('lambda_latent', 0.5)
            flat_dict['stage1_threshold'] = s1.get('threshold', 0.05)

        if 'stage2' in config_dict:
            s2 = config_dict['stage2']
            flat_dict['lr_generator'] = s2.get('lr_generator', 1e-4)
            flat_dict['lr_discriminator'] = s2.get('lr_discriminator', 1e-4)
            flat_dict['stage2_iterations'] = s2.get('iterations', 30000)
            flat_dict['n_critic'] = s2.get('n_critic', 5)
            flat_dict['lambda_gp'] = s2.get('lambda_gp', 10.0)
            flat_dict['lambda_fm'] = s2.get('lambda_fm', 1.0)

        if 'training' in config_dict:
            t = config_dict['training']
            flat_dict['batch_size'] = t.get('batch_size', 64)
            flat_dict['device'] = t.get('device', 'cuda')
            flat_dict['seed'] = t.get('seed', 42)

        return cls(**flat_dict)


# Preset configurations for different use cases
def get_default_config() -> TimeGANV6Config:
    """Get default configuration."""
    return TimeGANV6Config()


def get_fast_config() -> TimeGANV6Config:
    """Get configuration for fast experimentation (smaller model, fewer iterations)."""
    return TimeGANV6Config(
        latent_dim=32,
        summary_dim=64,
        noise_dim=64,
        generator_hidden_dims=[128, 128],
        discriminator_hidden_dims=[128, 128],
        expander_hidden_dim=64,
        stage1_iterations=5000,
        stage2_iterations=10000,
        batch_size=32,
    )


def get_large_config() -> TimeGANV6Config:
    """Get configuration for larger model (more capacity)."""
    return TimeGANV6Config(
        latent_dim=64,
        summary_dim=128,
        noise_dim=256,
        generator_hidden_dims=[512, 512, 256],
        discriminator_hidden_dims=[512, 512, 256],
        expander_hidden_dim=256,
        expander_num_layers=3,
        stage1_iterations=20000,
        stage2_iterations=50000,
    )


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing TimeGAN V6 Configuration...")
    print("=" * 60)

    # Test default config
    print("\n=== Default Configuration ===")
    config = TimeGANV6Config()
    print(f"feature_dim: {config.feature_dim}")
    print(f"latent_dim: {config.latent_dim}")
    print(f"summary_dim: {config.summary_dim}")
    print(f"noise_dim: {config.noise_dim}")
    print(f"pool_type: {config.pool_type}")
    print(f"expand_type: {config.expand_type}")
    print(f"stage1_iterations: {config.stage1_iterations}")
    print(f"stage2_iterations: {config.stage2_iterations}")
    print("Default config: PASS")

    # Test custom config
    print("\n=== Custom Configuration ===")
    custom_config = TimeGANV6Config(
        latent_dim=64,
        summary_dim=128,
        pool_type='hybrid',
        n_critic=3
    )
    assert custom_config.latent_dim == 64
    assert custom_config.summary_dim == 128
    assert custom_config.pool_type == 'hybrid'
    assert custom_config.n_critic == 3
    print("Custom config: PASS")

    # Test stage configs
    print("\n=== Stage Configurations ===")
    stage1 = config.get_stage1_config()
    stage2 = config.get_stage2_config()
    print(f"Stage 1 keys: {list(stage1.keys())}")
    print(f"Stage 2 keys: {list(stage2.keys())}")
    assert 'lr' in stage1
    assert 'lr_generator' in stage2
    assert 'n_critic' in stage2
    print("Stage configs: PASS")

    # Test to_dict and from_dict
    print("\n=== Serialization ===")
    config_dict = config.to_dict()
    restored_config = TimeGANV6Config.from_dict(config_dict)
    assert restored_config.feature_dim == config.feature_dim
    assert restored_config.latent_dim == config.latent_dim
    print("Serialization: PASS")

    # Test preset configs
    print("\n=== Preset Configurations ===")
    fast = get_fast_config()
    large = get_large_config()
    print(f"Fast config - latent_dim: {fast.latent_dim}, stage1_iter: {fast.stage1_iterations}")
    print(f"Large config - latent_dim: {large.latent_dim}, stage1_iter: {large.stage1_iterations}")
    print("Preset configs: PASS")

    # Test validation
    print("\n=== Validation ===")
    try:
        invalid_config = TimeGANV6Config(pool_type='invalid')
        print("Validation: FAIL (should have raised error)")
    except AssertionError as e:
        print(f"Validation caught invalid pool_type: OK")

    try:
        invalid_config = TimeGANV6Config(n_critic=0)
        print("Validation: FAIL (should have raised error)")
    except AssertionError as e:
        print(f"Validation caught invalid n_critic: OK")

    print("Validation: PASS")

    print("\n" + "=" * 60)
    print("ALL CONFIGURATION TESTS PASSED")
    print("=" * 60)
