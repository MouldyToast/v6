"""
V4 Model and Training Configuration

TimeGAN V4 - Position-Independent Style Learning

Key changes from V3:
- feature_dim: 4 -> 8 (dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
- Larger hidden dimensions to handle increased feature complexity
- Adjusted learning rates for new feature space

Feature Index Reference:
    [0] dx       - Relative x position from trajectory start
    [1] dy       - Relative y position from trajectory start
    [2] speed    - Movement speed (pixels/second)
    [3] accel    - Acceleration (pixels/second^2) - STYLE SIGNATURE
    [4] sin_h    - Sine of heading angle
    [5] cos_h    - Cosine of heading angle
    [6] ang_vel  - Angular velocity (radians/second) - CURVATURE STYLE
    [7] dt       - Time delta between samples (seconds)
"""

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

MODEL_CONFIG_V4 = {
    # Feature dimensions
    'feature_dim': 8,              # V4: dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt
    'condition_dim': 1,            # Distance conditioning

    # Sequence parameters
    'seq_len': 800,                # Max sequence length (storage)
                                   # Dynamic batching will use shorter lengths

    # Network dimensions
    'hidden_dim': 64,              # LSTM hidden size (increased from V3's 32)
    'latent_dim': 48,              # Latent representation (increased from V3's 32)
    'num_layers': 3,               # LSTM layers

    # Generator specific
    'gen_dropout': 0.2,            # Generator dropout rate (regularization)

    # Discriminator specific
    'disc_bilstm_hidden': 128,     # BiLSTM hidden size in discriminator
    'disc_cnn_channels': [64, 128], # CNN channel sizes
    'disc_dropout': 0.4,           # Discriminator dropout rate

    # Condition embedding
    'condition_embed_dim': 64,     # Distance embedding dimension (increased)
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG_V4 = {
    # Batch size
    'batch_size': 64,              # Batch size (adjust based on GPU memory)

    # Learning rates (carefully tuned for 8-feature V4)
    # CRITICAL: These are much lower than typical defaults for stability
    'lr_autoencoder': 1e-3,        # Embedder + Recovery learning rate (Phase 1)
    'lr_supervisor': 5e-4,         # Supervisor learning rate (Phase 2)
    'lr_generator': 3e-5,          # Generator learning rate (Phase 3)
    'lr_discriminator': 5e-5,      # Discriminator learning rate (Phase 3)

    # Phase 3: Embedder/Recovery fine-tuning LR
    # Uses lower LR to prevent degrading Phase 1 learning
    'lr_er_ratio': 0.1,            # E/R LR = lr_autoencoder * lr_er_ratio (default: 10%)
    # Or set directly: 'lr_er': 3e-6

    # Optimizer settings
    'betas': (0.0, 0.9),           # Adam betas (WGAN-GP recommended: beta1=0 for stability)

    # Training phases (iterations, not epochs)
    'phase1_iterations': 8000,     # 8000 Embedder + Recovery (reconstruction)
    'phase2_iterations': 6000,     # 6000 Supervisor (temporal dynamics) - EMBEDDER FROZEN
    'phase2_5_iterations': 5000,   # 5000 Generator pretraining (learns to mimic embedder latent space)
    'phase2_75_iterations': 0,   # Discriminator warmup (learns real vs fake before adversarial)
    'phase3_iterations': 5000,    # 8000 Full adversarial training

    # Validation
    'validate_every': 100,         # Validate every N iterations

    # WGAN-GP parameters
    'lambda_gp': 10.0,             # Gradient penalty weight
    'n_critic': 1,                 # Discriminator updates per generator update

    # Loss weights for generator
    'lambda_recon': 20.0,          # Reconstruction loss weight
    'lambda_supervised': 15.0,      # Supervised loss weight
    'lambda_var': 15.0,             # Variance matching loss weight (feature space: accel, ang_vel)
    'lambda_latent_var': 10.0,      # Latent variance loss weight (preserves Phase 2.5 distribution)
    'lambda_cond': 25.0,            # Condition reconstruction loss weight (path distance matching)

    # V4 Score weighting (validation metric for checkpointing)
    # V4 Score = recon + sup + (cond_weight * cond) + (var_weight * var) + (latent_var_weight * latent_var)
    'v4_score_cond_weight': 2.0,   # Weight for condition fidelity (higher = prioritize distance matching)
    'v4_score_var_weight': 1.0,    # Weight for variance matching (feature space)
    'v4_score_latent_var_weight': 1.0,  # Weight for latent variance matching

    # Gradient clipping
    'max_grad_norm': 10.0,          # Max gradient norm for clipping

    # LR Scheduler settings
    'lr_patience': 10,             # Validation cycles without improvement before LR reduction
                                   # With validate_every=500, this = 10,000 iters before LR drop
    'lr_factor': 0.7,              # Gentler LR reduction (0.7 instead of 0.5)
    'min_lr': 1e-6,                # Minimum learning rate

    # LR Warmup (Phase 3 adversarial training)
    'warmup_steps': 500,           # Linearly increase LR from near-zero over this many steps

    # Early stopping (0 = disabled)
    # Triggers when V4 score doesn't improve for patience * validate_every iterations
    'early_stopping_patience': 25,  # Set to e.g. 20 to enable (20 * 200 = 4000 iters)
    'early_stopping_min_delta': 1e-4,  # Minimum improvement to count as progress

    # EMA (Exponential Moving Average) for stable generation
    # Maintains smoothed weights for inference (0 = disabled)
    'ema_decay': 0.999,            # Higher = smoother (0.99, 0.999, 0.9999 are common)

    # Checkpointing
    'checkpoint_interval': 1000,   # Save checkpoint every N iterations
    'log_interval': 100,           # Log progress every N iterations

    # Phase 2.5: Comprehensive Generator Pretraining
    # These weights control the multi-objective pretraining that prepares
    # the generator for Phase 3 adversarial training
    'lambda_var_pretrain': 8.0,      # Variance matching (prevents mean-collapse)
    'lambda_moment_pretrain': 1.0,   # Higher moments (skewness/kurtosis matching)
    'lambda_cond_pretrain': 15.0,     # Condition learning (CRITICAL - teaches G to use condition)
    'lambda_smooth_pretrain': 0.0,   # Temporal smoothness (coherent sequences)
}


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG_V4 = {
    # Data directories
    'raw_data_dir': 'raw_trajectories_adaptive',
    'processed_data_dir': 'processed_data_v4',
    'checkpoint_dir': 'checkpoints_v4',
    'output_dir': 'generated_results_v4',

    # Preprocessing
    'max_seq_length': 800,
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'random_seed': 42,
}


# ============================================================================
# FEATURE INFORMATION (for reference)
# ============================================================================

FEATURE_INFO_V4 = {
    'feature_dim': 8,
    'feature_names': ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt'],
    'feature_indices': {
        'dx': 0,
        'dy': 1,
        'speed': 2,
        'accel': 3,
        'sin_h': 4,
        'cos_h': 5,
        'ang_vel': 6,
        'dt': 7,
    },
    'feature_descriptions': {
        'dx': 'Relative x position from trajectory start (pixels)',
        'dy': 'Relative y position from trajectory start (pixels)',
        'speed': 'Movement speed magnitude (pixels/second)',
        'accel': 'Acceleration - rate of speed change (pixels/second^2)',
        'sin_h': 'Sine of heading angle (direction, smooth encoding)',
        'cos_h': 'Cosine of heading angle (direction, smooth encoding)',
        'ang_vel': 'Angular velocity - turning rate (radians/second)',
        'dt': 'Time delta between consecutive samples (seconds)',
    },
    'style_features': ['accel', 'ang_vel'],  # Key style signatures
    'position_features': ['dx', 'dy'],
    'motion_features': ['speed', 'sin_h', 'cos_h'],
    'timing_features': ['dt'],
}


# ============================================================================
# STYLE LEARNING CONFIGURATION
# ============================================================================

STYLE_CONFIG_V4 = {
    # Feature-specific weights for reconstruction loss
    # Higher weight = more focus on accurately reproducing that feature
    # Order: [dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt]
    'feature_weights': [1.0, 1.0, 2.0, 5.0, 1.0, 1.0, 5.0, 3.0],
    #                   dx   dy  speed accel sin_h cos_h ang_vel dt
    # Style-critical features (accel, ang_vel) get 5x weight
    # Timing (dt) gets 3x weight - important for rhythm
    # Speed gets 2x weight - core motion characteristic

    # Style feature indices (used for higher moments and autocorrelation)
    'style_feature_indices': [3, 6, 7],  # accel, ang_vel, dt - key style signatures

    # Jerk loss (smoothness) - penalizes erratic acceleration changes
    # jerk = d(acceleration)/dt, high jerk = jerky/unnatural motion
    'lambda_jerk': 1.0,                  # Weight for jerk loss (0 = disabled)

    # Higher moments loss - match skewness and kurtosis of style features
    # Captures asymmetry (skewness) and tail behavior (kurtosis) of distributions
    'lambda_higher_moments': 1.0,        # Weight for higher moments loss (0 = disabled)
    'moment_epsilon': 1e-6,              # Numerical stability for moment computation

    # Autocorrelation loss - match temporal patterns in style features
    # Captures rhythmic/periodic patterns in acceleration, turning, timing
    'lambda_autocorr': 1.5,              # Weight for autocorrelation loss (0 = disabled)
    'autocorr_lags': [1, 2, 5, 10, 20],  # Lag values to compute autocorrelation for
    #                 ^  ^  ^   ^   ^
    #         immediate short medium longer-term pattern matching

    # V4 Score weights for new style losses (used in validation metric)
    'v4_score_jerk_weight': 0.5,         # Weight for jerk in V4 score
    'v4_score_moments_weight': 0.5,      # Weight for higher moments in V4 score
    'v4_score_autocorr_weight': 1.0,     # Weight for autocorrelation in V4 score
}


# ============================================================================
# PHYSICS MONITORING THRESHOLDS
# ============================================================================

PHYSICS_THRESHOLDS_V4 = {
    # Target ratios for generated vs real data (should be ~1.0)
    'speed_ratio_target': 1.0,
    'speed_ratio_tolerance': 0.3,  # Acceptable range: 0.7 - 1.3

    'accel_ratio_target': 1.0,
    'accel_ratio_tolerance': 0.4,  # More tolerance for acceleration

    'ang_vel_ratio_target': 1.0,
    'ang_vel_ratio_tolerance': 0.4,

    'dt_ratio_target': 1.0,
    'dt_ratio_tolerance': 0.3,

    # Warning thresholds
    'extreme_value_threshold': 10.0,  # Warn if any output exceeds this
    'gradient_norm_warning': 10.0,    # Warn if gradient norm exceeds this
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_full_config():
    """Return complete configuration dictionary."""
    return {
        'model': MODEL_CONFIG_V4,
        'training': TRAINING_CONFIG_V4,
        'data': DATA_CONFIG_V4,
        'features': FEATURE_INFO_V4,
        'style': STYLE_CONFIG_V4,
        'physics': PHYSICS_THRESHOLDS_V4,
    }


def print_config():
    """Print configuration summary."""
    print("=" * 70)
    print("V4 CONFIGURATION SUMMARY")
    print("=" * 70)

    print("\nModel Architecture:")
    print(f"  Feature dimension: {MODEL_CONFIG_V4['feature_dim']}")
    print(f"  Hidden dimension:  {MODEL_CONFIG_V4['hidden_dim']}")
    print(f"  Latent dimension:  {MODEL_CONFIG_V4['latent_dim']}")
    print(f"  LSTM layers:       {MODEL_CONFIG_V4['num_layers']}")

    print("\nTraining:")
    print(f"  Batch size:        {TRAINING_CONFIG_V4['batch_size']}")
    print(f"  Phase 1 (recon):   {TRAINING_CONFIG_V4['phase1_iterations']} iterations")
    print(f"  Phase 2 (super):   {TRAINING_CONFIG_V4['phase2_iterations']} iterations")
    print(f"  Phase 3 (adv):     {TRAINING_CONFIG_V4['phase3_iterations']} iterations")

    print("\nFeatures:")
    for i, name in enumerate(FEATURE_INFO_V4['feature_names']):
        desc = FEATURE_INFO_V4['feature_descriptions'][name]
        print(f"  [{i}] {name:8s}: {desc}")

    print("=" * 70)


if __name__ == '__main__':
    print_config()