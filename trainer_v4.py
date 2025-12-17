"""
TimeGAN V4 Trainer - Iteration-Based Training Pipeline

Phase 1: Autoencoder (Embedder + Recovery)
    - Train to reconstruct input from latent space
    - Loss: MSE reconstruction
    - Iterations: 6,000 (default)

Phase 2: Supervised (Supervisor)
    - Train to predict h(t+1) from h(t) in latent space
    - Loss: MSE next-step prediction
    - Iterations: 6,000 (default)

Phase 3: Joint Adversarial
    - Train Generator to fool Discriminator
    - Train Discriminator to distinguish real vs fake
    - Loss: WGAN-GP + Supervised + Reconstruction + Variance + Condition
    - Iterations: 20,000 (default)
    - CRITICAL: 1 iteration = 5 D updates + 1 G update

Key Features:
    - Iteration-based (consistent across dataset sizes)
    - V4 Score for validation-based checkpointing
      * Phase 1 & 2: reconstruction + supervised loss
      * Phase 3: + condition fidelity + variance matching (the real goals!)
    - ReduceLROnPlateau LR scheduling (simple and stable)
    - Variance matching loss for style features
    - Condition reconstruction loss for path distance matching

V4 Features: [dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt]
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timegan_v4 import TimeGANV4


# =============================================================================
# EARLY STOPPING CLASS
# =============================================================================

class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Usage:
        early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        for epoch in range(max_epochs):
            val_loss = train_and_validate()
            if early_stopping(val_loss):
                print("Early stopping triggered!")
                break
    """

    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        """
        Args:
            patience: Number of iterations with no improvement before stopping
            min_delta: Minimum change in metric to qualify as improvement
            mode: 'min' for metrics like loss (lower is better),
                  'max' for metrics like accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.triggered = False

    def __call__(self, value):
        """
        Check if training should stop.

        Args:
            value: Current validation metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.patience == 0:
            # Early stopping disabled
            return False

        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.triggered = False


# =============================================================================
# LINEAR WARMUP CLASS
# =============================================================================

class LinearWarmup:
    """
    Linearly increase LR from near-zero to target over warmup_steps.

    Usage:
        warmup = LinearWarmup(optimizer, warmup_steps=500, target_lr=1e-4)
        for iteration in range(max_iterations):
            # training step...
            still_warming = warmup.step()
            if not still_warming:
                scheduler.step()  # Use regular scheduler after warmup
    """

    def __init__(self, optimizer, warmup_steps, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.current_step = 0

        # Start at 1/warmup_steps of target
        initial_lr = target_lr / max(warmup_steps, 1)
        for pg in optimizer.param_groups:
            pg['lr'] = initial_lr

    def step(self):
        """
        Call once per iteration.
        Returns True if still warming up, False if warmup complete.
        """
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.target_lr * (self.current_step / self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            return True
        return False


# =============================================================================
# EXPONENTIAL MOVING AVERAGE CLASS
# =============================================================================

class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters for stable generation.

    EMA is a common technique in GANs where generation quality can fluctuate
    during training. The EMA weights provide smoother, more stable outputs.

    Usage:
        ema = ExponentialMovingAverage(generator, decay=0.999)
        for iteration in range(max_iterations):
            # training step...
            ema.update()

        # For inference, use EMA weights:
        ema.apply()      # Replace model weights with EMA weights
        # ... generate ...
        ema.restore()    # Restore original training weights
    """

    def __init__(self, model, decay=0.999):
        """
        Args:
            model: PyTorch model to track
            decay: Decay rate for averaging (higher = more weight to historical values)
                   Typical values: 0.99, 0.999, 0.9999
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.enabled = decay > 0

        # Initialize shadow parameters
        if self.enabled:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights with current model parameters."""
        if not self.enabled:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self):
        """Apply EMA weights to the model (for inference)."""
        if not self.enabled:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights after inference."""
        if not self.enabled:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'shadow': {k: v.clone() for k, v in self.shadow.items()},
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = {k: v.clone() for k, v in state_dict['shadow'].items()}


# =============================================================================
# NOTE: All training defaults are now in config_model_v4.py (TRAINING_CONFIG_V4)
# This ensures a single source of truth for all hyperparameters.
# =============================================================================


# =============================================================================
# TIMEGAN V4 TRAINER
# =============================================================================

class TimeGANTrainerV4:
    """
    Iteration-based trainer for TimeGAN V4.

    Training phases:
        1. Autoencoder: Learn latent representation (6000 iterations)
        2. Supervised: Learn temporal dynamics (6000 iterations)
        3. Joint: Adversarial training (20000 iterations)

    Key features:
        - Iteration-based training (consistent across dataset sizes)
        - V4 Score validation metric
        - LR scheduling (warmup + cosine/plateau)
        - Variance matching loss for style features
    """

    def __init__(self, model, config, device='cpu', norm_params=None, tb_logger=None):
        """
        Args:
            model: TimeGANV4 model instance
            config: Training configuration dictionary
            device: 'cpu' or 'cuda'
            norm_params: Normalization parameters (required for condition loss)
            tb_logger: Optional TensorBoardLogger instance for enhanced logging
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.norm_params = norm_params
        self.tb_logger = tb_logger  # Enhanced TensorBoard logger

        # Training hyperparameters (defaults match config_model_v4.py)
        self.lr_ae = config.get('lr_autoencoder', 3e-5)
        self.lr_sup = config.get('lr_supervisor', 3e-5)
        self.lr_g = config.get('lr_generator', 8e-5)
        self.lr_d = config.get('lr_discriminator', 1e-5)

        # Embedder/Recovery fine-tuning LR in Phase 3 (default: 10% of lr_ae)
        self.lr_er_ratio = config.get('lr_er_ratio', 0.1)
        self.lr_er = config.get('lr_er', self.lr_ae * self.lr_er_ratio)

        # Loss weights
        self.lambda_recon = config.get('lambda_recon', 20.0)
        self.lambda_sup = config.get('lambda_supervised', 1.0)
        self.lambda_gp = config.get('lambda_gp', 10.0)
        self.lambda_var = config.get('lambda_var', 5.0)
        self.lambda_cond = config.get('lambda_cond', 1.0)
        
        # Latent variance loss (preserves Phase 2.5 learning in Phase 3)
        # This is CRITICAL - Phase 3 must continue what Phase 2.5 learned
        self.lambda_latent_var = config.get('lambda_latent_var', 10.0)

        # V4 Score weighting (for validation metric)
        self.v4_score_cond_weight = config.get('v4_score_cond_weight', 2.0)
        self.v4_score_var_weight = config.get('v4_score_var_weight', 1.0)
        self.v4_score_latent_var_weight = config.get('v4_score_latent_var_weight', 1.0)

        # WGAN-GP settings
        self.n_critic = config.get('n_critic', 5)

        # Optimizer settings (WGAN-GP recommended: beta1=0 for stability)
        self.betas = config.get('betas', (0.0, 0.9))

        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 5.0)

        # Scheduler settings (ReduceLROnPlateau)
        self.lr_patience = config.get('lr_patience', 10)
        self.lr_factor = config.get('lr_factor', 0.5)
        self.min_lr = config.get('min_lr', 1e-7)

        # Early stopping settings
        self.early_stopping_patience = config.get('early_stopping_patience', 0)  # 0 = disabled
        self.early_stopping_min_delta = config.get('early_stopping_min_delta', 1e-4)

        # LR warmup settings for Phase 3
        self.warmup_steps = config.get('warmup_steps', 500)

        # EMA settings for generator (0 = disabled)
        self.ema_decay = config.get('ema_decay', 0.999)

        # Style-specific loss parameters
        # Feature weights for reconstruction (order: dx, dy, speed, accel, sin_h, cos_h, ang_vel, dt)
        self.feature_weights = config.get('feature_weights', [1.0, 1.0, 2.0, 5.0, 1.0, 1.0, 5.0, 3.0])
        self.style_feature_indices = config.get('style_feature_indices', [3, 6, 7])  # accel, ang_vel, dt

        # Jerk loss (smoothness)
        self.lambda_jerk = config.get('lambda_jerk', 1.0)

        # Higher moments loss (skewness/kurtosis matching)
        self.lambda_higher_moments = config.get('lambda_higher_moments', 1.0)
        self.moment_epsilon = config.get('moment_epsilon', 1e-6)

        # Autocorrelation loss
        self.lambda_autocorr = config.get('lambda_autocorr', 1.5)
        self.autocorr_lags = config.get('autocorr_lags', [1, 2, 5, 10, 20])

        # V4 Score weights for style losses
        self.v4_score_jerk_weight = config.get('v4_score_jerk_weight', 0.5)
        self.v4_score_moments_weight = config.get('v4_score_moments_weight', 0.5)
        self.v4_score_autocorr_weight = config.get('v4_score_autocorr_weight', 1.0)

        # Phase 2.5 comprehensive pretraining loss weights
        # These control how strongly each objective is weighted during generator pretraining
        self.lambda_var_pretrain = config.get('lambda_var_pretrain', 8.0)      # Variance matching
        self.lambda_moment_pretrain = config.get('lambda_moment_pretrain', 1.0)  # Higher moments
        self.lambda_cond_pretrain = config.get('lambda_cond_pretrain', 15.0)    # Condition learning (CRITICAL)
        self.lambda_smooth_pretrain = config.get('lambda_smooth_pretrain', 0.0)  # Temporal smoothness

        # Initialize optimizers and schedulers
        self._init_optimizers()

        # Initialize EMA for generator (for stable generation)
        self.ema_generator = ExponentialMovingAverage(self.model.generator, decay=self.ema_decay)
        self.ema_supervisor = ExponentialMovingAverage(self.model.supervisor, decay=self.ema_decay)
        self.ema_recovery = ExponentialMovingAverage(self.model.recovery, decay=self.ema_decay)

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')

        # Logging
        self.writer = None
        self.global_step = 0

        # Tracking for final metrics (used by tb_logger.log_final_metrics)
        self.best_v4_score = float('inf')
        self.last_val_recon = 0.0
        self.last_val_var = 0.0
        self.last_val_cond = 0.0
        self.last_val_sup = 0.0

        # Feature names for logging
        self.feature_names = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']

    def _init_optimizers(self):
        """Initialize optimizers and learning rate schedulers."""

        # =====================================================================
        # OPTIMIZERS
        # =====================================================================

        # Phase 1: Autoencoder (Embedder + Recovery)
        ae_params = list(self.model.embedder.parameters()) + \
                    list(self.model.recovery.parameters())
        self.optimizer_ae = optim.Adam(ae_params, lr=self.lr_ae, betas=self.betas)

        # Phase 2: Supervisor ONLY (embedder is FROZEN to preserve Phase 1 learning)
        sup_params = list(self.model.supervisor.parameters())
        self.optimizer_sup = optim.Adam(sup_params, lr=self.lr_sup, betas=self.betas)

        # Phase 3: Generator (generator + supervisor)
        g_params = list(self.model.generator.parameters()) + \
                   list(self.model.supervisor.parameters())
        self.optimizer_g = optim.Adam(g_params, lr=self.lr_g, betas=self.betas)

        # Phase 3: Discriminator
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.lr_d,
            betas=self.betas
        )

        # Phase 3: Embedder + Recovery (joint fine-tuning)
        # Uses configurable lr_er (default: 10% of lr_ae) to prevent degradation
        er_params = list(self.model.embedder.parameters()) + \
                    list(self.model.recovery.parameters())
        self.optimizer_er = optim.Adam(er_params, lr=self.lr_er, betas=self.betas)

        # =====================================================================
        # LEARNING RATE SCHEDULERS
        # =====================================================================

        # Phase 1: ReduceLROnPlateau (steps on validation loss)
        self.scheduler_ae = ReduceLROnPlateau(
            self.optimizer_ae,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr
        )

        # Phase 2: ReduceLROnPlateau
        self.scheduler_sup = ReduceLROnPlateau(
            self.optimizer_sup,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr
        )

        # Phase 3: ReduceLROnPlateau for G (simpler than cosine restarts)
        self.scheduler_g = ReduceLROnPlateau(
            self.optimizer_g,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr
        )

        # Phase 3: ReduceLROnPlateau for D
        self.scheduler_d = ReduceLROnPlateau(
            self.optimizer_d,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr
        )

        # Phase 3: ReduceLROnPlateau for Embedder/Recovery fine-tuning
        # Uses proportionally lower min_lr based on lr_er_ratio
        self.scheduler_er = ReduceLROnPlateau(
            self.optimizer_er,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr * self.lr_er_ratio
        )

    # =========================================================================
    # LOGGING HELPER METHODS
    # =========================================================================

    def _log_scalar(self, tag, value, step=None):
        """Log a scalar using either tb_logger or legacy writer."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_scalar(tag, value, step)
        elif self.writer:
            self.writer.add_scalar(tag, value, step)

    def _log_scalars(self, prefix, scalars, step=None):
        """Log multiple scalars with a prefix."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_scalars(prefix, scalars, step)
        elif self.writer:
            for name, value in scalars.items():
                if value is not None:
                    self.writer.add_scalar(f'{prefix}/{name}', value, step)

    def _log_histograms(self, prefix, step=None):
        """Log weight/gradient histograms for all model components."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_histograms(prefix, self.model, step)

    def _log_latent_distributions(self, h_real, h_fake=None, step=None):
        """Log latent space distributions."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_latent_distributions(h_real, h_fake, step)

    def _log_trajectory_comparison(self, real_traj, fake_traj, lengths=None, step=None, tag='Trajectories/comparison'):
        """Log trajectory comparison visualizations."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_trajectory_comparison(real_traj, fake_traj, lengths=lengths, step=step, tag=tag)

    def _log_distribution_comparison(self, real_data, fake_data, step=None):
        """Log feature distribution comparison plots."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_distribution_comparison(real_data, fake_data, step)

    def _log_feature_reconstruction_errors(self, real, reconstructed, step=None):
        """Log per-feature reconstruction errors."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_feature_reconstruction_errors(
                real, reconstructed, self.feature_names, step
            )

    def _log_gradient_norms(self, step=None):
        """Log gradient norms for all model components."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            return self.tb_logger.log_gradient_norms(self.model, step)
        return {}

    def _start_phase(self, phase_name):
        """Mark the start of a training phase."""
        if self.tb_logger:
            self.tb_logger.start_phase(phase_name)

    def _increment_global_step(self, n=1):
        """Increment global step counter."""
        self.global_step += n
        if self.tb_logger:
            self.tb_logger.increment_step(n)

    def _log_embeddings(self, h_real, h_fake=None, conditions=None, step=None):
        """Log embeddings for Projector visualization (t-SNE/PCA/UMAP)."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_embeddings(h_real, h_fake, conditions, step)

    def _log_trajectory_paths(self, real_traj, fake_traj, conditions=None, lengths=None, step=None):
        """Log enhanced trajectory path visualization."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_trajectory_paths(real_traj, fake_traj, conditions, lengths, step)

    def _log_speed_heatmap(self, real_traj, fake_traj, step=None):
        """Log trajectory paths colored by speed."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_speed_heatmap(real_traj, fake_traj, step)

    def _log_discriminator_scores(self, d_real, d_fake, step=None):
        """Log discriminator PR curve and score distribution."""
        step = step if step is not None else self.global_step
        if self.tb_logger:
            self.tb_logger.log_discriminator_pr_curve(d_real, d_fake, step)
            self.tb_logger.log_discriminator_histogram(d_real, d_fake, step)

    def _log_model_graph(self):
        """Log model component graphs for visualization."""
        if self.tb_logger:
            self.tb_logger.log_component_graphs(
                self.model,
                self.device,
                batch_size=2,
                seq_len=100,
                feature_dim=8,
                latent_dim=self.model.latent_dim
            )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _create_mask(self, lengths, seq_len):
        """
        Create mask for variable-length sequences.

        Args:
            lengths: (batch_size,) tensor of sequence lengths
            seq_len: Maximum sequence length

        Returns:
            mask: (batch_size, seq_len, 1) binary mask
        """
        batch_size = len(lengths)
        mask = torch.zeros(batch_size, seq_len, device=self.device)

        # Ensure lengths is a tensor on the correct device
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, device=self.device)

        # Clamp lengths to valid range [0, seq_len] to handle edge cases
        lengths_clamped = torch.clamp(lengths, min=0, max=seq_len)

        for i, length in enumerate(lengths_clamped):
            length_int = int(length.item()) if isinstance(length, torch.Tensor) else int(length)
            if length_int > 0:
                mask[i, :length_int] = 1.0

        return mask.unsqueeze(-1)  # (batch, seq_len, 1)

    def _masked_mse(self, pred, target, mask, use_feature_weights=False):
        """
        Compute MSE loss with masking for padded sequences.

        Args:
            pred: (batch, seq_len, feature_dim) predictions
            target: (batch, seq_len, feature_dim) targets
            mask: (batch, seq_len, 1) binary mask for valid timesteps
            use_feature_weights: If True, apply per-feature weighting

        Returns:
            Weighted mean squared error
        """
        loss = self.mse_loss(pred, target)  # (batch, seq_len, feature_dim)
        loss = loss * mask  # Apply sequence mask

        if use_feature_weights and len(self.feature_weights) == pred.size(-1):
            # Apply per-feature weights
            # Convert to tensor on correct device
            weights = torch.tensor(self.feature_weights, device=self.device, dtype=pred.dtype)
            # Normalize weights so they sum to feature_dim (preserves loss scale)
            weights = weights / weights.mean()
            # Apply weights: (batch, seq_len, feature_dim) * (feature_dim,)
            loss = loss * weights

        n_elements = mask.sum() * pred.size(-1)
        if n_elements > 0:
            return loss.sum() / n_elements
        return torch.tensor(0.0, device=self.device)

    def _compute_supervised_loss(self, h, h_supervised, lengths):
        """
        Compute supervised loss for temporal dynamics prediction.

        The supervisor predicts h(t+1) from h(t). This function computes
        the MSE between predictions and targets with proper masking.

        Args:
            h: (batch, seq_len, latent_dim) - original latent codes
            h_supervised: (batch, seq_len, latent_dim) - supervisor predictions
            lengths: (batch,) - sequence lengths

        Returns:
            loss: Scalar supervised loss
        """
        # Handle sequences that are too short for supervised learning
        # Need at least 2 timesteps to predict h(t+1) from h(t)
        min_length = lengths.min().item() if isinstance(lengths, torch.Tensor) else min(lengths)
        if min_length < 2:
            # For very short sequences, return zero loss
            return torch.tensor(0.0, device=self.device)

        h_target = h[:, 1:, :]           # Target: timesteps 1 to T
        h_pred = h_supervised[:, :-1, :] # Prediction: from timesteps 0 to T-1

        # Ensure lengths - 1 is at least 1 for valid sequences
        adjusted_lengths = torch.clamp(lengths - 1, min=1)
        mask = self._create_mask(adjusted_lengths, h_target.size(1))

        return self._masked_mse(h_pred, h_target.detach(), mask)

    def _gradient_penalty(self, real, fake, condition, lengths):
        """
        Compute gradient penalty for WGAN-GP.
        CuDNN disabled for double backwards support.
        """
        batch_size = real.size(0)

        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)

        with torch.backends.cudnn.flags(enabled=False):
            d_interpolated = self.model.discriminator(interpolated, condition, lengths)

            gradients = torch.autograd.grad(
                outputs=d_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones_like(d_interpolated),
                create_graph=True,
                retain_graph=True,
            )[0]

        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    # =========================================================================
    # VARIANCE MATCHING LOSS
    # =========================================================================

    def compute_variance_loss(self, x_real, x_fake, mask):
        """
        Penalize when generated variance doesn't match real data.

        Targets style-critical features:
            - Index 3: acceleration (px/s^2)
            - Index 6: angular velocity (rad/s)

        This prevents generated outputs from being too smooth/uniform.
        """
        style_indices = [3, 6]  # accel, ang_vel
        total_var_loss = 0.0

        for idx in style_indices:
            # Extract feature and apply mask
            real_feat = x_real[:, :, idx] * mask.squeeze(-1)
            fake_feat = x_fake[:, :, idx] * mask.squeeze(-1)

            # Compute variance along sequence dimension for each sample
            real_var = real_feat.var(dim=1).mean()
            fake_var = fake_feat.var(dim=1).mean()

            # Penalize variance mismatch
            total_var_loss += (real_var - fake_var).abs()

        return total_var_loss

    # =========================================================================
    # LATENT VARIANCE LOSS (Preserves Phase 2.5 Learning)
    # =========================================================================

    def compute_latent_variance_loss(self, h_real, h_fake, mask):
        """
        Match latent distribution: variance, per-sample variance, range, and std_ratio.

        This PRESERVES what Phase 2.5 learned!
        
        Phase 2.5 carefully trained the generator to match:
        - Global variance (distribution spread)
        - Per-sample variance (prevents identical outputs)
        - Range coverage (uses full latent space)
        - Std ratio ≈ 1.0 (most important)
        
        Without this in Phase 3, the adversarial training can "unlearn"
        the variance matching, causing mode collapse.

        Args:
            h_real: (batch, seq_len, latent_dim) - real embeddings from embedder
            h_fake: (batch, seq_len, latent_dim) - generator outputs (after supervisor)
            mask: (batch, seq_len, 1) - validity mask

        Returns:
            latent_var_loss: Combined variance loss
            std_ratio: σ_fake / σ_real (for monitoring, should be ~1.0)
        """
        n_valid = mask.sum()
        if n_valid <= 1:
            return torch.tensor(0.0, device=self.device), 1.0

        # Apply mask
        h_real_masked = h_real * mask
        h_fake_masked = h_fake * mask

        # =====================================================================
        # Global variance (across batch and sequence)
        # =====================================================================
        h_real_mean = h_real_masked.sum(dim=(0, 1)) / n_valid
        h_fake_mean = h_fake_masked.sum(dim=(0, 1)) / n_valid

        # Centered values
        h_real_centered = (h_real_masked - h_real_mean.unsqueeze(0).unsqueeze(0)) * mask
        h_fake_centered = (h_fake_masked - h_fake_mean.unsqueeze(0).unsqueeze(0)) * mask

        # Variance (2nd central moment)
        h_real_var = (h_real_centered ** 2).sum(dim=(0, 1)) / n_valid
        h_fake_var = (h_fake_centered ** 2).sum(dim=(0, 1)) / n_valid

        # Use std (sqrt of var) for more stable gradients
        h_real_std = torch.sqrt(h_real_var + 1e-8)
        h_fake_std = torch.sqrt(h_fake_var + 1e-8)

        # Variance loss: match standard deviations per latent dimension
        global_var_loss = (h_real_std - h_fake_std).abs().mean()

        # =====================================================================
        # Per-sample variance (prevents all samples being identical)
        # =====================================================================
        mask_sum_per_sample = mask.sum(dim=1, keepdim=True) + 1e-8  # (batch, 1, 1)

        per_sample_mean_real = h_real_masked.sum(dim=1, keepdim=True) / mask_sum_per_sample
        per_sample_mean_fake = h_fake_masked.sum(dim=1, keepdim=True) / mask_sum_per_sample

        mask_sum_squeezed = mask_sum_per_sample.squeeze(1)  # (batch, 1)

        per_sample_var_real = ((h_real_masked - per_sample_mean_real) ** 2 * mask).sum(dim=1) / mask_sum_squeezed
        per_sample_var_fake = ((h_fake_masked - per_sample_mean_fake) ** 2 * mask).sum(dim=1) / mask_sum_squeezed

        per_sample_var_loss = (per_sample_var_real.mean(dim=0) - per_sample_var_fake.mean(dim=0)).abs().mean()

        # =====================================================================
        # Range loss (ensures generator covers full range of real data)
        # =====================================================================
        h_real_range = h_real_masked.max() - h_real_masked.min()
        h_fake_range = h_fake_masked.max() - h_fake_masked.min()
        range_loss = torch.abs(h_real_range - h_fake_range)

        # =====================================================================
        # Std ratio loss (MOST IMPORTANT - keeps σ_ratio near 1.0)
        # =====================================================================
        mean_real_std = h_real_std.mean()
        mean_fake_std = h_fake_std.mean()
        std_ratio = mean_fake_std / (mean_real_std + 1e-8)
        # Penalize deviation from 1.0 (log makes it symmetric)
        std_ratio_loss = (torch.log(std_ratio + 1e-8)).abs()

        # Combined latent variance loss
        latent_var_loss = (global_var_loss + 
                          0.5 * per_sample_var_loss + 
                          0.2 * range_loss + 
                          10.0 * std_ratio_loss)

        return latent_var_loss, std_ratio.item()

    # =========================================================================
    # CONDITION RECONSTRUCTION LOSS (Path Distance Matching)
    # =========================================================================

    def compute_condition_loss(self, x_fake, condition, lengths):
        """
        Penalize when generated trajectory's path distance doesn't match condition.

        This is CRITICAL for condition-matched generation:
        - Condition tells generator "make a path of X pixels"
        - Generator outputs dx, dy
        - We compute actual path distance from dx, dy
        - Penalize mismatch between actual and expected distance

        The loss is computed in NORMALIZED space to avoid scale issues:
        - Compute path distance from normalized dx, dy
        - Normalize the result to [-1, 1] using same params as condition
        - Compare to input condition

        This implementation is FULLY VECTORIZED for GPU efficiency.

        Args:
            x_fake: (batch, seq_len, 8) generated trajectories (normalized [-1, 1])
            condition: (batch, 1) distance conditions (normalized [-1, 1])
            lengths: (batch,) sequence lengths

        Returns:
            cond_loss: Mean absolute error between generated and target distance
        """
        if self.norm_params is None:
            # Can't compute without normalization params
            return torch.tensor(0.0, device=self.device)

        batch_size = x_fake.size(0)
        seq_len = x_fake.size(1)

        # Get normalization params
        dx_min = self.norm_params['dx_min']
        dx_max = self.norm_params['dx_max']
        dy_min = self.norm_params['dy_min']
        dy_max = self.norm_params['dy_max']
        dist_min = self.norm_params['actual_dist_min']
        dist_max = self.norm_params['actual_dist_max']

        # Denormalize dx, dy from [-1, 1] to pixel space
        # Formula: original = ((normalized + 1) / 2) * (max - min) + min
        dx_norm = x_fake[:, :, 0]  # (batch, seq_len)
        dy_norm = x_fake[:, :, 1]

        dx_pixels = ((dx_norm + 1) / 2) * (dx_max - dx_min) + dx_min
        dy_pixels = ((dy_norm + 1) / 2) * (dy_max - dy_min) + dy_min

        # Compute point-to-point differences (vectorized)
        # Shape: (batch, seq_len - 1)
        dx_diff = dx_pixels[:, 1:] - dx_pixels[:, :-1]
        dy_diff = dy_pixels[:, 1:] - dy_pixels[:, :-1]

        # Euclidean step distances: (batch, seq_len - 1)
        step_distances = torch.sqrt(dx_diff**2 + dy_diff**2 + 1e-8)

        # Create mask for valid step distances
        # A step at position i is valid if i+1 < length (i.e., both points exist)
        # For lengths tensor, valid steps are 0 to length-2
        # Create mask: (batch, seq_len - 1)
        step_indices = torch.arange(seq_len - 1, device=self.device).unsqueeze(0)  # (1, seq_len-1)
        lengths_expanded = lengths.unsqueeze(1)  # (batch, 1)
        # Step i is valid if i < length - 1 (both i and i+1 must be valid)
        step_mask = (step_indices < (lengths_expanded - 1)).float()  # (batch, seq_len-1)

        # Apply mask and sum to get path distance for each sample
        # Masked step distances
        masked_step_distances = step_distances * step_mask  # (batch, seq_len-1)
        generated_distances = masked_step_distances.sum(dim=1)  # (batch,)

        # Handle samples with length < 2 (no valid steps)
        # These will naturally have distance 0 from the masking above

        # Normalize generated distances to [-1, 1] using same params as condition
        generated_norm = 2 * ((generated_distances - dist_min) / (dist_max - dist_min + 1e-8)) - 1

        # Clamp to reasonable range (allow some extrapolation but prevent extreme values)
        # Using [-1.5, 1.5] allows 50% extrapolation beyond training data range
        generated_norm = torch.clamp(generated_norm, -1.5, 1.5)

        # Target is the input condition
        target_norm = condition.squeeze(-1)  # (batch,)

        # Mean absolute error in normalized space
        cond_loss = (generated_norm - target_norm).abs().mean()

        return cond_loss

    # =========================================================================
    # JERK LOSS (Smoothness)
    # =========================================================================

    def compute_jerk_loss(self, x_real, x_fake, mask):
        """
        Compute jerk loss - penalizes difference in acceleration smoothness.

        Jerk is the derivative of acceleration (d(accel)/dt).
        High jerk = jerky, unnatural motion.
        This loss encourages generated trajectories to have similar smoothness
        characteristics as real trajectories.

        Args:
            x_real: (batch, seq_len, 8) real trajectories
            x_fake: (batch, seq_len, 8) generated trajectories
            mask: (batch, seq_len, 1) validity mask

        Returns:
            jerk_loss: Scalar loss value
        """
        if self.lambda_jerk == 0:
            return torch.tensor(0.0, device=self.device)

        # Extract acceleration feature (index 3)
        accel_real = x_real[:, :, 3]  # (batch, seq_len)
        accel_fake = x_fake[:, :, 3]

        # Compute jerk (first difference of acceleration)
        # jerk[t] = accel[t+1] - accel[t]
        jerk_real = accel_real[:, 1:] - accel_real[:, :-1]  # (batch, seq_len-1)
        jerk_fake = accel_fake[:, 1:] - accel_fake[:, :-1]

        # Create mask for jerk (valid if both timesteps are valid)
        jerk_mask = mask[:, :-1, 0] * mask[:, 1:, 0]  # (batch, seq_len-1)

        # Compute mean absolute jerk for real and fake
        real_jerk_masked = jerk_real.abs() * jerk_mask
        fake_jerk_masked = jerk_fake.abs() * jerk_mask

        n_valid = jerk_mask.sum() + 1e-8

        mean_jerk_real = real_jerk_masked.sum() / n_valid
        mean_jerk_fake = fake_jerk_masked.sum() / n_valid

        # Also compare jerk variance (captures how consistent the smoothness is)
        var_jerk_real = ((jerk_real ** 2) * jerk_mask).sum() / n_valid - mean_jerk_real ** 2
        var_jerk_fake = ((jerk_fake ** 2) * jerk_mask).sum() / n_valid - mean_jerk_fake ** 2

        # Loss: match both mean jerk and jerk variance
        jerk_loss = (mean_jerk_real - mean_jerk_fake).abs() + \
                    (var_jerk_real - var_jerk_fake).abs()

        return jerk_loss

    # =========================================================================
    # HIGHER MOMENTS LOSS (Skewness/Kurtosis)
    # =========================================================================

    def compute_higher_moments_loss(self, x_real, x_fake, mask):
        """
        Compute higher moments loss - match skewness and kurtosis of style features.

        Skewness captures asymmetry of the distribution (e.g., more sharp accels
        vs gradual decels). Kurtosis captures tail behavior (frequency of extreme
        values vs. moderate values).

        These are crucial for capturing the "feel" of a user's mouse movement style.

        Args:
            x_real: (batch, seq_len, 8) real trajectories
            x_fake: (batch, seq_len, 8) generated trajectories
            mask: (batch, seq_len, 1) validity mask

        Returns:
            moments_loss: Scalar loss value
        """
        if self.lambda_higher_moments == 0:
            return torch.tensor(0.0, device=self.device)

        eps = self.moment_epsilon
        total_loss = torch.tensor(0.0, device=self.device)

        for idx in self.style_feature_indices:
            # Extract feature
            feat_real = x_real[:, :, idx]  # (batch, seq_len)
            feat_fake = x_fake[:, :, idx]

            # Apply mask
            mask_2d = mask.squeeze(-1)  # (batch, seq_len)
            n_valid = mask_2d.sum() + eps

            # Compute masked statistics for real
            mean_real = (feat_real * mask_2d).sum() / n_valid
            centered_real = (feat_real - mean_real) * mask_2d
            var_real = (centered_real ** 2).sum() / n_valid + eps
            std_real = torch.sqrt(var_real)

            # Standardized values
            z_real = centered_real / (std_real + eps)

            # Skewness: E[(X - mu)^3 / sigma^3]
            skew_real = ((z_real ** 3) * mask_2d).sum() / n_valid

            # Kurtosis: E[(X - mu)â´ / sigmaâ´] - 3 (excess kurtosis)
            kurt_real = ((z_real ** 4) * mask_2d).sum() / n_valid - 3.0

            # Compute masked statistics for fake
            mean_fake = (feat_fake * mask_2d).sum() / n_valid
            centered_fake = (feat_fake - mean_fake) * mask_2d
            var_fake = (centered_fake ** 2).sum() / n_valid + eps
            std_fake = torch.sqrt(var_fake)

            z_fake = centered_fake / (std_fake + eps)
            skew_fake = ((z_fake ** 3) * mask_2d).sum() / n_valid
            kurt_fake = ((z_fake ** 4) * mask_2d).sum() / n_valid - 3.0

            # Loss: match both skewness and kurtosis
            total_loss = total_loss + (skew_real - skew_fake).abs() + \
                         0.5 * (kurt_real - kurt_fake).abs()  # Kurtosis weighted less (more noisy)

        # Average over style features
        total_loss = total_loss / max(len(self.style_feature_indices), 1)

        return total_loss

    # =========================================================================
    # AUTOCORRELATION LOSS
    # =========================================================================

    def _compute_autocorr(self, x, mask, lag):
        """
        Compute autocorrelation at a specific lag for masked sequence.

        Autocorrelation measures how correlated a signal is with itself at
        different time lags. This captures rhythmic/periodic patterns.

        Args:
            x: (batch, seq_len) feature values
            mask: (batch, seq_len) validity mask
            lag: Time lag to compute autocorrelation for

        Returns:
            autocorr: Scalar autocorrelation value
        """
        seq_len = x.size(1)
        if lag >= seq_len - 1:
            return torch.tensor(0.0, device=self.device)

        # Get overlapping regions
        x_t = x[:, :-lag]      # (batch, seq_len - lag)
        x_lag = x[:, lag:]     # (batch, seq_len - lag)

        # Mask for valid pairs (both timesteps must be valid)
        mask_t = mask[:, :-lag]
        mask_lag = mask[:, lag:]
        pair_mask = mask_t * mask_lag

        n_pairs = pair_mask.sum() + 1e-8

        # Compute means
        mean_t = (x_t * pair_mask).sum() / n_pairs
        mean_lag = (x_lag * pair_mask).sum() / n_pairs

        # Compute covariance and variances
        cov = ((x_t - mean_t) * (x_lag - mean_lag) * pair_mask).sum() / n_pairs
        var_t = ((x_t - mean_t) ** 2 * pair_mask).sum() / n_pairs + 1e-8
        var_lag = ((x_lag - mean_lag) ** 2 * pair_mask).sum() / n_pairs + 1e-8

        # Correlation coefficient
        autocorr = cov / (torch.sqrt(var_t) * torch.sqrt(var_lag) + 1e-8)

        return autocorr

    def compute_autocorr_loss(self, x_real, x_fake, mask):
        """
        Compute autocorrelation loss - match temporal patterns in style features.

        This captures rhythmic patterns in movement:
        - Lag 1-2: Immediate temporal dependencies
        - Lag 5-10: Short-term patterns
        - Lag 20+: Longer-term rhythmic patterns

        Args:
            x_real: (batch, seq_len, 8) real trajectories
            x_fake: (batch, seq_len, 8) generated trajectories
            mask: (batch, seq_len, 1) validity mask

        Returns:
            autocorr_loss: Scalar loss value
        """
        if self.lambda_autocorr == 0:
            return torch.tensor(0.0, device=self.device)

        mask_2d = mask.squeeze(-1)  # (batch, seq_len)
        total_loss = torch.tensor(0.0, device=self.device)
        n_terms = 0

        for idx in self.style_feature_indices:
            feat_real = x_real[:, :, idx]
            feat_fake = x_fake[:, :, idx]

            for lag in self.autocorr_lags:
                autocorr_real = self._compute_autocorr(feat_real, mask_2d, lag)
                autocorr_fake = self._compute_autocorr(feat_fake, mask_2d, lag)

                # Match autocorrelation at each lag
                total_loss = total_loss + (autocorr_real - autocorr_fake).abs()
                n_terms += 1

        # Average over all terms
        if n_terms > 0:
            total_loss = total_loss / n_terms

        return total_loss

    # =========================================================================
    # V4 SCORE COMPUTATION (Validation Metric)
    # =========================================================================

    def compute_v4_score(self, val_dataloader):
        """
        Compute validation metric for checkpointing.

        V4 Score = reconstruction_loss + supervised_loss

        Lower is better. Used to save best model checkpoint.

        Returns:
            v4_score: Combined validation metric
            avg_recon: Average reconstruction loss
            avg_sup: Average supervised loss
        """
        self.model.eval()
        total_recon = 0.0
        total_sup = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, condition, lengths in val_dataloader:
                x = x.to(self.device)
                lengths = lengths.to(self.device)

                # Reconstruction loss
                h = self.model.embedder(x, lengths)
                x_recon = self.model.recovery(h, lengths)
                mask = self._create_mask(lengths, x.size(1))
                recon_loss = self._masked_mse(x_recon, x, mask)

                # Supervised loss (using helper for edge case handling)
                h_sup = self.model.supervisor(h, lengths)
                sup_loss = self._compute_supervised_loss(h, h_sup, lengths)

                total_recon += recon_loss.item()
                total_sup += sup_loss.item()
                n_batches += 1

        self.model.train()

        avg_recon = total_recon / max(n_batches, 1)
        avg_sup = total_sup / max(n_batches, 1)
        v4_score = avg_recon + avg_sup

        return v4_score, avg_recon, avg_sup

    def compute_v4_score_phase3(self, val_dataloader):
        """
        Compute validation metric for Phase 3 checkpointing.

        V4 Score Phase 3 = reconstruction + supervised + condition + variance
                         + latent_variance + jerk + higher_moments + autocorrelation

        This is the PROPER metric for Phase 3 because it measures:
        - Autoencoder quality (recon + sup) - baseline from Phase 1 & 2
        - Condition fidelity - does generator respect distance condition?
        - Variance matching (feature space) - does generated data have realistic variance?
        - Latent variance matching - does generator preserve Phase 2.5 distribution?
        - Jerk matching - does generated have realistic smoothness?
        - Higher moments - does generated match skewness/kurtosis?
        - Autocorrelation - does generated match temporal patterns?

        Lower is better. Used to save best model and adjust LR in Phase 3.

        Returns:
            v4_score: Combined validation metric
            avg_recon: Average reconstruction loss
            avg_sup: Average supervised loss
            avg_cond: Average condition fidelity loss
            avg_var: Average variance matching loss (feature space)
            avg_jerk: Average jerk loss
            avg_moments: Average higher moments loss
            avg_autocorr: Average autocorrelation loss
            avg_latent_var: Average latent variance loss
            avg_std_ratio: Average std ratio (should be ~1.0)
        """
        self.model.eval()
        total_recon = 0.0
        total_sup = 0.0
        total_cond = 0.0
        total_var = 0.0
        total_latent_var = 0.0
        total_std_ratio = 0.0
        total_jerk = 0.0
        total_moments = 0.0
        total_autocorr = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, condition, lengths in val_dataloader:
                x = x.to(self.device)
                condition = condition.to(self.device)
                lengths = lengths.to(self.device)

                seq_len = x.size(1)
                batch_size = x.size(0)
                mask = self._create_mask(lengths, seq_len)

                # Reconstruction loss (autoencoder quality)
                h_real = self.model.embedder(x, lengths)
                x_recon = self.model.recovery(h_real, lengths)
                recon_loss = self._masked_mse(x_recon, x, mask)

                # Supervised loss (temporal dynamics, using helper for edge case handling)
                h_sup = self.model.supervisor(h_real, lengths)
                sup_loss = self._compute_supervised_loss(h_real, h_sup, lengths)

                # Generate fake data for condition and variance metrics (with supervisor)
                z = torch.randn(batch_size, seq_len, self.model.latent_dim, device=self.device)
                h_fake_raw = self.model.generator(z, condition, lengths)
                h_fake = self.model.supervisor(h_fake_raw, lengths)
                x_fake = self.model.recovery(h_fake, lengths)

                # Condition fidelity loss (does generated path distance match condition?)
                cond_loss = self.compute_condition_loss(x_fake, condition, lengths)

                # Variance matching loss - FEATURE SPACE (accel, ang_vel)
                var_loss = self.compute_variance_loss(x, x_fake, mask)

                # Latent variance loss - LATENT SPACE (preserves Phase 2.5 learning!)
                # CRITICAL: Use h_fake_raw (generator output), NOT h_fake (post-supervisor)
                # Must match Phase 2.5's formulation exactly for curriculum continuity
                latent_var_loss, std_ratio = self.compute_latent_variance_loss(h_real, h_fake_raw, mask)

                # Style-specific losses
                jerk_loss = self.compute_jerk_loss(x, x_fake, mask)
                moments_loss = self.compute_higher_moments_loss(x, x_fake, mask)
                autocorr_loss = self.compute_autocorr_loss(x, x_fake, mask)

                total_recon += recon_loss.item()
                total_sup += sup_loss.item()
                total_cond += cond_loss.item()
                total_var += var_loss.item()
                total_latent_var += latent_var_loss.item()
                total_std_ratio += std_ratio
                total_jerk += jerk_loss.item()
                total_moments += moments_loss.item()
                total_autocorr += autocorr_loss.item()
                n_batches += 1

        self.model.train()

        avg_recon = total_recon / max(n_batches, 1)
        avg_sup = total_sup / max(n_batches, 1)
        avg_cond = total_cond / max(n_batches, 1)
        avg_var = total_var / max(n_batches, 1)
        avg_latent_var = total_latent_var / max(n_batches, 1)
        avg_std_ratio = total_std_ratio / max(n_batches, 1)
        avg_jerk = total_jerk / max(n_batches, 1)
        avg_moments = total_moments / max(n_batches, 1)
        avg_autocorr = total_autocorr / max(n_batches, 1)

        # Combined score: all components matter for Phase 3
        # Weights are configurable
        v4_score = (avg_recon + avg_sup +
                    self.v4_score_cond_weight * avg_cond +
                    self.v4_score_var_weight * avg_var +
                    self.v4_score_latent_var_weight * avg_latent_var +
                    self.v4_score_jerk_weight * avg_jerk +
                    self.v4_score_moments_weight * avg_moments +
                    self.v4_score_autocorr_weight * avg_autocorr)

        return (v4_score, avg_recon, avg_sup, avg_cond, avg_var, 
                avg_jerk, avg_moments, avg_autocorr, avg_latent_var, avg_std_ratio)

    # =========================================================================
    # PHASE 1: AUTOENCODER TRAINING (Iteration-Based)
    # =========================================================================

    def train_autoencoder_step(self, x, lengths):
        """
        Single autoencoder training step with style-aware reconstruction.

        Loss = feature-weighted_MSE + jerk_preservation + moments_preservation + autocorr_preservation

        The style preservation losses ensure the autoencoder learns to preserve
        style-critical features (acceleration patterns, turning behavior, timing rhythm)
        not just raw feature values.
        """
        self.model.train()
        self.optimizer_ae.zero_grad()

        h = self.model.embedder(x, lengths)
        x_recon = self.model.recovery(h, lengths)

        mask = self._create_mask(lengths, x.size(1))

        # 1. Feature-weighted reconstruction loss
        recon_loss = self._masked_mse(x_recon, x, mask, use_feature_weights=True)

        # 2. Style preservation losses (input vs reconstructed)
        # These ensure the autoencoder preserves style characteristics, not just raw values
        jerk_loss = self.compute_jerk_loss(x, x_recon, mask)
        moments_loss = self.compute_higher_moments_loss(x, x_recon, mask)
        autocorr_loss = self.compute_autocorr_loss(x, x_recon, mask)

        # Total loss with style preservation (use same weights as Phase 3, scaled down)
        # Scale down style losses in Phase 1 since reconstruction is primary goal
        # NOTE: Set to 0 if style losses interfere with reconstruction convergence
        style_scale = 0.0  # Disabled - focus on pure reconstruction first
        total_loss = (recon_loss +
                      style_scale * self.lambda_jerk * jerk_loss +
                      style_scale * self.lambda_higher_moments * moments_loss +
                      style_scale * self.lambda_autocorr * autocorr_loss)

        total_loss.backward()

        # Compute gradient norms for diagnostics
        e_grad = torch.nn.utils.clip_grad_norm_(self.model.embedder.parameters(), self.max_grad_norm)
        r_grad = torch.nn.utils.clip_grad_norm_(self.model.recovery.parameters(), self.max_grad_norm)

        self.optimizer_ae.step()

        # Return dict with all loss components for logging
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'jerk_loss': jerk_loss.item(),
            'moments_loss': moments_loss.item(),
            'autocorr_loss': autocorr_loss.item(),
            'h': h,
            'x_recon': x_recon,
            'e_grad': e_grad.item(),
            'r_grad': r_grad.item()
        }

    def train_autoencoder(self, dataloader, val_dataloader, max_iterations,
                          validate_every=500, checkpoint_dir=None):
        """
        Phase 1: Train autoencoder (embedder + recovery).

        Goal: Learn latent representation that can reconstruct input faithfully.
        Success metric: val_recon < 0.01 (good), < 0.001 (excellent)
        """
        print("\n" + "=" * 70)
        print("PHASE 1: AUTOENCODER TRAINING")
        print("=" * 70)
        print(f"Purpose: Train Embedder + Recovery to learn latent representation")
        print(f"         Input -> Embedder -> Latent H -> Recovery -> Reconstructed Input")
        print("-" * 70)
        print(f"Iterations: {max_iterations}")
        print(f"Validate every: {validate_every}")
        print(f"LR_Autoencoder: {self.lr_ae:.2e}")
        print(f"LR Scheduler: ReduceLROnPlateau (patience={self.lr_patience}, factor={self.lr_factor})")
        print("-" * 70)
        print(f"Target: val_recon < 0.01 (good), < 0.001 (excellent)")
        print("=" * 70)

        # Flush stdout before creating DataLoader iterator to prevent
        # interleaved output on Windows when num_workers > 0
        sys.stdout.flush()

        iterator = iter(dataloader)
        best_v4_score = float('inf')
        lr_reductions = 0
        last_lr = self.lr_ae

        for iteration in range(max_iterations):
            # Get next batch (cycle if exhausted)
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            x, condition, lengths = batch
            x = x.to(self.device)
            lengths = lengths.to(self.device)

            # Training step (returns dict with all loss components)
            step_result = self.train_autoencoder_step(x, lengths)
            total_loss = step_result['total_loss']
            recon_loss = step_result['recon_loss']
            jerk_loss = step_result['jerk_loss']
            moments_loss = step_result['moments_loss']
            autocorr_loss = step_result['autocorr_loss']
            h = step_result['h']
            x_recon = step_result['x_recon']
            e_grad = step_result['e_grad']
            r_grad = step_result['r_grad']

            # Validate & checkpoint
            if iteration % validate_every == 0 or iteration == max_iterations - 1:
                v4_score, val_recon, val_sup = self.compute_v4_score(val_dataloader)

                # Scheduler step
                self.scheduler_ae.step(val_recon)

                # Check for LR reduction
                current_lr = self.optimizer_ae.param_groups[0]['lr']
                if current_lr < last_lr:
                    lr_reductions += 1
                    print(f"  [!]  LR reduced: {last_lr:.2e} -> {current_lr:.2e} (reduction #{lr_reductions})")
                    last_lr = current_lr

                # Compute diagnostic stats
                with torch.no_grad():
                    h_mean = h.mean().item()
                    h_std = h.std().item()
                    recon_err_per_feature = ((x_recon - x) ** 2).mean(dim=(0, 1))

                # Progress indicator
                progress = iteration / max_iterations * 100

                print(f"  [Iter {iteration:5d}/{max_iterations}] ({progress:5.1f}%) | "
                      f"Loss: {total_loss:.6f} (R:{recon_loss:.4f} J:{jerk_loss:.4f} M:{moments_loss:.4f} A:{autocorr_loss:.4f}) | "
                      f"Val: {val_recon:.6f} | LR: {current_lr:.2e}")

                # Log scalars (using helper for both tb_logger and legacy writer)
                self._log_scalars('Phase1', {
                    'train_loss': total_loss,
                    'recon_loss': recon_loss,
                    'jerk_loss': jerk_loss,
                    'moments_loss': moments_loss,
                    'autocorr_loss': autocorr_loss,
                    'val_recon': val_recon,
                    'v4_score': v4_score,
                    'lr': current_lr,
                    'grad_embedder': e_grad,
                    'grad_recovery': r_grad,
                    'h_mean': h_mean,
                    'h_std': h_std,
                }, step=self.global_step)

                # Log per-feature reconstruction errors (every validation step)
                self._log_scalars('Features', {
                    'recon_dx': recon_err_per_feature[0].item(),
                    'recon_dy': recon_err_per_feature[1].item(),
                    'recon_speed': recon_err_per_feature[2].item(),
                    'recon_accel': recon_err_per_feature[3].item(),
                    'recon_sin_h': recon_err_per_feature[4].item(),
                    'recon_cos_h': recon_err_per_feature[5].item(),
                    'recon_ang_vel': recon_err_per_feature[6].item(),
                    'recon_dt': recon_err_per_feature[7].item(),
                }, step=self.global_step)

                # Also log to Global for cross-phase comparison
                self._log_scalars('Global', {
                    'v4_score': v4_score,
                    'recon_loss': val_recon,
                }, step=self.global_step)

                # Enhanced logging (histograms/distributions - every 500 iterations to avoid overhead)
                if iteration % 500 == 0 and self.tb_logger:
                    # Log latent distributions (h_real only in Phase 1 - no generator yet)
                    self._log_latent_distributions(h, h_fake=None, step=self.global_step)
                    # Log weight/gradient histograms
                    self._log_histograms('Phase1', step=self.global_step)

                    # Trajectory visualization for Phase 1 (real vs reconstructed)
                    with torch.no_grad():
                        batch_size_viz = min(6, x.size(0))
                        h_viz = self.model.embedder(x[:batch_size_viz], lengths[:batch_size_viz])
                        x_recon_viz = self.model.recovery(h_viz, lengths[:batch_size_viz])

                        # Log trajectory comparison (real vs reconstructed)
                        self._log_trajectory_comparison(
                            x[:batch_size_viz], x_recon_viz,
                            lengths=lengths[:batch_size_viz],
                            step=self.global_step,
                            tag='Trajectories/phase1_reconstruction'
                        )

                        # Log enhanced trajectory paths (3-row view: real, reconstructed, overlay)
                        self._log_trajectory_paths(
                            x[:batch_size_viz], x_recon_viz,
                            conditions=None,
                            lengths=lengths[:batch_size_viz],
                            step=self.global_step
                        )

                        # Log feature distribution comparison (real vs reconstructed)
                        self._log_distribution_comparison(
                            x[:batch_size_viz], x_recon_viz,
                            step=self.global_step
                        )

                # Increment global step
                self._increment_global_step()

                # Save best (Phase 1: use val_recon only, supervisor not trained yet)
                if val_recon < best_v4_score and checkpoint_dir:
                    best_v4_score = val_recon
                    self.best_v4_score = val_recon  # Track for final metrics
                    self.last_val_recon = val_recon
                    self.save_checkpoint(os.path.join(checkpoint_dir, 'phase1_best.pt'))
                    print(f"         --> New best! (val_recon: {val_recon:.6f})")

        # Final diagnostics
        print("-" * 70)
        print(f"Phase 1 Complete!")
        print(f"  Final val_recon: {val_recon:.6f}")
        print(f"  Best V4 Score: {best_v4_score:.6f}")
        print(f"  LR reductions: {lr_reductions}")
        print(f"  Final LR: {current_lr:.2e}")
        print(f"  Latent H stats: mean={h_mean:+.4f}, std={h_std:.4f}")
        if val_recon > 0.05:
            print(f"  [!]  WARNING: val_recon is high - autoencoder may not have converged")
        elif val_recon < 0.001:
            print(f"  [OK] Excellent reconstruction quality!")
        else:
            print(f"  [OK] Good reconstruction quality")
        print("=" * 70)

        # Save final
        if checkpoint_dir:
            self.save_checkpoint(os.path.join(checkpoint_dir, 'phase1_final.pt'))

    # =========================================================================
    # PHASE 2: SUPERVISED TRAINING (Iteration-Based)
    # =========================================================================

    def train_supervised_step(self, x, lengths):
        """Single supervised training step. Embedder is FROZEN."""
        self.model.train()
        self.optimizer_sup.zero_grad()

        # CRITICAL: Embedder is frozen - don't modify Phase 1 learning
        with torch.no_grad():
            h = self.model.embedder(x, lengths)

        h_supervised = self.model.supervisor(h, lengths)

        # Use helper function with proper edge case handling
        loss = self._compute_supervised_loss(h, h_supervised, lengths)

        loss.backward()
        s_grad = torch.nn.utils.clip_grad_norm_(self.model.supervisor.parameters(), self.max_grad_norm)
        self.optimizer_sup.step()

        # Return prediction and target for diagnostics (handle edge case)
        h_target = h[:, 1:, :] if h.size(1) > 1 else h
        h_pred = h_supervised[:, :-1, :] if h_supervised.size(1) > 1 else h_supervised

        return loss.item(), h_pred, h_target, s_grad.item()

    def train_supervised(self, dataloader, val_dataloader, max_iterations,
                         validate_every=500, checkpoint_dir=None):
        """
        Phase 2: Train supervisor for temporal dynamics (embedder FROZEN).

        Goal: Teach supervisor to predict h(t+1) from h(t).
        Success metric: val_sup < 0.01 (good), < 0.001 (excellent)
        """
        print("\n" + "=" * 70)
        print("PHASE 2: SUPERVISED TRAINING (Embedder FROZEN)")
        print("=" * 70)
        print(f"Purpose: Train Supervisor to predict temporal dynamics h(t) -> h(t+1)")
        print(f"         Embedder is FROZEN to preserve Phase 1 autoencoder learning")
        print("-" * 70)
        print(f"Iterations: {max_iterations}")
        print(f"Validate every: {validate_every}")
        print(f"LR_Supervisor: {self.lr_sup:.2e}")
        print(f"LR Scheduler: ReduceLROnPlateau (patience={self.lr_patience}, factor={self.lr_factor})")
        print("-" * 70)
        print(f"Target: val_sup < 0.01 (good), < 0.001 (excellent)")
        print("=" * 70)

        # Flush stdout before creating DataLoader iterator to prevent
        # interleaved output on Windows when num_workers > 0
        sys.stdout.flush()

        iterator = iter(dataloader)
        best_v4_score = float('inf')
        lr_reductions = 0
        last_lr = self.lr_sup

        for iteration in range(max_iterations):
            # Get next batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            x, condition, lengths = batch
            x = x.to(self.device)
            lengths = lengths.to(self.device)

            # Training step
            loss, h_pred, h_target, s_grad = self.train_supervised_step(x, lengths)

            # Validate & checkpoint
            if iteration % validate_every == 0 or iteration == max_iterations - 1:
                v4_score, val_recon, val_sup = self.compute_v4_score(val_dataloader)

                # Scheduler step
                self.scheduler_sup.step(val_sup)

                # Check for LR reduction
                current_lr = self.optimizer_sup.param_groups[0]['lr']
                if current_lr < last_lr:
                    lr_reductions += 1
                    print(f"  [!]  LR reduced: {last_lr:.2e} -> {current_lr:.2e} (reduction #{lr_reductions})")
                    last_lr = current_lr

                # Compute diagnostic stats
                with torch.no_grad():
                    pred_err = (h_pred - h_target.detach()).abs().mean().item()

                # Progress indicator
                progress = iteration / max_iterations * 100

                print(f"  [Iter {iteration:5d}/{max_iterations}] ({progress:5.1f}%) | "
                      f"Loss: {loss:.6f} | Val_Sup: {val_sup:.6f} | Val_Recon: {val_recon:.6f} | "
                      f"Grad S:{s_grad:.2f} | LR: {current_lr:.2e}")

                # Log scalars
                self._log_scalars('Phase2', {
                    'train_loss': loss,
                    'val_sup': val_sup,
                    'val_recon': val_recon,
                    'v4_score': v4_score,
                    'lr': current_lr,
                    'grad_supervisor': s_grad,
                }, step=self.global_step)

                # Global metrics for cross-phase comparison
                self._log_scalars('Global', {
                    'v4_score': v4_score,
                    'sup_loss': val_sup,
                }, step=self.global_step)

                # Enhanced logging (every 500 iterations)
                if iteration % 500 == 0 and self.tb_logger:
                    self._log_histograms('Phase2', step=self.global_step)

                    # Trajectory visualization for Phase 2 (real vs supervised reconstruction)
                    with torch.no_grad():
                        batch_size_viz = min(6, x.size(0))
                        # Full pipeline: embedder -> supervisor -> recovery
                        h_viz = self.model.embedder(x[:batch_size_viz], lengths[:batch_size_viz])
                        h_sup_viz = self.model.supervisor(h_viz, lengths[:batch_size_viz])
                        x_sup_recon_viz = self.model.recovery(h_sup_viz, lengths[:batch_size_viz])

                        # Also get direct reconstruction (without supervisor) for comparison
                        x_direct_recon_viz = self.model.recovery(h_viz, lengths[:batch_size_viz])

                        # Log trajectory comparison (real vs supervised reconstruction)
                        self._log_trajectory_comparison(
                            x[:batch_size_viz], x_sup_recon_viz,
                            lengths=lengths[:batch_size_viz],
                            step=self.global_step,
                            tag='Trajectories/phase2_supervised'
                        )

                        # Log direct reconstruction for comparison (should be same as Phase 1)
                        self._log_trajectory_comparison(
                            x[:batch_size_viz], x_direct_recon_viz,
                            lengths=lengths[:batch_size_viz],
                            step=self.global_step,
                            tag='Trajectories/phase2_direct_recon'
                        )

                        # Log enhanced trajectory paths (3-row view)
                        self._log_trajectory_paths(
                            x[:batch_size_viz], x_sup_recon_viz,
                            conditions=None,
                            lengths=lengths[:batch_size_viz],
                            step=self.global_step
                        )

                        # Log feature distribution comparison
                        self._log_distribution_comparison(
                            x[:batch_size_viz], x_sup_recon_viz,
                            step=self.global_step
                        )

                        # Log latent distributions (h_real vs h_supervised)
                        self._log_latent_distributions(h_viz, h_sup_viz, step=self.global_step)

                # Increment global step
                self._increment_global_step()

                # Save best
                if v4_score < best_v4_score and checkpoint_dir:
                    best_v4_score = v4_score
                    self.best_v4_score = v4_score
                    self.last_val_recon = val_recon
                    self.last_val_sup = val_sup
                    self.save_checkpoint(os.path.join(checkpoint_dir, 'phase2_best.pt'))
                    print(f"         --> New best V4 Score!")

        # Final diagnostics
        print("-" * 70)
        print(f"Phase 2 Complete!")
        print(f"  Final val_sup: {val_sup:.6f}")
        print(f"  Final val_recon: {val_recon:.6f} (should be similar to Phase 1 - embedder was frozen)")
        print(f"  Best V4 Score: {best_v4_score:.6f}")
        print(f"  LR reductions: {lr_reductions}")
        print(f"  Final LR: {current_lr:.2e}")
        if val_sup > 0.05:
            print(f"  [!]  WARNING: val_sup is high - supervisor may not have converged")
        elif val_sup < 0.001:
            print(f"  [OK] Excellent temporal prediction quality!")
        else:
            print(f"  [OK] Good temporal prediction quality")
        print("=" * 70)

        # Save final
        if checkpoint_dir:
            self.save_checkpoint(os.path.join(checkpoint_dir, 'phase2_final.pt'))

    # =========================================================================
    # PHASE 2.5: GENERATOR PRETRAINING (CRITICAL FOR PHASE 3 STABILITY)
    # =========================================================================

    def train_generator_pretrain_step(self, x, condition, lengths):
        """
        Comprehensive generator pretraining step.

        Trains generator to:
        1. Match embedder's latent DISTRIBUTION (not just mean) via variance + moment matching
        2. Learn to USE the condition input via condition reconstruction loss
        3. Be compatible with supervisor via matching supervised outputs
        4. Produce temporally coherent outputs via smoothness loss

        Without these, the generator:
        - Collapses to outputting the mean (variance issue)
        - Ignores the condition input entirely
        - Produces outputs the supervisor can't process correctly
        - Lacks temporal structure
        """
        self.optimizer_g.zero_grad()

        batch_size = x.size(0)
        seq_len = x.size(1)

        # Get real embeddings and supervised outputs (embedder frozen)
        with torch.no_grad():
            h_real = self.model.embedder(x, lengths)
            h_real_sup = self.model.supervisor(h_real, lengths)

        # Generate fake embeddings
        z = torch.randn(batch_size, seq_len, self.model.latent_dim, device=self.device)
        h_fake = self.model.generator(z, condition, lengths)

        # Pass through supervisor (gradients flow to generator, not supervisor)
        h_fake_sup = self.model.supervisor(h_fake, lengths)

        # Create mask for valid timesteps
        mask = self._create_mask(lengths, seq_len)

        # =====================================================================
        # LOSS 1: MSE on SUPERVISED outputs (supervisor compatibility)
        # =====================================================================
        # By matching after supervisor, we ensure generator produces outputs
        # that the supervisor can process correctly
        mse_loss = self._masked_mse(h_fake_sup, h_real_sup, mask)

        # =====================================================================
        # LOSS 2: Distribution matching (variance + higher moments)
        # =====================================================================
        # Match the full distribution, not just the mean
        var_loss, moment_loss = self._compute_pretrain_distribution_loss(h_fake, h_real, mask)

        # =====================================================================
        # LOSS 3: Condition learning (CRITICAL)
        # =====================================================================
        # Decode h_fake and check if the resulting trajectory matches the condition
        # This teaches the generator to ACTUALLY USE the condition input
        x_fake = self.model.recovery(h_fake_sup, lengths)
        cond_loss = self.compute_condition_loss(x_fake, condition, lengths)

        # =====================================================================
        # LOSS 4: Temporal smoothness
        # =====================================================================
        # Penalize large jumps in latent space - real trajectories are smooth
        smooth_loss = self._compute_pretrain_smoothness_loss(h_fake, mask)

        # =====================================================================
        # Combined loss with configurable weights
        # =====================================================================
        total_loss = (
            mse_loss +
            self.lambda_var_pretrain * var_loss +
            self.lambda_moment_pretrain * moment_loss +
            self.lambda_cond_pretrain * cond_loss +
            self.lambda_smooth_pretrain * smooth_loss
        )

        total_loss.backward()

        # Clip gradients and record norm for monitoring
        g_grad = torch.nn.utils.clip_grad_norm_(
            self.model.generator.parameters(),
            self.max_grad_norm
        )

        self.optimizer_g.step()

        # Return dictionary with all losses and diagnostics for logging
        return {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'var_loss': var_loss.item(),
            'moment_loss': moment_loss.item(),
            'cond_loss': cond_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'h_fake': h_fake.detach(),
            'h_real': h_real.detach(),
            'x_fake': x_fake.detach(),  # For trajectory visualization
            'g_grad': g_grad.item(),    # Gradient norm for monitoring
        }

    def _compute_pretrain_distribution_loss(self, h_fake, h_real, mask):
        """
        Match distribution statistics: variance, skewness, kurtosis, and range.

        This prevents the generator from collapsing to outputting the mean,
        which minimizes MSE but kills diversity.

        Includes:
        - Global variance (distribution spread)
        - Per-sample variance (prevents identical outputs)
        - Higher moments (skewness, kurtosis for distribution shape)
        - Range loss (ensures generator covers full range)

        Returns:
            var_loss: Variance + range matching loss
            moment_loss: Higher moments loss (skewness + kurtosis)
        """
        n_valid = mask.sum()
        if n_valid <= 1:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        # Apply mask
        h_real_masked = h_real * mask
        h_fake_masked = h_fake * mask

        # =====================================================================
        # Global variance (across batch and sequence)
        # =====================================================================
        h_real_mean = h_real_masked.sum(dim=(0, 1)) / n_valid
        h_fake_mean = h_fake_masked.sum(dim=(0, 1)) / n_valid

        # Centered values
        h_real_centered = (h_real_masked - h_real_mean.unsqueeze(0).unsqueeze(0)) * mask
        h_fake_centered = (h_fake_masked - h_fake_mean.unsqueeze(0).unsqueeze(0)) * mask

        # Variance (2nd central moment)
        h_real_var = (h_real_centered ** 2).sum(dim=(0, 1)) / n_valid
        h_fake_var = (h_fake_centered ** 2).sum(dim=(0, 1)) / n_valid

        # Use std (sqrt of var) for more stable gradients
        h_real_std = torch.sqrt(h_real_var + 1e-8)
        h_fake_std = torch.sqrt(h_fake_var + 1e-8)

        # Variance loss: match standard deviations per latent dimension
        global_var_loss = (h_real_std - h_fake_std).abs().mean()

        # =====================================================================
        # Per-sample variance (prevents all samples being identical)
        # =====================================================================
        # Each generated sample should have internal variance, not just global variance
        # Use keepdim=True for proper broadcasting: (batch, 1, 1)

        mask_sum_per_sample = mask.sum(dim=1, keepdim=True) + 1e-8  # (batch, 1, 1)

        per_sample_mean_real = h_real_masked.sum(dim=1, keepdim=True) / mask_sum_per_sample  # (batch, 1, latent)
        per_sample_mean_fake = h_fake_masked.sum(dim=1, keepdim=True) / mask_sum_per_sample  # (batch, 1, latent)

        # Squeeze mask_sum for the final division: (batch, 1, 1) -> (batch, 1)

        mask_sum_squeezed = mask_sum_per_sample.squeeze(1)  # (batch, 1)

        per_sample_var_real = ((h_real_masked - per_sample_mean_real) ** 2 * mask).sum(dim=1) / mask_sum_squeezed
        per_sample_var_fake = ((h_fake_masked - per_sample_mean_fake) ** 2 * mask).sum(dim=1) / mask_sum_squeezed

        # Average variance per sample, then compare (shape: latent_dim)
        per_sample_var_loss = (per_sample_var_real.mean(dim=0) - per_sample_var_fake.mean(dim=0)).abs().mean()

        # =====================================================================
        # Range loss (ensures generator covers full range of real data)
        # =====================================================================
        # One-sided: only penalize if fake range is SMALLER than real range
        h_real_range = h_real_masked.max() - h_real_masked.min()
        h_fake_range = h_fake_masked.max() - h_fake_masked.min()
        range_loss = torch.abs(h_real_range - h_fake_range)


        # This is the most direct way to keep sigma_ratio near 1.0
        mean_real_std = h_real_std.mean()
        mean_fake_std = h_fake_std.mean()
        std_ratio = mean_fake_std / (mean_real_std + 1e-8)
        # Penalize deviation from 1.0 (log makes it symmetric: 0.5 and 2.0 have same penalty)
        std_ratio_loss = (torch.log(std_ratio + 1e-8)).abs()

        # Combined variance loss
        var_loss = global_var_loss + 0.5 * per_sample_var_loss + 0.2 * range_loss + 10.0 * std_ratio_loss

        # =====================================================================
        # Higher moments: skewness (3rd) and kurtosis (4th)
        # =====================================================================
        # These capture distribution shape beyond mean/variance

        # Standardized values for moment computation
        h_real_standardized = h_real_centered / (h_real_std.unsqueeze(0).unsqueeze(0) + 1e-8)
        h_fake_standardized = h_fake_centered / (h_fake_std.unsqueeze(0).unsqueeze(0) + 1e-8)

        # Skewness (3rd standardized moment) - measures asymmetry
        h_real_skew = ((h_real_standardized ** 3) * mask).sum(dim=(0, 1)) / n_valid
        h_fake_skew = ((h_fake_standardized ** 3) * mask).sum(dim=(0, 1)) / n_valid

        # Kurtosis (4th standardized moment) - measures tail behavior
        h_real_kurt = ((h_real_standardized ** 4) * mask).sum(dim=(0, 1)) / n_valid
        h_fake_kurt = ((h_fake_standardized ** 4) * mask).sum(dim=(0, 1)) / n_valid

        # Moment loss: weighted combination
        skew_loss = (h_real_skew - h_fake_skew).abs().mean()
        kurt_loss = (h_real_kurt - h_fake_kurt).abs().mean()

        # Kurtosis is typically larger, so weight it less
        moment_loss = skew_loss + 0.3 * kurt_loss

        # DEBUG: Verify std_ratio_loss is being computed
        #print(f"DEBUG var_loss components: global={global_var_loss.item():.4f}, "
        #      f"per_sample={per_sample_var_loss.item():.4f}, range={range_loss.item():.4f}, "
        #      f"std_ratio={std_ratio_loss.item():.4f} (sigma={std_ratio.item():.3f})")

        return var_loss, moment_loss

    def _compute_pretrain_smoothness_loss(self, h_fake, mask):
        """
        Penalize large jumps in latent space between consecutive timesteps.

        Real mouse trajectories are smooth, so latent codes should also be smooth.
        This encourages temporally coherent generator outputs.

        Args:
            h_fake: (batch, seq_len, latent_dim) generated latent codes
            mask: (batch, seq_len, 1) validity mask

        Returns:
            smooth_loss: Mean squared difference between consecutive timesteps
        """
        if h_fake.size(1) <= 1:
            return torch.tensor(0.0, device=self.device)

        # Compute differences between consecutive timesteps
        h_diff = h_fake[:, 1:, :] - h_fake[:, :-1, :]  # (batch, seq-1, latent)

        # Mask for valid transitions (both timesteps must be valid)
        mask_transitions = mask[:, 1:, :] * mask[:, :-1, :]  # (batch, seq-1, 1)

        # Mean squared difference (penalize large jumps)
        n_transitions = mask_transitions.sum() + 1e-8
        smooth_loss = ((h_diff ** 2) * mask_transitions).sum() / n_transitions

        return smooth_loss

    def train_generator_pretrain(self, dataloader, max_iterations, log_every=100, checkpoint_dir=None):
        """
        Phase 2.5: Comprehensive generator pretraining.

        This phase prepares the generator for Phase 3 by teaching it to:
        1. Match the embedder's latent DISTRIBUTION (not just mean)
        2. Use the condition input to produce correct path distances
        3. Produce outputs compatible with the supervisor
        4. Generate temporally smooth outputs

        Without this comprehensive pretraining:
        - Generator produces low-variance outputs (easy for D to spot)
        - Generator ignores condition input
        - Supervisor can't process generator outputs correctly
        - Generated trajectories lack temporal coherence
        """
        print("\n" + "=" * 70)
        print("PHASE 2.5: COMPREHENSIVE GENERATOR PRETRAINING")
        print("=" * 70)
        print(f"Purpose: Prepare generator for Phase 3 with multi-objective training")
        print(f"")
        print(f"Training objectives:")
        print(f"  1. MSE Loss:      Match supervised latent distribution")
        print(f"  2. Variance Loss: Match distribution spread (prevent mean-collapse)")
        print(f"  3. Moment Loss:   Match higher moments (skewness, kurtosis)")
        print(f"  4. Condition Loss: Learn to use condition input (lambda={self.lambda_cond_pretrain:.1f})")
        print(f"  5. Smooth Loss:   Temporal coherence (lambda={self.lambda_smooth_pretrain:.2f})")
        print("-" * 70)
        print(f"Iterations: {max_iterations}")
        print(f"LR_Generator: {self.lr_g:.2e}")
        print(f"Latent dim: {self.model.latent_dim}")
        print(f"Target: std_ratio ~ 1.0, cond_loss decreasing")
        print("=" * 70)

        # Flush stdout before creating iterator (Windows compatibility)
        sys.stdout.flush()

        self.model.train()

        # =====================================================================
        # CRITICAL: Freeze supervisor and recovery during Phase 2.5
        # =====================================================================
        # We want to train ONLY the generator to match the embedder's latent space.
        # If supervisor/recovery update, their output distribution shifts and
        # the generator is chasing a moving target.
        print("  Freezing: Embedder, Supervisor, Recovery (only Generator trains)")
        for param in self.model.embedder.parameters():
            param.requires_grad = False
        for param in self.model.supervisor.parameters():
            param.requires_grad = False
        for param in self.model.recovery.parameters():
            param.requires_grad = False

        iterator = iter(dataloader)

        # Track best metrics for monitoring
        best_cond_loss = float('inf')
        best_std_ratio = 0.0
        best_std_ratio_iter = 0

        for iteration in range(max_iterations):
            # Get next batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            x, condition, lengths = batch
            x = x.to(self.device)
            condition = condition.to(self.device)
            lengths = lengths.to(self.device)

            # Training step (returns dictionary with all losses)
            results = self.train_generator_pretrain_step(x, condition, lengths)

            # Extract values
            total_loss = results['total_loss']
            mse_loss = results['mse_loss']
            var_loss = results['var_loss']
            moment_loss = results['moment_loss']
            cond_loss = results['cond_loss']
            smooth_loss = results['smooth_loss']
            h_fake = results['h_fake']
            h_real = results['h_real']
            g_grad = results['g_grad']

            # Track best condition loss
            if cond_loss < best_cond_loss:
                best_cond_loss = cond_loss

            # Logging
            if iteration % log_every == 0 or iteration == max_iterations - 1:
                # Compute diagnostic stats
                with torch.no_grad():
                    # Create mask for this batch
                    seq_len = h_fake.size(1)
                    mask = self._create_mask(lengths, seq_len)
                    n_valid = mask.sum()
                    
                    # Masked means
                    h_fake_masked = h_fake * mask
                    h_real_masked = h_real * mask
                    h_fake_mean = (h_fake_masked.sum() / (n_valid * h_fake.size(-1))).item()
                    h_real_mean = (h_real_masked.sum() / (n_valid * h_real.size(-1))).item()
                    
                    # Masked variances (matching loss function computation)
                    h_fake_mean_expanded = h_fake_masked.sum(dim=(0, 1)) / n_valid
                    h_real_mean_expanded = h_real_masked.sum(dim=(0, 1)) / n_valid
                    
                    h_fake_centered = (h_fake_masked - h_fake_mean_expanded.unsqueeze(0).unsqueeze(0)) * mask
                    h_real_centered = (h_real_masked - h_real_mean_expanded.unsqueeze(0).unsqueeze(0)) * mask
                    
                    h_fake_var = (h_fake_centered ** 2).sum(dim=(0, 1)) / n_valid
                    h_real_var = (h_real_centered ** 2).sum(dim=(0, 1)) / n_valid
                    
                    h_fake_std = torch.sqrt(h_fake_var + 1e-8).mean().item()
                    h_real_std = torch.sqrt(h_real_var + 1e-8).mean().item()
                    std_ratio = h_fake_std / (h_real_std + 1e-8)

                    # Compute range coverage
                    h_fake_min, h_fake_max = h_fake.min().item(), h_fake.max().item()
                    h_real_min, h_real_max = h_real.min().item(), h_real.max().item()
                    range_ratio = (h_fake_max - h_fake_min) / (h_real_max - h_real_min + 1e-8)

                # Track best std_ratio (closest to 1.0)
                std_ratio_quality = abs(std_ratio - 1.0)
                best_std_ratio_quality = abs(best_std_ratio - 1.0)
                is_best_std = std_ratio_quality < best_std_ratio_quality
                if is_best_std:
                    best_std_ratio = std_ratio
                    best_std_ratio_iter = iteration

                # Status indicators
                std_status = "[OK]" if 0.7 <= std_ratio <= 1.3 else "[!]"
                cond_status = "[OK]" if cond_loss < 0.5 else "[!]"
                best_marker = " -> BEST sigma" if is_best_std else ""

                # Detailed logging
                print(f"  [Iter {iteration:5d}/{max_iterations}] "
                      f"Loss: {total_loss:.4f} (MSE:{mse_loss:.4f} Var:{var_loss:.4f} "
                      f"Mom:{moment_loss:.4f} Cond:{cond_loss:.4f} Sm:{smooth_loss:.4f})")
                print(f"                        "
                      f"sigma_ratio: {std_ratio:.3f} {std_status} | range: {range_ratio:.3f} | "
                      f"cond: {cond_status} | g_grad: {g_grad:.3f}{best_marker}")

                # Log scalars
                self._log_scalars('Phase2_5', {
                    'total_loss': total_loss,
                    'mse_loss': mse_loss,
                    'var_loss': var_loss,
                    'moment_loss': moment_loss,
                    'cond_loss': cond_loss,
                    'smooth_loss': smooth_loss,
                    'h_fake_mean': h_fake_mean,
                    'h_fake_std': h_fake_std,
                    'h_real_std': h_real_std,
                    'std_ratio': std_ratio,
                    'range_ratio': range_ratio,
                    'g_grad_norm': g_grad,
                }, step=self.global_step)

                # Enhanced logging - latent distribution comparison
                if iteration % 500 == 0 and self.tb_logger:
                    self._log_latent_distributions(h_real, h_fake, step=self.global_step)

                # Trajectory visualization (like Phase 3)
                if iteration % 500 == 0 and self.tb_logger:
                    x_fake = results['x_fake']
                    self._log_trajectory_comparison(
                        x, x_fake,
                        lengths=lengths,
                        step=self.global_step,
                        tag='Trajectories/Phase2_5'
                    )

                # Increment global step
                self._increment_global_step()

        # =====================================================================
        # Unfreeze for subsequent phases
        # =====================================================================
        print("  Unfreezing: Embedder, Supervisor, Recovery")
        for param in self.model.embedder.parameters():
            param.requires_grad = True
        for param in self.model.supervisor.parameters():
            param.requires_grad = True
        for param in self.model.recovery.parameters():
            param.requires_grad = True

        # Final diagnostic
        print("-" * 70)
        print(f"Phase 2.5 Complete!")
        print(f"  Final losses:")
        print(f"    Total: {total_loss:.6f}")
        print(f"    MSE: {mse_loss:.6f} | Var: {var_loss:.6f} | Mom: {moment_loss:.6f}")
        print(f"    Cond: {cond_loss:.6f} (best: {best_cond_loss:.6f}) | Smooth: {smooth_loss:.6f}")
        print(f"  Generator H range: [{h_fake.min().item():.3f}, {h_fake.max().item():.3f}]")
        print(f"  Real H range:      [{h_real.min().item():.3f}, {h_real.max().item():.3f}]")
        print(f"  Std ratio (fake/real): {std_ratio:.3f} (target: ~1.0)")
        print(f"  Best std ratio: {best_std_ratio:.3f} (at iteration {best_std_ratio_iter})")
        print(f"  Range ratio: {range_ratio:.3f} (target: ~1.0)")

        # Quality assessment
        issues = []
        if std_ratio < 0.7:
            issues.append(f"variance too low (sigma_ratio={std_ratio:.2f})")
        elif std_ratio > 1.4:
            issues.append(f"variance too high (sigma_ratio={std_ratio:.2f})")

        if range_ratio < 0.7:
            issues.append(f"range too narrow (range_ratio={range_ratio:.2f})")

        if cond_loss > 0.5:
            issues.append(f"condition not learned well (cond_loss={cond_loss:.3f})")

        if issues:
            print(f"  [!]  WARNINGS:")
            for issue in issues:
                print(f"      - {issue}")
            print(f"  Consider: more iterations, higher lambda_cond_pretrain, or check data")
        else:
            print(f"  [OK] Generator pretraining looks good!")
            if cond_loss < 0.3:
                print(f"  [OK] Condition learning successful (cond_loss={cond_loss:.3f})")
        print("=" * 70)

        # Save checkpoint
        if checkpoint_dir:
            self.save_checkpoint(os.path.join(checkpoint_dir, 'phase2_5_final.pt'))

    # =========================================================================
    # PHASE 2.75: DISCRIMINATOR WARMUP (CRITICAL FOR PHASE 3 STABILITY)
    # =========================================================================

    def train_discriminator_warmup(self, dataloader, max_iterations, log_every=100, checkpoint_dir=None):
        """
        Phase 2.75: Warm up discriminator before adversarial training.

        Pre-trains discriminator to distinguish real data from generator output
        BEFORE starting full adversarial training.

        Without warmup:
        - Discriminator sees fake data for the first time in Phase 3
        - May output extreme values due to sudden distribution shift
        - Can cause training instability

        With warmup:
        - Discriminator gradually learns to distinguish real vs fake
        - Outputs are more reasonable at Phase 3 start
        - Smoother transition into adversarial training
        """
        print("\n" + "=" * 70)
        print("PHASE 2.75: DISCRIMINATOR WARMUP")
        print("=" * 70)
        print(f"Purpose: Pre-train discriminator on real vs pretrained-generator output")
        print(f"         This prevents extreme D outputs at Phase 3 start")
        print("-" * 70)
        print(f"Iterations: {max_iterations}")
        print(f"LR_Discriminator: {self.lr_d:.2e}")
        print(f"Lambda_GP: {self.lambda_gp}")
        print(f"Generator: FROZEN (not updated)")
        print("=" * 70)

        self.model.train()

        # Freeze generator during warmup
        for param in self.model.generator.parameters():
            param.requires_grad = False

        iterator = iter(dataloader)

        for iteration in range(max_iterations):
            # Get next batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            x, condition, lengths = batch
            x = x.to(self.device)
            condition = condition.to(self.device)
            lengths = lengths.to(self.device)

            batch_size = x.size(0)
            seq_len = x.size(1)

            self.optimizer_d.zero_grad()

            # Generate fake data (no gradients to generator, with supervisor)
            with torch.no_grad():
                z = torch.randn(batch_size, seq_len, self.model.latent_dim, device=self.device)
                h_fake_raw = self.model.generator(z, condition, lengths)
                h_fake = self.model.supervisor(h_fake_raw, lengths)
                x_fake = self.model.recovery(h_fake, lengths)

            # Discriminator on real and fake
            d_real = self.model.discriminator(x, condition, lengths)
            d_fake = self.model.discriminator(x_fake, condition, lengths)

            # WGAN loss: D wants to maximize d_real - d_fake
            # So we minimize d_fake - d_real
            d_loss = d_fake.mean() - d_real.mean()

            # Gradient penalty
            gp = self._gradient_penalty(x, x_fake, condition, lengths)

            total_loss = d_loss + self.lambda_gp * gp

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.max_grad_norm)
            self.optimizer_d.step()

            # Logging
            if iteration % log_every == 0 or iteration == max_iterations - 1:
                d_real_mean = d_real.mean().item()
                d_fake_mean = d_fake.mean().item()
                wasserstein_dist = d_real_mean - d_fake_mean

                print(f"  [Iter {iteration:5d}/{max_iterations}] D_loss: {d_loss.item():+.4f} | "
                      f"GP: {gp.item():.4f} | "
                      f"D(real): {d_real_mean:+.3f} | D(fake): {d_fake_mean:+.3f} | "
                      f"W_dist: {wasserstein_dist:.3f}")

                # Log scalars
                self._log_scalars('Phase2_75', {
                    'd_loss': d_loss.item(),
                    'gp': gp.item(),
                    'd_real_mean': d_real_mean,
                    'd_fake_mean': d_fake_mean,
                    'wasserstein_dist': wasserstein_dist,
                }, step=self.global_step)

                # Increment global step
                self._increment_global_step()

        # Unfreeze generator
        for param in self.model.generator.parameters():
            param.requires_grad = True

        # Final diagnostic
        print("-" * 70)
        print(f"Phase 2.75 Complete!")
        print(f"  Final D_loss: {d_loss.item():.4f}")
        print(f"  D(real) mean: {d_real.mean().item():+.3f} (should be positive)")
        print(f"  D(fake) mean: {d_fake.mean().item():+.3f} (should be negative)")
        print(f"  Wasserstein distance: {wasserstein_dist:.3f}")
        print(f"  D(real) range: [{d_real.min().item():.3f}, {d_real.max().item():.3f}]")
        print(f"  D(fake) range: [{d_fake.min().item():.3f}, {d_fake.max().item():.3f}]")

        if abs(d_real.mean().item()) > 50 or abs(d_fake.mean().item()) > 50:
            print(f"  [!]  WARNING: Discriminator outputs are extreme - may cause instability")
        elif wasserstein_dist < 0:
            print(f"  [!]  WARNING: Negative W_dist - discriminator thinks fake is more real!")
        else:
            print(f"  [OK] Discriminator outputs look reasonable for Phase 3")
        print("=" * 70)

        # Save checkpoint
        if checkpoint_dir:
            self.save_checkpoint(os.path.join(checkpoint_dir, 'phase2_75_final.pt'))

    # =========================================================================
    # PHASE 3: JOINT ADVERSARIAL TRAINING (Iteration-Based)
    # =========================================================================

    def train_discriminator_step(self, x, condition, lengths):
        """Single discriminator training step (WGAN-GP)."""
        self.optimizer_d.zero_grad()

        batch_size = x.size(0)
        seq_len = x.size(1)

        # Generate fake data (with supervisor for temporal coherence)
        z = torch.randn(batch_size, seq_len, self.model.latent_dim, device=self.device)
        with torch.no_grad():
            h_fake_raw = self.model.generator(z, condition, lengths)
            h_fake = self.model.supervisor(h_fake_raw, lengths)
            x_fake = self.model.recovery(h_fake, lengths)

        # Discriminator outputs
        d_real = self.model.discriminator(x, condition, lengths)
        d_fake = self.model.discriminator(x_fake, condition, lengths)

        # WGAN loss
        d_loss = d_fake.mean() - d_real.mean()

        # Gradient penalty
        gp = self._gradient_penalty(x, x_fake, condition, lengths)

        # Total discriminator loss
        total_loss = d_loss + self.lambda_gp * gp

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.max_grad_norm)
        self.optimizer_d.step()

        return d_loss.item(), gp.item()

    def train_generator_step(self, x, condition, lengths):
        """
        Single generator training step.

        Generator loss = adversarial + supervised + reconstruction + variance + condition
                       + jerk + higher_moments + autocorrelation + LATENT_VARIANCE

        The generation flow is:
        1. Generator produces initial latent codes (h_fake_raw)
        2. Supervisor refines temporal dynamics (h_fake)
        3. Recovery decodes to feature space (x_fake)

        Returns:
            Dictionary with all loss components for logging
        """
        self.optimizer_g.zero_grad()
        self.optimizer_er.zero_grad()

        batch_size = x.size(0)
        seq_len = x.size(1)
        mask = self._create_mask(lengths, seq_len)

        # Generate fake data (with supervisor for temporal coherence)
        z = torch.randn(batch_size, seq_len, self.model.latent_dim, device=self.device)
        h_fake_raw = self.model.generator(z, condition, lengths)
        h_fake = self.model.supervisor(h_fake_raw, lengths)
        x_fake = self.model.recovery(h_fake, lengths)

        # Adversarial loss
        d_fake = self.model.discriminator(x_fake, condition, lengths)
        g_loss = -d_fake.mean()

        # Supervised loss: verify temporal coherence of refined latent codes
        # The supervisor's prediction at step t should match the actual value at t+1
        h_fake_next_pred = self.model.supervisor(h_fake, lengths)
        sup_loss = self._compute_supervised_loss(h_fake, h_fake_next_pred, lengths)

        # Reconstruction loss with feature weighting (keeps autoencoder sharp)
        h_real = self.model.embedder(x, lengths)
        x_recon = self.model.recovery(h_real, lengths)
        recon_loss = self._masked_mse(x_recon, x, mask, use_feature_weights=True)

        # Variance matching loss (feature space - accel, ang_vel)
        var_loss = self.compute_variance_loss(x, x_fake, mask)

        # LATENT VARIANCE LOSS (preserves Phase 2.5 learning!)
        # CRITICAL: Use h_fake_raw (generator direct output), NOT h_fake (post-supervisor)
        # Phase 2.5 trains generator to output h_fake_raw with correct variance
        # Phase 3 must continue this by measuring the SAME thing: h_fake_raw vs h_real
        latent_var_loss, std_ratio = self.compute_latent_variance_loss(h_real, h_fake_raw, mask)

        # Condition reconstruction loss (path distance matching)
        cond_loss = self.compute_condition_loss(x_fake, condition, lengths)

        # Style-specific losses
        jerk_loss = self.compute_jerk_loss(x, x_fake, mask)
        moments_loss = self.compute_higher_moments_loss(x, x_fake, mask)
        autocorr_loss = self.compute_autocorr_loss(x, x_fake, mask)

        # Total generator loss (now includes latent variance!)
        total_loss = (g_loss +
                      self.lambda_sup * sup_loss +
                      self.lambda_recon * recon_loss +
                      self.lambda_var * var_loss +
                      self.lambda_latent_var * latent_var_loss +
                      self.lambda_cond * cond_loss +
                      self.lambda_jerk * jerk_loss +
                      self.lambda_higher_moments * moments_loss +
                      self.lambda_autocorr * autocorr_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer_g.step()
        self.optimizer_er.step()

        # Return all losses as a dictionary for flexible logging
        return {
            'g_loss': g_loss.item(),
            'sup_loss': sup_loss.item(),
            'recon_loss': recon_loss.item(),
            'var_loss': var_loss.item(),
            'latent_var_loss': latent_var_loss.item(),
            'std_ratio': std_ratio,
            'cond_loss': cond_loss.item(),
            'jerk_loss': jerk_loss.item(),
            'moments_loss': moments_loss.item(),
            'autocorr_loss': autocorr_loss.item(),
        }

    def train_joint(self, dataloader, val_dataloader, max_iterations,
                    validate_every=500, checkpoint_dir=None):
        """
        Phase 3: Joint adversarial training.

        Goal: Generate trajectories that fool discriminator while maintaining
        condition fidelity (correct path length) and variance (style diversity).

        CRITICAL: 1 iteration = 1 Generator update
                  Each iteration uses n_critic+1 batches
        """
        print("\n" + "=" * 70)
        print("PHASE 3: JOINT ADVERSARIAL TRAINING")
        print("=" * 70)
        print(f"Purpose: Full GAN training - Generator vs Discriminator")
        print(f"         Generator learns to produce realistic trajectories")
        print(f"         Discriminator learns to detect fake trajectories")
        print("-" * 70)
        print(f"Iterations: {max_iterations}")
        print(f"  = {max_iterations:,} Generator updates")
        print(f"  = {max_iterations * self.n_critic:,} Discriminator updates (n_critic={self.n_critic})")
        print(f"Validate every: {validate_every}")
        print("-" * 70)
        print(f"Learning Rates:")
        print(f"  LR_Generator: {self.lr_g:.2e}")
        print(f"  LR_Discriminator: {self.lr_d:.2e}")
        print(f"Loss Weights:")
        print(f"  Lambda_Recon: {self.lambda_recon:.1f}")
        print(f"  Lambda_Sup: {self.lambda_sup:.1f}")
        print(f"  Lambda_GP: {self.lambda_gp:.1f}")
        print(f"  Lambda_Var: {self.lambda_var:.1f}")
        print(f"  Lambda_Cond: {self.lambda_cond:.1f}")
        print("-" * 70)
        print(f"V4 Score = recon + sup + {self.v4_score_cond_weight}*cond + {self.v4_score_var_weight}*var")
        print(f"Target: V4 < 0.1 (good), < 0.05 (excellent)")
        if self.norm_params is None:
            print(f"[!]  WARNING: norm_params not provided - condition loss disabled")
        print("-" * 70)
        print(f"Warmup: {self.warmup_steps} steps")
        print(f"Early Stopping: {'Disabled' if self.early_stopping_patience == 0 else f'patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta}'}")
        print("=" * 70)

        # Flush stdout before creating DataLoader iterator to prevent
        # interleaved output on Windows when num_workers > 0
        sys.stdout.flush()

        iterator = iter(dataloader)
        best_v4_score = float('inf')
        lr_g_reductions = 0
        lr_d_reductions = 0
        last_lr_g = self.lr_g
        last_lr_d = self.lr_d

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            mode='min'
        )

        # Initialize warmup schedulers for Generator and Discriminator
        warmup_g = LinearWarmup(self.optimizer_g, self.warmup_steps, self.lr_g)
        warmup_d = LinearWarmup(self.optimizer_d, self.warmup_steps, self.lr_d)
        warmup_er = LinearWarmup(self.optimizer_er, self.warmup_steps, self.lr_er)
        in_warmup = True

        def get_batch():
            """Get next batch, cycling if needed."""
            nonlocal iterator
            try:
                return next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                return next(iterator)

        for iteration in range(max_iterations):

            # =================================================================
            # STEP 1: Train Discriminator n_critic times
            # =================================================================
            for _ in range(self.n_critic):
                x, condition, lengths = get_batch()
                x = x.to(self.device)
                condition = condition.to(self.device)
                lengths = lengths.to(self.device)

                d_loss, gp = self.train_discriminator_step(x, condition, lengths)

            # =================================================================
            # STEP 2: Train Generator 1 time
            # =================================================================
            x, condition, lengths = get_batch()
            x = x.to(self.device)
            condition = condition.to(self.device)
            lengths = lengths.to(self.device)

            g_losses = self.train_generator_step(x, condition, lengths)
            g_loss = g_losses['g_loss']
            sup_loss = g_losses['sup_loss']
            recon_loss = g_losses['recon_loss']
            var_loss = g_losses['var_loss']
            latent_var_loss = g_losses['latent_var_loss']
            train_std_ratio = g_losses['std_ratio']
            cond_loss = g_losses['cond_loss']
            jerk_loss = g_losses['jerk_loss']
            moments_loss = g_losses['moments_loss']
            autocorr_loss = g_losses['autocorr_loss']

            # Update EMA weights for generator, supervisor, and recovery
            self.ema_generator.update()
            self.ema_supervisor.update()
            self.ema_recovery.update()

            # =================================================================
            # STEP 2.5: Warmup Learning Rate
            # =================================================================
            if in_warmup:
                still_warming_g = warmup_g.step()
                still_warming_d = warmup_d.step()
                still_warming_er = warmup_er.step()
                in_warmup = still_warming_g or still_warming_d or still_warming_er
                if not in_warmup:
                    print(f"  [Iter {iteration:5d}] Warmup complete - switching to ReduceLROnPlateau")

            # =================================================================
            # STEP 3: Validate & Checkpoint
            # =================================================================
            if iteration % validate_every == 0 or iteration == max_iterations - 1:
                # Use Phase 3 score that includes condition fidelity, variance, and style losses
                (v4_score, val_recon, val_sup, val_cond, val_var,
                 val_jerk, val_moments, val_autocorr, val_latent_var, val_std_ratio) = self.compute_v4_score_phase3(val_dataloader)

                # ReduceLROnPlateau - only step after warmup is complete
                if not in_warmup:
                    self.scheduler_g.step(v4_score)
                    self.scheduler_d.step(v4_score)
                    self.scheduler_er.step(val_recon)

                # Check for LR reductions
                lr_g = self.optimizer_g.param_groups[0]['lr']
                lr_d = self.optimizer_d.param_groups[0]['lr']

                if not in_warmup and lr_g < last_lr_g:
                    lr_g_reductions += 1
                    print(f"  [!]  LR_G reduced: {last_lr_g:.2e} -> {lr_g:.2e} (reduction #{lr_g_reductions})")
                    last_lr_g = lr_g

                if not in_warmup and lr_d < last_lr_d:
                    lr_d_reductions += 1
                    print(f"  [!]  LR_D reduced: {last_lr_d:.2e} -> {lr_d:.2e} (reduction #{lr_d_reductions})")
                    last_lr_d = lr_d

                # Progress indicator
                progress = iteration / max_iterations * 100

                # Std ratio status indicator (should be ~1.0)
                std_status = "[OK]" if 0.7 <= val_std_ratio <= 1.3 else "[!]"

                # Comprehensive logging line (now includes std_ratio!)
                print(f"  [Iter {iteration:5d}/{max_iterations}] ({progress:5.1f}%) | "
                      f"D:{d_loss:+7.3f} G:{g_loss:+7.3f} GP:{gp:.3f} | "
                      f"Recon:{val_recon:.4f} Cond:{val_cond:.4f} Var:{val_var:.4f} sigma:{val_std_ratio:.2f}{std_status} | "
                      f"V4:{v4_score:.4f}")

                # Log scalars
                self._log_scalars('Phase3', {
                    'd_loss': d_loss,
                    'g_loss': g_loss,
                    'sup_loss': sup_loss,
                    'recon_loss': recon_loss,
                    'var_loss': var_loss,
                    'latent_var_loss': latent_var_loss,
                    'std_ratio': train_std_ratio,
                    'cond_loss': cond_loss,
                    'jerk_loss': jerk_loss,
                    'moments_loss': moments_loss,
                    'autocorr_loss': autocorr_loss,
                    'val_recon': val_recon,
                    'val_sup': val_sup,
                    'val_cond': val_cond,
                    'val_var': val_var,
                    'val_latent_var': val_latent_var,
                    'val_std_ratio': val_std_ratio,
                    'val_jerk': val_jerk,
                    'val_moments': val_moments,
                    'val_autocorr': val_autocorr,
                    'gp': gp,
                    'v4_score': v4_score,
                    'lr_g': lr_g,
                    'lr_d': lr_d,
                }, step=self.global_step)

                # Global metrics for cross-phase comparison
                self._log_scalars('Global', {
                    'v4_score': v4_score,
                    'recon_loss': val_recon,
                    'var_loss': val_var,
                    'latent_var_loss': val_latent_var,
                    'std_ratio': val_std_ratio,
                    'cond_loss': val_cond,
                }, step=self.global_step)

                # Enhanced logging (every 500 iterations)
                if iteration % 500 == 0 and self.tb_logger:
                    # Log weight/gradient histograms
                    self._log_histograms('Phase3', step=self.global_step)

                    # Generate samples for visualization
                    with torch.no_grad():
                        batch_size_viz = min(6, x.size(0))
                        z_viz = torch.randn(batch_size_viz, x.size(1), self.model.latent_dim, device=self.device)
                        h_fake_viz = self.model.generator(z_viz, condition[:batch_size_viz], lengths[:batch_size_viz])
                        h_sup_viz = self.model.supervisor(h_fake_viz, lengths[:batch_size_viz])
                        x_fake_viz = self.model.recovery(h_sup_viz, lengths[:batch_size_viz])

                        # Log trajectory comparison (basic) - now with lengths to mask padding!
                        self._log_trajectory_comparison(
                            x[:batch_size_viz], x_fake_viz,
                            lengths=lengths[:batch_size_viz],
                            step=self.global_step,
                            tag='Trajectories/phase3_comparison'
                        )

                        # Log enhanced trajectory paths (3-row view: real, fake, overlay)
                        self._log_trajectory_paths(
                            x[:batch_size_viz], x_fake_viz,
                            conditions=condition[:batch_size_viz],
                            lengths=lengths[:batch_size_viz],
                            step=self.global_step
                        )

                        # Log speed heatmap
                        self._log_speed_heatmap(x[:batch_size_viz], x_fake_viz, step=self.global_step)

                        # Log feature distribution comparison
                        self._log_distribution_comparison(
                            x[:batch_size_viz], x_fake_viz,
                            step=self.global_step
                        )

                        # Log latent distributions (histograms)
                        h_real_viz = self.model.embedder(x[:batch_size_viz], lengths[:batch_size_viz])
                        self._log_latent_distributions(h_real_viz, h_fake_viz, step=self.global_step)

                        # Log discriminator score distributions
                        d_real_viz = self.model.discriminator(x[:batch_size_viz], condition[:batch_size_viz], lengths[:batch_size_viz])
                        d_fake_viz = self.model.discriminator(x_fake_viz, condition[:batch_size_viz], lengths[:batch_size_viz])
                        self._log_discriminator_scores(d_real_viz, d_fake_viz, step=self.global_step)

                # Embedding projector logging (every 2000 iterations - expensive)
                if iteration % 2000 == 0 and self.tb_logger:
                    with torch.no_grad():
                        batch_size_emb = min(32, x.size(0))
                        h_real_emb = self.model.embedder(x[:batch_size_emb], lengths[:batch_size_emb])
                        z_emb = torch.randn(batch_size_emb, x.size(1), self.model.latent_dim, device=self.device)
                        h_fake_emb = self.model.generator(z_emb, condition[:batch_size_emb], lengths[:batch_size_emb])
                        h_fake_emb = self.model.supervisor(h_fake_emb, lengths[:batch_size_emb])

                        # Log embeddings for Projector visualization (t-SNE/PCA/UMAP)
                        self._log_embeddings(
                            h_real_emb, h_fake_emb,
                            conditions=condition[:batch_size_emb],
                            step=self.global_step
                        )

                # Increment global step
                self._increment_global_step()

                # Save best model
                if v4_score < best_v4_score and checkpoint_dir:
                    best_v4_score = v4_score
                    self.best_v4_score = v4_score
                    self.last_val_recon = val_recon
                    self.last_val_var = val_var
                    self.last_val_cond = val_cond
                    self.last_val_sup = val_sup
                    self.save_checkpoint(os.path.join(checkpoint_dir, 'model_best_v4.pt'))
                    print(f"         --> New best V4 Score! (Recon:{val_recon:.4f} Cond:{val_cond:.4f} Var:{val_var:.4f})")

                # Check early stopping (only after warmup)
                if not in_warmup and early_stopping(v4_score):
                    print(f"\n  [!]  EARLY STOPPING triggered at iteration {iteration}")
                    print(f"      No improvement for {self.early_stopping_patience} validation checks")
                    print(f"      Best V4 Score: {best_v4_score:.6f}")
                    break

        # Final diagnostics
        print("-" * 70)
        print(f"Phase 3 Complete!{' (early stopped)' if early_stopping.triggered else ''}")
        print(f"  Iterations completed: {iteration + 1} / {max_iterations}")
        print(f"  Final V4 Score: {v4_score:.6f}")
        print(f"  Best V4 Score: {best_v4_score:.6f}")
        print(f"  Core metrics: Recon={val_recon:.4f}, Sup={val_sup:.4f}, Cond={val_cond:.4f}, Var={val_var:.4f}")
        print(f"  Style metrics: Jerk={val_jerk:.4f}, Moments={val_moments:.4f}, Autocorr={val_autocorr:.4f}")
        print(f"  LR_G reductions: {lr_g_reductions} (final: {lr_g:.2e})")
        print(f"  LR_D reductions: {lr_d_reductions} (final: {lr_d:.2e})")
        print(f"  Final D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")
        if val_cond > 0.5:
            print(f"  [!]  WARNING: Condition fidelity is poor - generated paths don't match requested distance")
        if val_var > 0.5:
            print(f"  [!]  WARNING: Variance matching is poor - generated paths lack style diversity")
        if val_autocorr > 0.5:
            print(f"  [!]  WARNING: Autocorrelation matching is poor - temporal patterns don't match")
        if v4_score < 0.05:
            print(f"  [OK] Excellent generation quality!")
        elif v4_score < 0.1:
            print(f"  [OK] Good generation quality")
        else:
            print(f"  [!]  Model may need more training or hyperparameter tuning")
        print("=" * 70)

        # Save final
        if checkpoint_dir:
            self.save_checkpoint(os.path.join(checkpoint_dir, 'phase3_final.pt'))

    # =========================================================================
    # FULL TRAINING PIPELINE
    # =========================================================================

    def train(self, train_dataloader, val_dataloader,
              iterations_ae=6000, iterations_sup=6000,
              iterations_gen_pretrain=2000, iterations_disc_warmup=500,
              iterations_joint=20000,
              validate_every=500, checkpoint_dir=None, log_dir=None):
        """
        Run complete 5-phase training pipeline (iteration-based).

        Training Phases:
            1.    Autoencoder (iterations_ae): Train Embedder + Recovery
            2.    Supervised (iterations_sup): Train Supervisor (embedder FROZEN)
            2.5   Generator Pretrain (iterations_gen_pretrain): Generator mimics embedder latent
            2.75  Discriminator Warmup (iterations_disc_warmup): D learns real vs fake
            3.    Joint Adversarial (iterations_joint): Full GAN training

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (REQUIRED)
            iterations_ae: Iterations for autoencoder phase
            iterations_sup: Iterations for supervised phase (embedder frozen)
            iterations_gen_pretrain: Iterations for generator pretraining
            iterations_disc_warmup: Iterations for discriminator warmup
            iterations_joint: Iterations for joint adversarial phase
            validate_every: Iterations between validation checks
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
        """
        total_iters = iterations_ae + iterations_sup + iterations_gen_pretrain + iterations_disc_warmup + iterations_joint

        print("\n" + "=" * 70)
        print("TIMEGAN V4 - COMPLETE TRAINING PIPELINE")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"")
        print(f"Training Phases:")
        print(f"  Phase 1:    Autoencoder        {iterations_ae:>6,} iters  (Embedder + Recovery)")
        print(f"  Phase 2:    Supervised         {iterations_sup:>6,} iters  (Supervisor, embedder FROZEN)")
        print(f"  Phase 2.5:  Generator Pretrain {iterations_gen_pretrain:>6,} iters  (G mimics embedder latent)")
        print(f"  Phase 2.75: Discriminator Warm {iterations_disc_warmup:>6,} iters  (D learns real vs fake)")
        print(f"  Phase 3:    Joint Adversarial  {iterations_joint:>6,} iters  (Full GAN training)")
        print(f"  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(f"  TOTAL:                         {total_iters:>6,} iters")
        print(f"")
        print(f"Validate every: {validate_every} iterations")
        print(f"LR Scheduler: ReduceLROnPlateau (patience={self.lr_patience}, factor={self.lr_factor})")
        print("=" * 70)

        # Setup logging
        # Use tb_logger if provided (preferred), otherwise fall back to legacy SummaryWriter
        if self.tb_logger:
            print(f"TensorBoard (Enhanced): {self.tb_logger.run_dir}")
            self._start_phase('training')
            # Log model architecture graphs
            print("Logging model architecture graphs...")
            self._log_model_graph()
        elif log_dir:
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard (Legacy): {log_dir}")

        # Setup checkpoints
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Checkpoints: {checkpoint_dir}")

        start_time = time.time()

        # =====================================================================
        # Phase 1: Autoencoder
        # =====================================================================
        self.train_autoencoder(
            train_dataloader, val_dataloader, iterations_ae,
            validate_every, checkpoint_dir
        )

        # =====================================================================
        # Phase 2: Supervised (embedder FROZEN)
        # =====================================================================
        self.train_supervised(
            train_dataloader, val_dataloader, iterations_sup,
            validate_every, checkpoint_dir
        )

        # =====================================================================
        # Phase 2.5: Generator Pretraining (NEW - critical for Phase 3!)
        # =====================================================================
        if iterations_gen_pretrain > 0:
            self.train_generator_pretrain(
                train_dataloader, iterations_gen_pretrain,
                log_every=100, checkpoint_dir=checkpoint_dir
            )

        # =====================================================================
        # Phase 2.75: Discriminator Warmup (NEW - critical for Phase 3!)
        # =====================================================================
        if iterations_disc_warmup > 0:
            self.train_discriminator_warmup(
                train_dataloader, iterations_disc_warmup,
                log_every=100, checkpoint_dir=checkpoint_dir
            )

        # =====================================================================
        # Phase 3: Joint Adversarial
        # =====================================================================
        self.train_joint(
            train_dataloader, val_dataloader, iterations_joint,
            validate_every, checkpoint_dir
        )

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Total iterations: {total_iters:,}")
        print(f"")
        print(f"Checkpoints saved:")
        print(f"  Best model: {checkpoint_dir}/model_best_v4.pt")
        print(f"  Phase finals: phase1_final.pt, phase2_final.pt, phase2_5_final.pt, phase2_75_final.pt, phase3_final.pt")
        print("=" * 70)

        # Flush/close logging
        if self.tb_logger:
            self.tb_logger.flush()  # Don't close - managed by caller
        elif self.writer:
            self.writer.close()

    # =========================================================================
    # CHECKPOINTING
    # =========================================================================

    def save_checkpoint(self, path, include_ema=True):
        """
        Save model checkpoint with optional EMA weights.

        Args:
            path: Path to save checkpoint
            include_ema: Whether to include EMA weights (default: True)
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
        }

        # Include EMA state if available and enabled
        if include_ema and self.ema_decay > 0:
            checkpoint['ema_generator'] = self.ema_generator.state_dict()
            checkpoint['ema_supervisor'] = self.ema_supervisor.state_dict()
            checkpoint['ema_recovery'] = self.ema_recovery.state_dict()

        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}{' (with EMA)' if include_ema and self.ema_decay > 0 else ''}")

    def load_checkpoint(self, path, load_ema=True):
        """
        Load model checkpoint with optional EMA weights.

        Args:
            path: Path to load checkpoint from
            load_ema: Whether to load EMA weights if available (default: True)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)

        # Load EMA state if available
        if load_ema and 'ema_generator' in checkpoint:
            self.ema_generator.load_state_dict(checkpoint['ema_generator'])
            self.ema_supervisor.load_state_dict(checkpoint['ema_supervisor'])
            self.ema_recovery.load_state_dict(checkpoint['ema_recovery'])
            print(f"  Checkpoint loaded: {path} (with EMA)")
        else:
            print(f"  Checkpoint loaded: {path}")

    def apply_ema_weights(self):
        """Apply EMA weights for inference. Call restore_training_weights() when done."""
        self.ema_generator.apply()
        self.ema_supervisor.apply()
        self.ema_recovery.apply()

    def restore_training_weights(self):
        """Restore original training weights after EMA inference."""
        self.ema_generator.restore()
        self.ema_supervisor.restore()
        self.ema_recovery.restore()


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == '__main__':
    print("Testing TimeGAN V4 Trainer (Iteration-Based)...")
    print("=" * 60)

    from config_model_v4 import MODEL_CONFIG_V4

    # Create model
    model = TimeGANV4(MODEL_CONFIG_V4)

    # Training config
    train_config = {
        'lr_autoencoder': 1e-3,
        'lr_supervisor': 1e-3,
        'lr_generator': 1e-4,
        'lr_discriminator': 1e-4,
        'lambda_recon': 10.0,
        'lambda_supervised': 1.0,
        'lambda_gp': 10.0,
        'lambda_var': 5.0,
        'n_critic': 5,
    }

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = TimeGANTrainerV4(model, train_config, device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy data
    batch_size = 4
    seq_len = 100

    x = torch.randn(batch_size, seq_len, 8).clamp(-1, 1)
    condition = torch.rand(batch_size, 1) * 2 - 1
    lengths = torch.randint(50, seq_len, (batch_size,))

    x = x.to(device)
    condition = condition.to(device)
    lengths = lengths.to(device)

    # Test single steps
    print("\nTesting single training steps...")

    ae_result = trainer.train_autoencoder_step(x, lengths)
    print(f"  Autoencoder step - Total: {ae_result['total_loss']:.6f}, "
          f"Recon: {ae_result['recon_loss']:.6f}, Jerk: {ae_result['jerk_loss']:.6f}, "
          f"Moments: {ae_result['moments_loss']:.6f}, Autocorr: {ae_result['autocorr_loss']:.6f}")

    sup_loss = trainer.train_supervised_step(x, lengths)
    print(f"  Supervised step loss: {sup_loss:.6f}")

    d_loss, gp = trainer.train_discriminator_step(x, condition, lengths)
    print(f"  Discriminator step - D_loss: {d_loss:.4f}, GP: {gp:.4f}")

    g_losses = trainer.train_generator_step(x, condition, lengths)
    print(f"  Generator step - G: {g_losses['g_loss']:.4f}, Sup: {g_losses['sup_loss']:.6f}, "
          f"Recon: {g_losses['recon_loss']:.6f}, Var: {g_losses['var_loss']:.6f}, Cond: {g_losses['cond_loss']:.6f}")
    print(f"  Style losses - Jerk: {g_losses['jerk_loss']:.6f}, Moments: {g_losses['moments_loss']:.6f}, "
          f"Autocorr: {g_losses['autocorr_loss']:.6f}")

    # Test variance loss
    mask = trainer._create_mask(lengths, seq_len)
    x_fake = torch.randn_like(x).clamp(-1, 1)
    var_loss = trainer.compute_variance_loss(x, x_fake, mask)
    print(f"  Variance loss: {var_loss:.6f}")

    # Test style losses
    jerk_loss = trainer.compute_jerk_loss(x, x_fake, mask)
    print(f"  Jerk loss: {jerk_loss:.6f}")

    moments_loss = trainer.compute_higher_moments_loss(x, x_fake, mask)
    print(f"  Higher moments loss: {moments_loss:.6f}")

    autocorr_loss = trainer.compute_autocorr_loss(x, x_fake, mask)
    print(f"  Autocorrelation loss: {autocorr_loss:.6f}")

    print("\nTrainer V4 (Iteration-Based) test PASSED!")
    print("=" * 60)
