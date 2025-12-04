"""
TensorBoard Logger for V4 Training

Comprehensive logging system for cross-session comparison and hyperparameter tuning.

Features:
- Hyperparameter logging for run comparison (HPARAMS tab)
- Custom scalar layouts for organized dashboards
- Distribution tracking (histograms of weights, gradients, latents)
- Trajectory visualization (real vs generated plots)
- Global step tracking across phases
- Run metadata (git commit, system info, config summaries)
- Feature-wise metric breakdowns
"""

import os
import json
import time
import socket
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class TensorBoardLogger:
    """
    Enhanced TensorBoard logger for training comparison and analysis.

    Usage:
        logger = TensorBoardLogger(
            log_dir='runs',
            experiment_name='baseline_v4',
            config=full_config_dict
        )

        # During training
        logger.log_scalars('Phase1', {'loss': 0.5, 'val_loss': 0.6}, step=100)
        logger.log_histograms('Phase1', model, step=100)
        logger.log_trajectory_comparison(real_traj, fake_traj, step=100)

        # At end of training
        logger.log_final_metrics({'best_v4_score': 0.123})
        logger.close()
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        comment: str = '',
        flush_secs: int = 30,
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name for this experiment (auto-generated if None)
            config: Full configuration dictionary for hyperparameter logging
            comment: Additional comment to append to run name
            flush_secs: How often to flush to disk (seconds)
        """
        self.config = config or {}
        self.experiment_name = experiment_name or self._generate_experiment_name()
        self.global_step = 0
        self.phase_start_steps = {}  # Track where each phase starts

        # Create run directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{self.experiment_name}_{timestamp}"
        if comment:
            run_name += f"_{comment}"

        self.run_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(
            log_dir=self.run_dir,
            flush_secs=flush_secs
        )

        # Store metadata
        self.start_time = time.time()
        self.metadata = self._collect_metadata()

        # Log initial setup
        self._log_run_metadata()
        self._log_config_text()
        self._setup_custom_layouts()

        print(f"\n{'='*70}")
        print(f"TensorBoard Logger Initialized")
        print(f"{'='*70}")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Run dir:    {self.run_dir}")
        print(f"  View with:  tensorboard --logdir {log_dir}")

    def _fig_to_tensor(self, fig) -> torch.Tensor:
        """
        Convert matplotlib figure to tensor for TensorBoard.
        Compatible with both old and new matplotlib versions.
        """
        try:
            # Try newer matplotlib API first (3.8+)
            img_array = np.array(fig.canvas.buffer_rgba())
            # Convert RGBA to RGB
            img_array = img_array[:, :, :3]
        except AttributeError:
            try:
                # Try older matplotlib API
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Fallback: save to buffer and reload
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                from PIL import Image
                img = Image.open(buf)
                img_array = np.array(img.convert('RGB'))
                buf.close()

        img_tensor = torch.from_numpy(img_array.copy()).permute(2, 0, 1)
        return img_tensor

    def _generate_experiment_name(self) -> str:
        """Generate default experiment name from config."""
        if not self.config:
            return 'experiment'

        # Extract key parameters for name
        training = self.config.get('training', {})
        model = self.config.get('model', {})

        parts = []

        # Add key hyperparameters to name
        if 'lr_generator' in training:
            lr = training['lr_generator']
            parts.append(f"lr{lr:.0e}".replace('e-0', 'e-'))

        if 'batch_size' in training:
            parts.append(f"bs{training['batch_size']}")

        if 'hidden_dim' in model:
            parts.append(f"h{model['hidden_dim']}")

        if 'latent_dim' in model:
            parts.append(f"z{model['latent_dim']}")

        return '_'.join(parts) if parts else 'experiment'

    def _collect_metadata(self) -> Dict:
        """Collect run metadata for logging."""
        metadata = {
            'start_time': datetime.now().isoformat(),
            'hostname': socket.gethostname(),
            'python_version': self._get_python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'git_commit': self._get_git_commit(),
            'git_branch': self._get_git_branch(),
            'git_dirty': self._is_git_dirty(),
        }
        return metadata

    def _get_python_version(self) -> str:
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_git_commit(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()[:8]
        except:
            return None

    def _get_git_branch(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            return None

    def _is_git_dirty(self) -> Optional[bool]:
        try:
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            return len(status) > 0
        except:
            return None

    def _log_run_metadata(self):
        """Log run metadata as text."""
        text = "## Run Metadata\n\n"
        for key, value in self.metadata.items():
            text += f"- **{key}**: {value}\n"
        self.writer.add_text('Run/Metadata', text, 0)

    def _log_config_text(self):
        """Log full configuration as formatted text."""
        if not self.config:
            return

        # Log each config section
        for section_name, section_config in self.config.items():
            if isinstance(section_config, dict):
                text = f"## {section_name.upper()} Configuration\n\n```json\n"
                text += json.dumps(section_config, indent=2, default=str)
                text += "\n```"
                self.writer.add_text(f'Config/{section_name}', text, 0)

        # Log full config as single text block
        full_text = "## Full Configuration\n\n```json\n"
        full_text += json.dumps(self.config, indent=2, default=str)
        full_text += "\n```"
        self.writer.add_text('Config/full', full_text, 0)

    def _setup_custom_layouts(self):
        """Setup custom scalar layouts for organized comparison."""
        layout = {
            "Training Overview": {
                "V4 Score (↓ better)": [
                    "Multiline",
                    ["Phase1/v4_score", "Phase2/v4_score", "Phase3/v4_score", "Global/v4_score"]
                ],
                "Learning Rates": [
                    "Multiline",
                    ["Phase1/lr", "Phase2/lr", "Phase3/lr_g", "Phase3/lr_d"]
                ],
            },
            "Losses - Phase 1 (Autoencoder)": {
                "Total Loss": ["Multiline", ["Phase1/train_loss"]],
                "Components": [
                    "Multiline",
                    ["Phase1/recon_loss", "Phase1/jerk_loss", "Phase1/moments_loss", "Phase1/autocorr_loss"]
                ],
                "Validation": ["Multiline", ["Phase1/val_recon", "Phase1/v4_score"]],
            },
            "Losses - Phase 3 (Adversarial)": {
                "GAN Losses": ["Multiline", ["Phase3/d_loss", "Phase3/g_loss", "Phase3/gp"]],
                "Style Losses": [
                    "Multiline",
                    ["Phase3/var_loss", "Phase3/jerk_loss", "Phase3/moments_loss", "Phase3/autocorr_loss"]
                ],
                "Reconstruction": ["Multiline", ["Phase3/recon_loss", "Phase3/sup_loss", "Phase3/cond_loss"]],
                "Validation": [
                    "Multiline",
                    ["Phase3/val_recon", "Phase3/val_var", "Phase3/val_cond", "Phase3/v4_score"]
                ],
            },
            "Gradient Health": {
                "Gradient Norms": [
                    "Multiline",
                    ["Gradients/embedder", "Gradients/recovery", "Gradients/supervisor",
                     "Gradients/generator", "Gradients/discriminator"]
                ],
            },
            "Latent Space": {
                "Statistics": [
                    "Multiline",
                    ["Latent/h_mean", "Latent/h_std", "Latent/h_fake_mean", "Latent/h_fake_std"]
                ],
            },
            "Feature-wise Reconstruction": {
                "Position": ["Multiline", ["Features/recon_dx", "Features/recon_dy"]],
                "Motion": ["Multiline", ["Features/recon_speed", "Features/recon_accel"]],
                "Direction": ["Multiline", ["Features/recon_sin_h", "Features/recon_cos_h"]],
                "Style": ["Multiline", ["Features/recon_ang_vel", "Features/recon_dt"]],
            },
        }
        self.writer.add_custom_scalars(layout)

    # =========================================================================
    # HYPERPARAMETER LOGGING
    # =========================================================================

    def log_hyperparameters(self, final_metrics: Optional[Dict[str, float]] = None):
        """
        Log hyperparameters for HPARAMS tab comparison.

        Call this at the END of training with final metrics for proper comparison.

        Args:
            final_metrics: Dictionary of final metric values (e.g., best_v4_score)
        """
        if not self.config:
            return

        # Flatten config for hparams
        hparam_dict = self._flatten_config(self.config)

        # Default metrics if not provided
        metric_dict = final_metrics or {
            'hparam/best_v4_score': 0.0,
            'hparam/final_recon': 0.0,
            'hparam/final_var': 0.0,
            'hparam/training_time_min': (time.time() - self.start_time) / 60,
        }

        # Ensure metric keys have hparam/ prefix
        metric_dict = {
            k if k.startswith('hparam/') else f'hparam/{k}': v
            for k, v in metric_dict.items()
        }

        self.writer.add_hparams(hparam_dict, metric_dict)

    def _flatten_config(self, config: Dict, prefix: str = '') -> Dict:
        """Flatten nested config dict for hparams."""
        flat = {}

        # Key hyperparameters to include (avoid cluttering hparams view)
        important_keys = {
            # Training
            'batch_size', 'lr_autoencoder', 'lr_supervisor', 'lr_generator',
            'lr_discriminator', 'lr_er_ratio',
            'phase1_iterations', 'phase2_iterations', 'phase2_5_iterations',
            'phase2_75_iterations', 'phase3_iterations',
            'lambda_gp', 'n_critic', 'lambda_recon', 'lambda_supervised',
            'lambda_var', 'lambda_cond', 'max_grad_norm', 'ema_decay',
            'warmup_steps', 'lr_patience', 'lr_factor',
            # Model
            'feature_dim', 'hidden_dim', 'latent_dim', 'num_layers',
            'gen_dropout', 'disc_dropout', 'disc_bilstm_hidden',
            # Style
            'lambda_jerk', 'lambda_higher_moments', 'lambda_autocorr',
            'v4_score_cond_weight', 'v4_score_var_weight',
            'v4_score_jerk_weight', 'v4_score_moments_weight', 'v4_score_autocorr_weight',
        }

        for key, value in config.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(self._flatten_config(value, f"{full_key}/"))
            elif key in important_keys:
                # Convert to hparams-compatible type
                if isinstance(value, (int, float, str, bool)):
                    flat[full_key] = value
                elif isinstance(value, (list, tuple)):
                    # Convert lists to string representation
                    flat[full_key] = str(value)

        return flat

    # =========================================================================
    # SCALAR LOGGING
    # =========================================================================

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a single scalar value."""
        step = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, prefix: str, scalars: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple scalars with a common prefix.

        Args:
            prefix: Prefix for all scalar names (e.g., 'Phase1')
            scalars: Dictionary of name -> value
            step: Step number (uses global_step if None)
        """
        step = step if step is not None else self.global_step
        for name, value in scalars.items():
            if value is not None and not np.isnan(value):
                self.writer.add_scalar(f'{prefix}/{name}', value, step)

    def log_scalars_global(self, scalars: Dict[str, float], step: Optional[int] = None):
        """Log scalars with Global prefix for cross-phase comparison."""
        self.log_scalars('Global', scalars, step)

    # =========================================================================
    # DISTRIBUTION / HISTOGRAM LOGGING
    # =========================================================================

    def log_histograms(
        self,
        prefix: str,
        model: nn.Module,
        step: Optional[int] = None,
        log_weights: bool = True,
        log_gradients: bool = True
    ):
        """
        Log weight and gradient histograms for model components.

        Args:
            prefix: Prefix for histogram names
            model: The TimeGAN model
            step: Step number
            log_weights: Whether to log weight distributions
            log_gradients: Whether to log gradient distributions
        """
        step = step if step is not None else self.global_step

        # Component name mapping - safely get attributes
        components = {}
        for comp_name in ['embedder', 'recovery', 'supervisor', 'generator', 'discriminator']:
            comp = getattr(model, comp_name, None)
            if comp is not None:
                components[comp_name] = comp

        for comp_name, component in components.items():
            try:
                # Aggregate all parameters for this component
                all_weights = []
                all_grads = []

                for name, param in component.named_parameters():
                    if param.data is not None and log_weights:
                        # Detach and move to CPU for histogram
                        data = param.data.detach().cpu().flatten()
                        # Skip NaN/Inf values
                        if not (torch.isnan(data).any() or torch.isinf(data).any()):
                            all_weights.append(data)

                    if param.grad is not None and log_gradients:
                        grad = param.grad.detach().cpu().flatten()
                        # Skip NaN/Inf values
                        if not (torch.isnan(grad).any() or torch.isinf(grad).any()):
                            all_grads.append(grad)

                # Log aggregated histograms
                if all_weights:
                    weights_tensor = torch.cat(all_weights)
                    if weights_tensor.numel() > 0:
                        self.writer.add_histogram(
                            f'{prefix}/weights/{comp_name}',
                            weights_tensor,
                            step
                        )

                if all_grads:
                    grads_tensor = torch.cat(all_grads)
                    if grads_tensor.numel() > 0:
                        self.writer.add_histogram(
                            f'{prefix}/gradients/{comp_name}',
                            grads_tensor,
                            step
                        )
            except Exception as e:
                # Silently skip failed histogram logging
                pass

    def log_latent_distributions(
        self,
        h_real: torch.Tensor,
        h_fake: Optional[torch.Tensor] = None,
        step: Optional[int] = None
    ):
        """
        Log latent space distributions.

        Args:
            h_real: Real latent embeddings [batch, seq, latent_dim]
            h_fake: Generated latent sequences [batch, seq, latent_dim]
            step: Step number
        """
        step = step if step is not None else self.global_step

        try:
            # Flatten for histogram (move to CPU for TensorBoard)
            h_real_flat = h_real.detach().cpu().flatten()

            # Skip if contains NaN or Inf (training exploded)
            if torch.isnan(h_real_flat).any() or torch.isinf(h_real_flat).any():
                return

            # Skip if empty
            if h_real_flat.numel() == 0:
                return

            self.writer.add_histogram('Latent/h_real', h_real_flat, step)

            # Log statistics
            self.log_scalars('Latent', {
                'h_mean': h_real_flat.mean().item(),
                'h_std': h_real_flat.std().item(),
                'h_min': h_real_flat.min().item(),
                'h_max': h_real_flat.max().item(),
            }, step)

            if h_fake is not None:
                h_fake_flat = h_fake.detach().cpu().flatten()

                # Skip if contains NaN or Inf
                if torch.isnan(h_fake_flat).any() or torch.isinf(h_fake_flat).any():
                    return

                if h_fake_flat.numel() == 0:
                    return

                self.writer.add_histogram('Latent/h_fake', h_fake_flat, step)

                self.log_scalars('Latent', {
                    'h_fake_mean': h_fake_flat.mean().item(),
                    'h_fake_std': h_fake_flat.std().item(),
                    'std_ratio': h_fake_flat.std().item() / (h_real_flat.std().item() + 1e-8),
                }, step)
        except Exception as e:
            # Silently skip if latent logging fails
            pass

    def log_feature_distributions(
        self,
        real_data: torch.Tensor,
        fake_data: Optional[torch.Tensor] = None,
        feature_names: List[str] = None,
        step: Optional[int] = None
    ):
        """
        Log per-feature distributions.

        Args:
            real_data: Real trajectories [batch, seq, features]
            fake_data: Generated trajectories [batch, seq, features]
            feature_names: Names for each feature dimension
            step: Step number
        """
        step = step if step is not None else self.global_step
        feature_names = feature_names or ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']

        for i, name in enumerate(feature_names):
            if i < real_data.shape[-1]:
                real_feat = real_data[..., i].detach().flatten()
                self.writer.add_histogram(f'Features/real_{name}', real_feat, step)

                if fake_data is not None and i < fake_data.shape[-1]:
                    fake_feat = fake_data[..., i].detach().flatten()
                    self.writer.add_histogram(f'Features/fake_{name}', fake_feat, step)

    # =========================================================================
    # FEATURE-WISE METRICS
    # =========================================================================

    def log_feature_reconstruction_errors(
        self,
        real: torch.Tensor,
        reconstructed: torch.Tensor,
        feature_names: List[str] = None,
        step: Optional[int] = None
    ):
        """
        Log per-feature reconstruction errors.

        Args:
            real: Original data [batch, seq, features]
            reconstructed: Reconstructed data [batch, seq, features]
            feature_names: Names for each feature
            step: Step number
        """
        step = step if step is not None else self.global_step
        feature_names = feature_names or ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']

        # Compute per-feature MSE
        mse_per_feature = ((real - reconstructed) ** 2).mean(dim=(0, 1))

        errors = {}
        for i, name in enumerate(feature_names):
            if i < mse_per_feature.shape[0]:
                errors[f'recon_{name}'] = mse_per_feature[i].item()

        self.log_scalars('Features', errors, step)
        return errors

    # =========================================================================
    # TRAJECTORY VISUALIZATION
    # =========================================================================

    def log_trajectory_comparison(
        self,
        real_trajectory: torch.Tensor,
        fake_trajectory: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        num_samples: int = 4,
        tag: str = 'Trajectories/comparison'
    ):
        """
        Log visual comparison of real vs generated trajectories.

        Args:
            real_trajectory: Real trajectory batch [batch, seq, features]
            fake_trajectory: Generated trajectory batch [batch, seq, features]
            lengths: Actual sequence lengths [batch] - IMPORTANT for masking padding!
            step: Step number
            num_samples: Number of trajectory pairs to plot
            tag: TensorBoard tag for the image
        """
        step = step if step is not None else self.global_step

        # Create figure with subplots
        fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)

        real_np = real_trajectory.detach().cpu().numpy()
        fake_np = fake_trajectory.detach().cpu().numpy()

        for i in range(min(num_samples, real_np.shape[0], fake_np.shape[0])):
            # Get actual sequence length (mask padding!)
            if lengths is not None:
                seq_len = int(lengths[i].item())
            else:
                seq_len = real_np.shape[1]

            # Extract dx, dy UP TO actual length (ignore padding)
            real_dx, real_dy = real_np[i, :seq_len, 0], real_np[i, :seq_len, 1]
            fake_dx, fake_dy = fake_np[i, :seq_len, 0], fake_np[i, :seq_len, 1]

            # Cumsum to get absolute positions
            real_x, real_y = np.cumsum(real_dx), np.cumsum(real_dy)
            fake_x, fake_y = np.cumsum(fake_dx), np.cumsum(fake_dy)

            # Plot real
            axes[0, i].plot(real_x, real_y, 'b-', linewidth=1.5, alpha=0.8)
            axes[0, i].scatter([real_x[0]], [real_y[0]], c='green', s=50, zorder=5, label='Start')
            axes[0, i].scatter([real_x[-1]], [real_y[-1]], c='red', s=50, zorder=5, label='End')
            axes[0, i].set_title(f'Real #{i+1} (len={seq_len})')
            axes[0, i].set_aspect('equal')
            axes[0, i].grid(True, alpha=0.3)

            # Plot fake
            axes[1, i].plot(fake_x, fake_y, 'r-', linewidth=1.5, alpha=0.8)
            axes[1, i].scatter([fake_x[0]], [fake_y[0]], c='green', s=50, zorder=5)
            axes[1, i].scatter([fake_x[-1]], [fake_y[-1]], c='red', s=50, zorder=5)
            axes[1, i].set_title(f'Generated #{i+1}')
            axes[1, i].set_aspect('equal')
            axes[1, i].grid(True, alpha=0.3)

        axes[0, 0].legend(loc='upper left', fontsize=8)

        plt.suptitle(f'Trajectory Comparison (Step {step})', fontsize=14)
        plt.tight_layout()

        # Convert to tensor for TensorBoard (compatible with newer matplotlib)
        fig.canvas.draw()
        img_tensor = self._fig_to_tensor(fig)

        self.writer.add_image(tag, img_tensor, step)
        plt.close(fig)

    def log_velocity_profile_comparison(
        self,
        real_trajectory: torch.Tensor,
        fake_trajectory: torch.Tensor,
        step: Optional[int] = None,
        tag: str = 'Trajectories/velocity_profiles'
    ):
        """
        Log velocity/acceleration profile comparisons.

        Args:
            real_trajectory: Real trajectory [batch, seq, features]
            fake_trajectory: Generated trajectory [batch, seq, features]
            step: Step number
            tag: TensorBoard tag
        """
        step = step if step is not None else self.global_step

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        real_np = real_trajectory[0].detach().cpu().numpy()  # Take first sample
        fake_np = fake_trajectory[0].detach().cpu().numpy()

        # Feature indices: speed=2, accel=3, ang_vel=6, dt=7
        features = [
            (2, 'Speed', 'pixels/sec'),
            (3, 'Acceleration', 'pixels/sec²'),
            (6, 'Angular Velocity', 'rad/sec'),
        ]

        time_steps = np.arange(real_np.shape[0])

        for col, (idx, name, unit) in enumerate(features):
            # Real
            axes[0, col].plot(time_steps, real_np[:, idx], 'b-', alpha=0.8, label='Real')
            axes[0, col].set_title(f'Real {name}')
            axes[0, col].set_xlabel('Time Step')
            axes[0, col].set_ylabel(unit)
            axes[0, col].grid(True, alpha=0.3)

            # Fake
            axes[1, col].plot(time_steps, fake_np[:, idx], 'r-', alpha=0.8, label='Generated')
            axes[1, col].set_title(f'Generated {name}')
            axes[1, col].set_xlabel('Time Step')
            axes[1, col].set_ylabel(unit)
            axes[1, col].grid(True, alpha=0.3)

        plt.suptitle(f'Velocity Profile Comparison (Step {step})', fontsize=14)
        plt.tight_layout()

        # Convert to tensor (compatible with newer matplotlib)
        fig.canvas.draw()
        img_tensor = self._fig_to_tensor(fig)

        self.writer.add_image(tag, img_tensor, step)
        plt.close(fig)

    def log_distribution_comparison(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        step: Optional[int] = None,
        tag: str = 'Distributions/style_features'
    ):
        """
        Log distribution comparison plots for style features.

        Args:
            real_data: Real trajectories [batch, seq, features]
            fake_data: Generated trajectories [batch, seq, features]
            step: Step number
            tag: TensorBoard tag
        """
        step = step if step is not None else self.global_step

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        real_np = real_data.detach().cpu().numpy().reshape(-1, real_data.shape[-1])
        fake_np = fake_data.detach().cpu().numpy().reshape(-1, fake_data.shape[-1])

        feature_names = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']

        for i, name in enumerate(feature_names):
            row, col = i // 4, i % 4

            # Plot histograms
            axes[row, col].hist(real_np[:, i], bins=50, alpha=0.6, label='Real', color='blue', density=True)
            axes[row, col].hist(fake_np[:, i], bins=50, alpha=0.6, label='Generated', color='red', density=True)
            axes[row, col].set_title(name)
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(True, alpha=0.3)

        plt.suptitle(f'Feature Distribution Comparison (Step {step})', fontsize=14)
        plt.tight_layout()

        # Convert to tensor (compatible with newer matplotlib)
        fig.canvas.draw()
        img_tensor = self._fig_to_tensor(fig)

        self.writer.add_image(tag, img_tensor, step)
        plt.close(fig)

    # =========================================================================
    # EMBEDDING PROJECTOR (t-SNE/PCA/UMAP visualization)
    # =========================================================================

    def log_embeddings(
        self,
        h_real: torch.Tensor,
        h_fake: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        max_samples: int = 500,
        tag_prefix: str = 'Embeddings'
    ):
        """
        Log embeddings for Projector visualization (t-SNE/PCA/UMAP).

        This is CRITICAL for GANs - lets you visually see if fake latents
        overlap with real latents, detect mode collapse, and understand
        the latent space structure.

        Args:
            h_real: Real latent embeddings [batch, seq, latent_dim] or [N, latent_dim]
            h_fake: Generated latent sequences [batch, seq, latent_dim] or [N, latent_dim]
            conditions: Distance conditions for coloring points [batch] or [N]
            step: Step number
            max_samples: Maximum samples to log (for performance)
            tag_prefix: Prefix for embedding tags
        """
        step = step if step is not None else self.global_step

        try:
            # Flatten if 3D (batch, seq, latent_dim) -> (batch*seq, latent_dim)
            if h_real.dim() == 3:
                batch_size, seq_len, latent_dim = h_real.shape
                h_real_flat = h_real.reshape(-1, latent_dim)
            else:
                h_real_flat = h_real
                batch_size = h_real.shape[0]
                seq_len = 1

            # Subsample if too many points
            n_real = h_real_flat.shape[0]
            if n_real > max_samples:
                indices = torch.randperm(n_real)[:max_samples]
                h_real_flat = h_real_flat[indices]

            # Skip if NaN/Inf
            if torch.isnan(h_real_flat).any() or torch.isinf(h_real_flat).any():
                return

            # Create metadata (labels for each point)
            metadata = ['real'] * h_real_flat.shape[0]

            # Add condition info if available
            if conditions is not None:
                cond_flat = conditions.detach().cpu()
                if cond_flat.dim() > 1:
                    cond_flat = cond_flat.squeeze()
                # Repeat condition for each timestep
                if seq_len > 1:
                    cond_flat = cond_flat.repeat_interleave(seq_len)
                if len(cond_flat) > max_samples:
                    cond_flat = cond_flat[indices] if n_real > max_samples else cond_flat[:max_samples]
                metadata = [f'real_d{c.item():.2f}' for c in cond_flat[:h_real_flat.shape[0]]]

            # Log real embeddings
            self.writer.add_embedding(
                h_real_flat.detach().cpu(),
                metadata=metadata,
                tag=f'{tag_prefix}/real',
                global_step=step
            )

            # Log fake embeddings if provided
            if h_fake is not None:
                if h_fake.dim() == 3:
                    h_fake_flat = h_fake.reshape(-1, h_fake.shape[-1])
                else:
                    h_fake_flat = h_fake

                n_fake = h_fake_flat.shape[0]
                if n_fake > max_samples:
                    indices_fake = torch.randperm(n_fake)[:max_samples]
                    h_fake_flat = h_fake_flat[indices_fake]

                if torch.isnan(h_fake_flat).any() or torch.isinf(h_fake_flat).any():
                    return

                metadata_fake = ['fake'] * h_fake_flat.shape[0]

                self.writer.add_embedding(
                    h_fake_flat.detach().cpu(),
                    metadata=metadata_fake,
                    tag=f'{tag_prefix}/fake',
                    global_step=step
                )

                # Combined view: real + fake together for direct comparison
                combined = torch.cat([h_real_flat.detach().cpu(), h_fake_flat.detach().cpu()], dim=0)
                combined_metadata = metadata + metadata_fake

                self.writer.add_embedding(
                    combined,
                    metadata=combined_metadata,
                    tag=f'{tag_prefix}/combined',
                    global_step=step
                )

        except Exception as e:
            # Silently skip if embedding logging fails
            print(f"Warning: Embedding logging failed: {e}")

    # =========================================================================
    # MODEL GRAPH VISUALIZATION
    # =========================================================================

    def log_model_graph(
        self,
        model: nn.Module,
        sample_input: Tuple[torch.Tensor, ...],
        model_name: str = 'TimeGAN'
    ):
        """
        Log model architecture graph for visualization.

        Args:
            model: The model to visualize
            sample_input: Tuple of sample inputs (x, condition, lengths)
            model_name: Name for the graph
        """
        try:
            self.writer.add_graph(model, sample_input)
            print(f"  Model graph logged: {model_name}")
        except Exception as e:
            # Graph logging can fail for complex models
            print(f"  Warning: Could not log model graph: {e}")

    def log_component_graphs(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 2,
        seq_len: int = 100,
        feature_dim: int = 8,
        latent_dim: int = 48
    ):
        """
        Log individual component graphs (embedder, generator, etc.).

        Args:
            model: The TimeGAN model
            device: Device to create dummy inputs on
            batch_size: Batch size for dummy inputs
            seq_len: Sequence length for dummy inputs
            feature_dim: Feature dimension (8 for V4)
            latent_dim: Latent dimension
        """
        # Dummy inputs
        x = torch.randn(batch_size, seq_len, feature_dim, device=device)
        condition = torch.randn(batch_size, 1, device=device)
        lengths = torch.tensor([seq_len] * batch_size)
        z = torch.randn(batch_size, seq_len, latent_dim, device=device)
        h = torch.randn(batch_size, seq_len, latent_dim, device=device)

        components = [
            ('Embedder', model.embedder, (x, lengths)),
            ('Recovery', model.recovery, (h, lengths)),
            ('Supervisor', model.supervisor, (h, lengths)),
            ('Generator', model.generator, (z, condition, lengths)),
            ('Discriminator', model.discriminator, (x, condition, lengths)),
        ]

        for name, component, inputs in components:
            try:
                self.writer.add_graph(component, inputs)
                print(f"  {name} graph logged")
            except Exception as e:
                print(f"  Warning: Could not log {name} graph: {e}")

    # =========================================================================
    # TRAJECTORY PATH VISUALIZATION (Enhanced)
    # =========================================================================

    def log_trajectory_paths(
        self,
        real_trajectories: torch.Tensor,
        fake_trajectories: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        num_samples: int = 6,
        tag: str = 'Paths/comparison'
    ):
        """
        Enhanced trajectory path visualization with multiple views.

        Args:
            real_trajectories: Real trajectory batch [batch, seq, features]
            fake_trajectories: Generated trajectory batch [batch, seq, features]
            conditions: Distance conditions [batch, 1]
            lengths: Actual sequence lengths [batch]
            step: Step number
            num_samples: Number of trajectory pairs to plot
            tag: TensorBoard tag for the image
        """
        step = step if step is not None else self.global_step

        n_samples = min(num_samples, real_trajectories.shape[0], fake_trajectories.shape[0])

        # Create figure: 3 rows (real, fake, overlay) x n_samples columns
        fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12))
        if n_samples == 1:
            axes = axes.reshape(3, 1)

        real_np = real_trajectories.detach().cpu().numpy()
        fake_np = fake_trajectories.detach().cpu().numpy()

        for i in range(n_samples):
            # Get actual length if provided
            seq_len = int(lengths[i].item()) if lengths is not None else real_np.shape[1]

            # Extract dx, dy and compute cumulative position
            real_dx, real_dy = real_np[i, :seq_len, 0], real_np[i, :seq_len, 1]
            fake_dx, fake_dy = fake_np[i, :seq_len, 0], fake_np[i, :seq_len, 1]

            # Cumsum to get absolute positions
            real_x, real_y = np.cumsum(real_dx), np.cumsum(real_dy)
            fake_x, fake_y = np.cumsum(fake_dx), np.cumsum(fake_dy)

            # Get condition if available
            cond_str = f" (d={conditions[i].item():.2f})" if conditions is not None else ""

            # Row 0: Real trajectory
            axes[0, i].plot(real_x, real_y, 'b-', linewidth=1.5, alpha=0.8)
            axes[0, i].scatter([real_x[0]], [real_y[0]], c='green', s=80, zorder=5, marker='o')
            axes[0, i].scatter([real_x[-1]], [real_y[-1]], c='red', s=80, zorder=5, marker='x')
            axes[0, i].set_title(f'Real #{i+1}{cond_str}', fontsize=10)
            axes[0, i].set_aspect('equal')
            axes[0, i].grid(True, alpha=0.3)

            # Row 1: Fake trajectory
            axes[1, i].plot(fake_x, fake_y, 'r-', linewidth=1.5, alpha=0.8)
            axes[1, i].scatter([fake_x[0]], [fake_y[0]], c='green', s=80, zorder=5, marker='o')
            axes[1, i].scatter([fake_x[-1]], [fake_y[-1]], c='red', s=80, zorder=5, marker='x')
            axes[1, i].set_title(f'Generated #{i+1}{cond_str}', fontsize=10)
            axes[1, i].set_aspect('equal')
            axes[1, i].grid(True, alpha=0.3)

            # Row 2: Overlay comparison
            axes[2, i].plot(real_x, real_y, 'b-', linewidth=1.5, alpha=0.7, label='Real')
            axes[2, i].plot(fake_x, fake_y, 'r--', linewidth=1.5, alpha=0.7, label='Generated')
            axes[2, i].scatter([real_x[0], fake_x[0]], [real_y[0], fake_y[0]], c='green', s=60, zorder=5)
            axes[2, i].scatter([real_x[-1], fake_x[-1]], [real_y[-1], fake_y[-1]], c='red', s=60, zorder=5)
            axes[2, i].set_title(f'Overlay #{i+1}', fontsize=10)
            axes[2, i].set_aspect('equal')
            axes[2, i].grid(True, alpha=0.3)
            if i == 0:
                axes[2, i].legend(loc='upper left', fontsize=8)

        # Row labels
        axes[0, 0].set_ylabel('Real', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Generated', fontsize=12, fontweight='bold')
        axes[2, 0].set_ylabel('Overlay', fontsize=12, fontweight='bold')

        plt.suptitle(f'Trajectory Path Comparison (Step {step})', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Convert to tensor
        fig.canvas.draw()
        img_tensor = self._fig_to_tensor(fig)
        self.writer.add_image(tag, img_tensor, step)
        plt.close(fig)

    def log_speed_heatmap(
        self,
        real_trajectories: torch.Tensor,
        fake_trajectories: torch.Tensor,
        step: Optional[int] = None,
        tag: str = 'Paths/speed_heatmap'
    ):
        """
        Log trajectory paths colored by speed.

        Args:
            real_trajectories: Real trajectory batch [batch, seq, features]
            fake_trajectories: Generated trajectory batch [batch, seq, features]
            step: Step number
            tag: TensorBoard tag
        """
        step = step if step is not None else self.global_step

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, data, title in [(axes[0], real_trajectories, 'Real'),
                                 (axes[1], fake_trajectories, 'Generated')]:
            data_np = data[0].detach().cpu().numpy()  # First sample

            dx, dy = data_np[:, 0], data_np[:, 1]
            x, y = np.cumsum(dx), np.cumsum(dy)
            speed = data_np[:, 2]  # Speed is feature index 2

            # Create line segments colored by speed
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            from matplotlib.collections import LineCollection
            norm = plt.Normalize(speed.min(), speed.max())
            lc = LineCollection(segments, cmap='plasma', norm=norm)
            lc.set_array(speed[:-1])
            lc.set_linewidth(2)

            ax.add_collection(lc)
            ax.autoscale()
            ax.set_aspect('equal')
            ax.set_title(f'{title} Path (colored by speed)')
            ax.grid(True, alpha=0.3)

            # Colorbar
            cbar = plt.colorbar(lc, ax=ax)
            cbar.set_label('Speed')

        plt.suptitle(f'Speed Heatmap Comparison (Step {step})', fontsize=14)
        plt.tight_layout()

        fig.canvas.draw()
        img_tensor = self._fig_to_tensor(fig)
        self.writer.add_image(tag, img_tensor, step)
        plt.close(fig)

    # =========================================================================
    # PROFILER INTEGRATION
    # =========================================================================

    def create_profiler(
        self,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1
    ):
        """
        Create a PyTorch profiler for performance analysis.

        Usage:
            profiler = logger.create_profiler()
            with profiler:
                for step in range(10):
                    # training step
                    profiler.step()

        Args:
            wait: Steps to wait before profiling
            warmup: Warmup steps
            active: Active profiling steps
            repeat: Number of cycles

        Returns:
            torch.profiler.profile context manager
        """
        try:
            return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=wait,
                    warmup=warmup,
                    active=active,
                    repeat=repeat
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.run_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
        except Exception as e:
            print(f"Warning: Could not create profiler: {e}")
            return None

    # =========================================================================
    # PR CURVES (for Discriminator evaluation)
    # =========================================================================

    def log_discriminator_pr_curve(
        self,
        d_real_scores: torch.Tensor,
        d_fake_scores: torch.Tensor,
        step: Optional[int] = None,
        tag: str = 'Discriminator/PR_Curve'
    ):
        """
        Log Precision-Recall curve for discriminator performance.

        Args:
            d_real_scores: Discriminator scores for real samples [N]
            d_fake_scores: Discriminator scores for fake samples [N]
            step: Step number
            tag: TensorBoard tag
        """
        step = step if step is not None else self.global_step

        try:
            # Combine scores and create labels
            scores = torch.cat([d_real_scores, d_fake_scores]).detach().cpu()
            labels = torch.cat([
                torch.ones(d_real_scores.shape[0]),
                torch.zeros(d_fake_scores.shape[0])
            ])

            # Convert to probabilities (sigmoid for WGAN scores)
            probs = torch.sigmoid(scores)

            self.writer.add_pr_curve(
                tag,
                labels,
                probs,
                step
            )
        except Exception as e:
            print(f"Warning: PR curve logging failed: {e}")

    def log_discriminator_histogram(
        self,
        d_real_scores: torch.Tensor,
        d_fake_scores: torch.Tensor,
        step: Optional[int] = None,
        tag: str = 'Discriminator/score_distribution'
    ):
        """
        Log discriminator score distributions for real vs fake.

        Args:
            d_real_scores: Discriminator scores for real samples
            d_fake_scores: Discriminator scores for fake samples
            step: Step number
            tag: TensorBoard tag
        """
        step = step if step is not None else self.global_step

        fig, ax = plt.subplots(figsize=(10, 6))

        real_np = d_real_scores.detach().cpu().numpy().flatten()
        fake_np = d_fake_scores.detach().cpu().numpy().flatten()

        ax.hist(real_np, bins=50, alpha=0.6, label='Real', color='blue', density=True)
        ax.hist(fake_np, bins=50, alpha=0.6, label='Fake', color='red', density=True)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Decision boundary')
        ax.set_xlabel('Discriminator Score')
        ax.set_ylabel('Density')
        ax.set_title(f'Discriminator Score Distribution (Step {step})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.canvas.draw()
        img_tensor = self._fig_to_tensor(fig)
        self.writer.add_image(tag, img_tensor, step)
        plt.close(fig)

    # =========================================================================
    # GRADIENT TRACKING
    # =========================================================================

    def log_gradient_norms(
        self,
        model: nn.Module,
        step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Log gradient norms for each model component.

        Args:
            model: The TimeGAN model
            step: Step number

        Returns:
            Dictionary of component -> gradient norm
        """
        step = step if step is not None else self.global_step

        components = {
            'embedder': model.embedder,
            'recovery': model.recovery,
            'supervisor': model.supervisor,
            'generator': model.generator,
            'discriminator': model.discriminator,
        }

        grad_norms = {}
        for name, component in components.items():
            if component is None:
                continue

            total_norm = 0.0
            for param in component.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms[name] = total_norm

        self.log_scalars('Gradients', grad_norms, step)
        return grad_norms

    # =========================================================================
    # PHASE TRACKING
    # =========================================================================

    def start_phase(self, phase_name: str):
        """Mark the start of a training phase."""
        self.phase_start_steps[phase_name] = self.global_step
        self.writer.add_text(
            f'Phases/{phase_name}',
            f'Started at global step {self.global_step}',
            self.global_step
        )

    def increment_step(self, n: int = 1):
        """Increment the global step counter."""
        self.global_step += n

    def get_phase_step(self, phase_name: str) -> int:
        """Get the step within a specific phase."""
        start = self.phase_start_steps.get(phase_name, 0)
        return self.global_step - start

    # =========================================================================
    # FINAL METRICS & CLEANUP
    # =========================================================================

    def log_final_metrics(self, metrics: Dict[str, float]):
        """
        Log final training metrics and hyperparameters.

        Call this at the end of training.

        Args:
            metrics: Dictionary of final metric values
        """
        # Add training time
        training_time = (time.time() - self.start_time) / 60  # minutes
        metrics['training_time_min'] = training_time

        # Log as scalars
        for name, value in metrics.items():
            self.writer.add_scalar(f'Final/{name}', value, self.global_step)

        # Log hyperparameters with final metrics
        self.log_hyperparameters(metrics)

        # Log summary text
        summary = "## Training Complete\n\n"
        summary += f"- **Total Steps**: {self.global_step}\n"
        summary += f"- **Training Time**: {training_time:.1f} minutes\n"
        summary += "\n### Final Metrics\n\n"
        for name, value in metrics.items():
            summary += f"- **{name}**: {value:.6f}\n"

        self.writer.add_text('Run/Summary', summary, self.global_step)

    def flush(self):
        """Force flush all pending events to disk."""
        self.writer.flush()

    def close(self):
        """Close the logger and finalize."""
        self.flush()
        self.writer.close()

        total_time = (time.time() - self.start_time) / 60
        print(f"\n{'='*70}")
        print(f"TensorBoard Logger Closed")
        print(f"{'='*70}")
        print(f"  Total training time: {total_time:.1f} minutes")
        print(f"  Total steps logged: {self.global_step}")
        print(f"  Log directory: {self.run_dir}")
        print(f"{'='*70}\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_logger(
    log_dir: str = 'runs',
    experiment_name: Optional[str] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> TensorBoardLogger:
    """
    Convenience function to create a TensorBoard logger.

    Args:
        log_dir: Base directory for logs
        experiment_name: Name for the experiment
        config: Configuration dictionary
        **kwargs: Additional arguments for TensorBoardLogger

    Returns:
        Configured TensorBoardLogger instance
    """
    return TensorBoardLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        config=config,
        **kwargs
    )
