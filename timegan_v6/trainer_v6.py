"""
Trainer for TimeGAN V6 - RTSGAN-Style Two-Stage Training

Implements the full training pipeline:
- Stage 1: Autoencoder training to convergence
- Stage 2: WGAN-GP training in frozen latent space

Features:
- Convergence detection for Stage 1
- WGAN-GP with gradient penalty for Stage 2
- Feature matching loss option
- TensorBoard logging
- Checkpointing between stages
- Early stopping and learning rate scheduling

Usage:
    from timegan_v6 import TimeGANV6, TimeGANV6Config
    from timegan_v6.trainer_v6 import TimeGANV6Trainer

    config = TimeGANV6Config()
    model = TimeGANV6(config)
    trainer = TimeGANV6Trainer(model, config)

    # Train both stages
    trainer.train(train_loader)

    # Or train stages separately
    trainer.train_stage1(train_loader)
    trainer.train_stage2(train_loader)
"""

import os
import time
import json
from typing import Dict, Optional, Tuple, Iterator, Callable
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model_v6 import TimeGANV6
from .config_model_v6 import TimeGANV6Config
from .utils_v6 import compute_gradient_penalty_latent


class EarlyStopping:
    """Early stopping helper for convergence detection."""

    def __init__(self, patience: int = 1000, min_delta: float = 0.0,
                 mode: str = 'min'):
        """
        Args:
            patience: Number of iterations to wait after best
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_iteration = 0

    def __call__(self, value: float, iteration: int) -> bool:
        """
        Check if should stop.

        Args:
            value: Current metric value
            iteration: Current iteration

        Returns:
            True if should stop
        """
        if self.mode == 'min':
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.counter = 0
            self.best_iteration = iteration
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self):
        """Reset state."""
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_iteration = 0


class MovingAverage:
    """Compute moving average of values."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value: float):
        self.values.append(value)

    def get(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values.clear()


class TimeGANV6Trainer:
    """
    Trainer for TimeGAN V6 two-stage training.

    Stage 1: Train autoencoder (encoder + pooler + expander + decoder)
             until reconstruction loss converges.

    Stage 2: Freeze autoencoder, train WGAN-GP (generator + discriminator)
             in the fixed latent space.
    """

    def __init__(self, model: TimeGANV6, config: TimeGANV6Config,
                 log_dir: str = None, device: torch.device = None):
        """
        Args:
            model: TimeGANV6 model instance
            config: TimeGANV6Config instance
            log_dir: Directory for TensorBoard logs (optional)
            device: Device to train on (inferred from config if None)
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Setup logging
        self.log_dir = log_dir or config.checkpoint_dir
        self.writer = None  # TensorBoard writer (lazy init)

        # Setup optimizers (created when needed)
        self.opt_autoencoder = None
        self.opt_generator = None
        self.opt_discriminator = None

        # Training state
        self.global_step = 0
        self.stage1_completed = False
        self.stage2_completed = False

        # Metrics tracking
        self.metrics = {
            'stage1': {},
            'stage2': {}
        }

    # =========================================================================
    # Stage 1: Autoencoder Training
    # =========================================================================

    def train_stage1(self, train_loader: DataLoader,
                     val_loader: DataLoader = None,
                     callback: Callable = None) -> Dict[str, float]:
        """
        Train Stage 1: Autoencoder to convergence.

        Args:
            train_loader: DataLoader yielding (x, condition, lengths)
            val_loader: Optional validation DataLoader
            callback: Optional callback(trainer, iteration, metrics) called each log interval

        Returns:
            Final metrics dict
        """
        print("=" * 60)
        print("STAGE 1: Autoencoder Training")
        print("=" * 60)

        # Ensure autoencoder is unfrozen
        self.model.unfreeze_autoencoder()

        # Setup optimizer
        self.opt_autoencoder = optim.Adam(
            self.model.get_autoencoder_parameters(),
            lr=self.config.lr_autoencoder,
            weight_decay=self.config.stage1_weight_decay
        )

        # Setup convergence detection
        early_stopping = EarlyStopping(
            patience=self.config.stage1_patience,
            min_delta=0.001,
            mode='min'
        )

        # Moving averages for smooth logging
        loss_recon_avg = MovingAverage(100)
        loss_latent_avg = MovingAverage(100)
        loss_direct_avg = MovingAverage(100)

        # Training loop
        self.model.train()
        data_iter = self._infinite_loader(train_loader)

        start_time = time.time()

        for iteration in range(1, self.config.stage1_iterations + 1):
            # Get batch
            x_real, condition, lengths = next(data_iter)
            x_real = x_real.to(self.device)
            condition = condition.to(self.device)
            lengths = lengths.to(self.device)

            # Forward pass
            x_recon, losses = self.model.forward_autoencoder(x_real, lengths)

            # Backward pass
            self.opt_autoencoder.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            if self.config.stage1_grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.get_autoencoder_parameters(),
                    self.config.stage1_grad_clip
                )

            self.opt_autoencoder.step()

            # Update metrics
            loss_recon_avg.update(losses['recon'].item())
            loss_latent_avg.update(losses['latent'].item())
            loss_direct_avg.update(losses['direct'].item())

            # Logging
            if iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                iter_per_sec = iteration / elapsed

                metrics = {
                    'loss_recon': loss_recon_avg.get(),
                    'loss_latent': loss_latent_avg.get(),
                    'loss_direct': loss_direct_avg.get(),
                    'loss_total': losses['total'].item()
                }

                print(f"[Stage 1] Iter {iteration:6d}/{self.config.stage1_iterations} | "
                      f"Direct: {metrics['loss_direct']:.4f} | "
                      f"Recon: {metrics['loss_recon']:.4f} | "
                      f"Latent: {metrics['loss_latent']:.4f} | "
                      f"{iter_per_sec:.1f} it/s")

                self._log_metrics('stage1', metrics, iteration)

                # Log trajectory visualizations and embeddings at eval_interval
                if iteration % self.config.eval_interval == 0:
                    self._log_trajectories('stage1', x_real, x_recon, lengths, iteration)
                    self._log_embeddings(x_real, condition, lengths, iteration)

                if callback:
                    callback(self, iteration, metrics)

            # Check convergence based on direct loss (encoder quality)
            if loss_direct_avg.get() < self.config.stage1_threshold:
                if early_stopping(loss_direct_avg.get(), iteration):
                    print(f"\nConverged at iteration {iteration}!")
                    print(f"Best direct reconstruction loss: {early_stopping.best:.4f}")
                    break

            # Checkpointing
            if iteration % self.config.save_interval == 0:
                self._save_checkpoint(f'stage1_iter{iteration}.pt', stage=1)

        # Final metrics
        final_metrics = {
            'loss_recon': loss_recon_avg.get(),
            'loss_latent': loss_latent_avg.get(),
            'loss_direct': loss_direct_avg.get(),
            'iterations': iteration,
            'converged': loss_direct_avg.get() < self.config.stage1_threshold
        }

        self.metrics['stage1'] = final_metrics
        self.stage1_completed = True

        # Save Stage 1 checkpoint
        self._save_checkpoint('stage1_final.pt', stage=1)

        print(f"\nStage 1 Complete!")
        print(f"  Final direct loss: {final_metrics['loss_direct']:.4f}")
        print(f"  Final recon loss: {final_metrics['loss_recon']:.4f}")
        print(f"  Converged: {final_metrics['converged']}")
        print("=" * 60)

        return final_metrics

    # =========================================================================
    # Stage 2: WGAN-GP Training
    # =========================================================================

    def train_stage2(self, train_loader: DataLoader,
                     val_loader: DataLoader = None,
                     callback: Callable = None) -> Dict[str, float]:
        """
        Train Stage 2: WGAN-GP in frozen latent space.

        Args:
            train_loader: DataLoader yielding (x, condition, lengths)
            val_loader: Optional validation DataLoader
            callback: Optional callback(trainer, iteration, metrics)

        Returns:
            Final metrics dict
        """
        print("=" * 60)
        print("STAGE 2: WGAN-GP Training")
        print("=" * 60)

        # Freeze autoencoder
        self.model.freeze_autoencoder()
        print("Autoencoder frozen.")

        # Setup optimizers with WGAN-GP recommended betas
        self.opt_generator = optim.Adam(
            self.model.get_generator_parameters(),
            lr=self.config.lr_generator,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.stage2_weight_decay
        )

        self.opt_discriminator = optim.Adam(
            self.model.get_discriminator_parameters(),
            lr=self.config.lr_discriminator,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.stage2_weight_decay
        )

        # Moving averages
        d_loss_avg = MovingAverage(100)
        g_loss_avg = MovingAverage(100)
        wasserstein_avg = MovingAverage(100)
        gp_avg = MovingAverage(100)

        # Training loop
        self.model.train()
        data_iter = self._infinite_loader(train_loader)

        start_time = time.time()

        for iteration in range(1, self.config.stage2_iterations + 1):
            # ─── Discriminator Updates ─────────────────────────────
            d_losses_accum = []

            for _ in range(self.config.n_critic):
                x_real, condition, lengths = next(data_iter)
                x_real = x_real.to(self.device)
                condition = condition.to(self.device)
                lengths = lengths.to(self.device)

                # Discriminator forward
                d_losses = self.model.forward_discriminator(x_real, condition, lengths)

                # Backward
                self.opt_discriminator.zero_grad()
                d_losses['d_loss'].backward()

                if self.config.stage2_grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.get_discriminator_parameters(),
                        self.config.stage2_grad_clip
                    )

                self.opt_discriminator.step()

                d_losses_accum.append({
                    'd_loss': d_losses['d_loss'].item(),
                    'wasserstein': d_losses['wasserstein'].item(),
                    'gp': d_losses['gp'].item()
                })

            # Average D losses over critic updates
            d_loss_avg.update(sum(d['d_loss'] for d in d_losses_accum) / len(d_losses_accum))
            wasserstein_avg.update(sum(d['wasserstein'] for d in d_losses_accum) / len(d_losses_accum))
            gp_avg.update(sum(d['gp'] for d in d_losses_accum) / len(d_losses_accum))

            # ─── Generator Update ──────────────────────────────────
            x_real, condition, lengths = next(data_iter)
            x_real = x_real.to(self.device)
            condition = condition.to(self.device)
            lengths = lengths.to(self.device)

            batch_size = x_real.size(0)

            g_losses = self.model.forward_generator(
                batch_size, condition,
                x_real if self.config.use_feature_matching else None,
                lengths if self.config.use_feature_matching else None
            )

            self.opt_generator.zero_grad()
            g_losses['g_loss'].backward()

            if self.config.stage2_grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.get_generator_parameters(),
                    self.config.stage2_grad_clip
                )

            self.opt_generator.step()

            g_loss_avg.update(g_losses['g_loss'].item())

            # Logging
            if iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                iter_per_sec = iteration / elapsed

                metrics = {
                    'd_loss': d_loss_avg.get(),
                    'g_loss': g_loss_avg.get(),
                    'wasserstein': wasserstein_avg.get(),
                    'gp': gp_avg.get(),
                }

                print(f"[Stage 2] Iter {iteration:6d}/{self.config.stage2_iterations} | "
                      f"D: {metrics['d_loss']:.4f} | "
                      f"G: {metrics['g_loss']:.4f} | "
                      f"W: {metrics['wasserstein']:.4f} | "
                      f"{iter_per_sec:.1f} it/s")

                self._log_metrics('stage2', metrics, iteration)

                if callback:
                    callback(self, iteration, metrics)

            # Evaluation and visualization
            if iteration % self.config.eval_interval == 0:
                # Log generated trajectory visualizations and embeddings
                self._log_generated_trajectories(x_real, condition, lengths, iteration)
                self._log_embeddings(x_real, condition, lengths, iteration)

                if val_loader is not None:
                    eval_metrics = self.evaluate(val_loader)
                    print(f"  [Eval] Recon MSE: {eval_metrics.get('recon_mse', 0):.4f}")

            # Checkpointing
            if iteration % self.config.save_interval == 0:
                self._save_checkpoint(f'stage2_iter{iteration}.pt', stage=2)

        # Final metrics
        final_metrics = {
            'd_loss': d_loss_avg.get(),
            'g_loss': g_loss_avg.get(),
            'wasserstein': wasserstein_avg.get(),
            'iterations': iteration
        }

        self.metrics['stage2'] = final_metrics
        self.stage2_completed = True

        # Save final checkpoint
        self._save_checkpoint('stage2_final.pt', stage=2)

        print(f"\nStage 2 Complete!")
        print(f"  Final D loss: {final_metrics['d_loss']:.4f}")
        print(f"  Final G loss: {final_metrics['g_loss']:.4f}")
        print(f"  Final Wasserstein: {final_metrics['wasserstein']:.4f}")
        print("=" * 60)

        return final_metrics

    # =========================================================================
    # Full Training Pipeline
    # =========================================================================

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader = None,
              skip_stage1: bool = False,
              stage1_checkpoint: str = None) -> Dict[str, Dict]:
        """
        Run full two-stage training pipeline.

        Args:
            train_loader: DataLoader for training
            val_loader: Optional DataLoader for validation
            skip_stage1: Skip Stage 1 (use if loading pretrained autoencoder)
            stage1_checkpoint: Path to Stage 1 checkpoint to load

        Returns:
            Dict with 'stage1' and 'stage2' metrics
        """
        print("\n" + "=" * 60)
        print("TimeGAN V6 Training Pipeline")
        print("=" * 60 + "\n")

        # Load Stage 1 checkpoint if provided
        if stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {stage1_checkpoint}")
            self.load_checkpoint(stage1_checkpoint)
            skip_stage1 = True

        # Stage 1: Autoencoder
        if not skip_stage1:
            stage1_metrics = self.train_stage1(train_loader, val_loader)
        else:
            print("Skipping Stage 1 (using pretrained autoencoder)")
            stage1_metrics = self.metrics.get('stage1', {})

        # Stage 2: WGAN-GP
        stage2_metrics = self.train_stage2(train_loader, val_loader)

        # Final checkpoint
        self._save_checkpoint('final.pt', stage=2)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return {
            'stage1': stage1_metrics,
            'stage2': stage2_metrics
        }

    # =========================================================================
    # Evaluation
    # =========================================================================

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader,
                 n_batches: int = 10) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            data_loader: Validation DataLoader
            n_batches: Number of batches to evaluate

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()

        recon_losses = []
        gen_quality = []

        for i, (x_real, condition, lengths) in enumerate(data_loader):
            if i >= n_batches:
                break

            x_real = x_real.to(self.device)
            condition = condition.to(self.device)
            lengths = lengths.to(self.device)

            # Reconstruction quality
            x_recon, losses = self.model.forward_autoencoder(x_real, lengths)
            recon_losses.append(losses['recon'].item())

            # Generation quality (compare distributions)
            x_fake = self.model.generate(x_real.size(0), condition)

            # Simple metric: MSE between real and fake means
            real_mean = x_real.mean(dim=(0, 1))
            fake_mean = x_fake[:, :x_real.size(1), :].mean(dim=(0, 1))
            gen_quality.append(((real_mean - fake_mean) ** 2).mean().item())

        self.model.train()

        return {
            'recon_mse': sum(recon_losses) / len(recon_losses) if recon_losses else 0,
            'gen_quality': sum(gen_quality) / len(gen_quality) if gen_quality else 0
        }

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _save_checkpoint(self, filename: str, stage: int):
        """Save training checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'stage': stage,
            'metrics': self.metrics,
            'stage1_completed': self.stage1_completed,
            'stage2_completed': self.stage2_completed,
        }

        # Save optimizer states
        if self.opt_autoencoder is not None:
            checkpoint['opt_autoencoder'] = self.opt_autoencoder.state_dict()
        if self.opt_generator is not None:
            checkpoint['opt_generator'] = self.opt_generator.state_dict()
        if self.opt_discriminator is not None:
            checkpoint['opt_discriminator'] = self.opt_discriminator.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, load_optimizers: bool = True):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint.get('metrics', {})
        self.stage1_completed = checkpoint.get('stage1_completed', False)
        self.stage2_completed = checkpoint.get('stage2_completed', False)

        if load_optimizers:
            if 'opt_autoencoder' in checkpoint and self.opt_autoencoder is not None:
                self.opt_autoencoder.load_state_dict(checkpoint['opt_autoencoder'])
            if 'opt_generator' in checkpoint and self.opt_generator is not None:
                self.opt_generator.load_state_dict(checkpoint['opt_generator'])
            if 'opt_discriminator' in checkpoint and self.opt_discriminator is not None:
                self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])

        print(f"Loaded checkpoint: {path}")
        print(f"  Stage 1 completed: {self.stage1_completed}")
        print(f"  Stage 2 completed: {self.stage2_completed}")

    # =========================================================================
    # Utilities
    # =========================================================================

    def _infinite_loader(self, data_loader: DataLoader) -> Iterator:
        """Create infinite iterator from DataLoader."""
        while True:
            for batch in data_loader:
                yield batch

    def _log_metrics(self, stage: str, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard (if available)."""
        if self.writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.log_dir)
            except ImportError:
                return

        for name, value in metrics.items():
            self.writer.add_scalar(f'{stage}/{name}', value, step)

    def _log_trajectories(self, stage: str, x_real: torch.Tensor, x_recon: torch.Tensor,
                          lengths: torch.Tensor, step: int, n_samples: int = 3):
        """
        Log trajectory visualizations to TensorBoard.

        Creates side-by-side plots of real vs reconstructed/generated trajectories.
        """
        if self.writer is None:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # Move to CPU and numpy
        x_real_np = x_real[:n_samples].detach().cpu().numpy()
        x_recon_np = x_recon[:n_samples].detach().cpu().numpy()
        lengths_np = lengths[:n_samples].detach().cpu().numpy()

        # Create figure with subplots for each sample
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            length = int(lengths_np[i])
            real = x_real_np[i, :length, :]
            recon = x_recon_np[i, :length, :]

            # Column 1: XY trajectory (cumsum of dx, dy)
            ax = axes[i, 0]
            real_x, real_y = np.cumsum(real[:, 0]), np.cumsum(real[:, 1])
            recon_x, recon_y = np.cumsum(recon[:, 0]), np.cumsum(recon[:, 1])
            ax.plot(real_x, real_y, 'b-', label='Real', linewidth=2, alpha=0.8)
            ax.plot(recon_x, recon_y, 'r--', label='Recon', linewidth=2, alpha=0.8)
            ax.scatter([real_x[0]], [real_y[0]], c='green', s=80, marker='o', zorder=5)
            ax.set_title(f'Sample {i+1}: XY Trajectory')
            ax.legend(loc='best', fontsize=8)
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

            # Column 2: Speed over time
            ax = axes[i, 1]
            time = np.arange(length)
            ax.plot(time, real[:, 2], 'b-', label='Real', linewidth=1.5)
            ax.plot(time, recon[:, 2], 'r--', label='Recon', linewidth=1.5)
            ax.set_title('Speed')
            ax.set_xlabel('Time')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Column 3: Feature error heatmap
            ax = axes[i, 2]
            error = np.abs(real - recon)
            feature_names = ['dx', 'dy', 'spd', 'acc', 'sin', 'cos', 'ang', 'dt']
            im = ax.imshow(error.T, aspect='auto', cmap='hot', vmin=0, vmax=0.5)
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names, fontsize=8)
            ax.set_xlabel('Time')
            ax.set_title(f'Abs Error (mean={error.mean():.3f})')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()

        # Log to TensorBoard
        self.writer.add_figure(f'{stage}/trajectories', fig, step)
        plt.close(fig)

    def _log_generated_trajectories(self, x_real: torch.Tensor, condition: torch.Tensor,
                                     lengths: torch.Tensor, step: int, n_samples: int = 3):
        """
        Log generated vs real trajectory visualizations to TensorBoard (Stage 2).
        """
        if self.writer is None:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # Generate fake trajectories with matching conditions
        self.model.eval()
        with torch.no_grad():
            seq_len = x_real.size(1)
            x_fake = self.model.generate(n_samples, condition[:n_samples], seq_len=seq_len)
        self.model.train()

        # Move to CPU and numpy
        x_real_np = x_real[:n_samples].detach().cpu().numpy()
        x_fake_np = x_fake[:n_samples].detach().cpu().numpy()
        lengths_np = lengths[:n_samples].detach().cpu().numpy()
        cond_np = condition[:n_samples].detach().cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            length = int(lengths_np[i])
            real = x_real_np[i, :length, :]
            fake = x_fake_np[i, :length, :]
            cond = cond_np[i, 0] if cond_np.ndim > 1 else cond_np[i]

            # Column 1: XY trajectory
            ax = axes[i, 0]
            real_x, real_y = np.cumsum(real[:, 0]), np.cumsum(real[:, 1])
            fake_x, fake_y = np.cumsum(fake[:, 0]), np.cumsum(fake[:, 1])
            ax.plot(real_x, real_y, 'b-', label='Real', linewidth=2, alpha=0.8)
            ax.plot(fake_x, fake_y, 'g--', label='Generated', linewidth=2, alpha=0.8)
            ax.scatter([real_x[0]], [real_y[0]], c='blue', s=80, marker='o', zorder=5)
            ax.scatter([fake_x[0]], [fake_y[0]], c='green', s=80, marker='o', zorder=5)
            ax.set_title(f'Sample {i+1}: XY (cond={cond:.2f})')
            ax.legend(loc='best', fontsize=8)
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

            # Column 2: Speed comparison
            ax = axes[i, 1]
            time = np.arange(length)
            ax.plot(time, real[:, 2], 'b-', label='Real', linewidth=1.5)
            ax.plot(time, fake[:, 2], 'g--', label='Generated', linewidth=1.5)
            ax.set_title('Speed')
            ax.set_xlabel('Time')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Column 3: Feature distributions
            ax = axes[i, 2]
            feature_names = ['dx', 'dy', 'spd', 'acc']
            x_pos = np.arange(len(feature_names))
            width = 0.35
            real_means = [real[:, j].mean() for j in range(4)]
            fake_means = [fake[:, j].mean() for j in range(4)]
            ax.bar(x_pos - width/2, real_means, width, label='Real', color='blue', alpha=0.7)
            ax.bar(x_pos + width/2, fake_means, width, label='Generated', color='green', alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(feature_names)
            ax.set_title('Feature Means')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self.writer.add_figure('stage2/generated_trajectories', fig, step)
        plt.close(fig)

    def _log_embeddings(self, x_real: torch.Tensor, condition: torch.Tensor,
                        lengths: torch.Tensor, step: int, n_samples: int = 200):
        """
        Log latent embeddings to TensorBoard projector.

        Visualizes z_summary vectors colored by condition value.
        View in TensorBoard under "Projector" tab.
        """
        if self.writer is None:
            return

        # Limit samples for performance
        n = min(n_samples, x_real.size(0))

        self.model.eval()
        with torch.no_grad():
            # Encode to latent summaries
            z_summary = self.model.encode(x_real[:n], lengths[:n])
        self.model.train()

        # Create metadata (condition values and lengths)
        cond_np = condition[:n].detach().cpu().numpy()
        lengths_np = lengths[:n].detach().cpu().numpy()

        # Format metadata as list of strings
        metadata = []
        for i in range(n):
            cond_val = cond_np[i, 0] if cond_np.ndim > 1 else cond_np[i]
            metadata.append(f"cond={cond_val:.2f}_len={int(lengths_np[i])}")

        # Log embeddings with metadata
        self.writer.add_embedding(
            z_summary,
            metadata=metadata,
            tag='latent_space',
            global_step=step
        )

    def get_model(self) -> TimeGANV6:
        """Get the model."""
        return self.model


# ============================================================================
# Convenience Functions
# ============================================================================

def create_trainer(config: TimeGANV6Config = None,
                   device: str = None) -> Tuple[TimeGANV6, TimeGANV6Trainer]:
    """
    Create model and trainer with given config.

    Args:
        config: Configuration (uses defaults if None)
        device: Device string ('cuda' or 'cpu')

    Returns:
        (model, trainer) tuple
    """
    if config is None:
        config = TimeGANV6Config()

    if device is not None:
        config.device = device

    model = TimeGANV6(config)
    trainer = TimeGANV6Trainer(model, config)

    return model, trainer


def train_v6_optimized(data_dir: str, config: TimeGANV6Config = None,
                       device: str = None, skip_stage1: bool = False,
                       stage1_checkpoint: str = None) -> Tuple[TimeGANV6, Dict]:
    """
    Train TimeGAN V6 with stage-optimized data loading.

    Uses:
    - Length-aware batching for Stage 1 (reduced padding waste)
    - Condition-stratified sampling for Stage 2 (uniform conditions)

    This is the recommended way to train V6 for best results.

    Args:
        data_dir: Directory containing preprocessed V4/V6 data
        config: TimeGANV6Config (uses defaults if None)
        device: Device string ('cuda' or 'cpu')
        skip_stage1: Skip Stage 1 (use pretrained autoencoder)
        stage1_checkpoint: Path to Stage 1 checkpoint to load

    Returns:
        (model, metrics) tuple

    Usage:
        from timegan_v6 import train_v6_optimized

        model, metrics = train_v6_optimized(
            data_dir='processed_data_v4',
            device='cuda'
        )

        # Generate samples
        conditions = torch.rand(100, 1)
        x_fake = model.generate(100, conditions)
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_loader_v6 import create_stage1_loader, create_stage2_loader, validate_v6_data

    # Validate data
    validate_v6_data(data_dir)

    # Setup config
    if config is None:
        config = TimeGANV6Config()
    if device is not None:
        config.device = device

    # Create model and trainer
    model = TimeGANV6(config)
    trainer = TimeGANV6Trainer(model, config)

    # Load Stage 1 checkpoint if provided
    if stage1_checkpoint:
        print(f"Loading Stage 1 checkpoint: {stage1_checkpoint}")
        trainer.load_checkpoint(stage1_checkpoint)
        skip_stage1 = True

    metrics = {'stage1': {}, 'stage2': {}}

    # Stage 1: Autoencoder with length-aware batching
    if not skip_stage1:
        print("\n" + "=" * 60)
        print("Loading Stage 1 data with LENGTH-AWARE BATCHING")
        print("=" * 60)

        stage1_loader = create_stage1_loader(
            data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            augment=True,
            quality_filter=True
        )

        metrics['stage1'] = trainer.train_stage1(stage1_loader)
    else:
        print("Skipping Stage 1 (using pretrained autoencoder)")

    # Stage 2: WGAN-GP with condition-stratified sampling
    print("\n" + "=" * 60)
    print("Loading Stage 2 data with CONDITION-STRATIFIED SAMPLING")
    print("=" * 60)

    stage2_loader = create_stage2_loader(
        data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=False,  # No augmentation for GAN training
        quality_filter=True
    )

    metrics['stage2'] = trainer.train_stage2(stage2_loader)

    # Save final model
    trainer._save_checkpoint('final.pt', stage=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    return model, metrics


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("Testing TimeGAN V6 Trainer...")
    print("=" * 60)

    # Create dummy data loader
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=100, seq_len=50, feature_dim=8):
            self.n_samples = n_samples
            self.seq_len = seq_len
            self.feature_dim = feature_dim

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            x = torch.randn(self.seq_len, self.feature_dim).clamp(-1, 1)
            condition = torch.rand(1)
            length = torch.randint(20, self.seq_len + 1, (1,)).item()
            return x, condition, torch.tensor(length)

    def collate_fn(batch):
        xs, conditions, lengths = zip(*batch)
        xs = torch.stack(xs)
        conditions = torch.stack(conditions)
        lengths = torch.stack(lengths)
        return xs, conditions, lengths

    # Create data loader
    dataset = DummyDataset(n_samples=200)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
    )

    # Create model and trainer with fast config for testing
    from timegan_v6.config_model_v6 import get_fast_config

    config = get_fast_config()
    config.device = 'cpu'
    config.stage1_iterations = 50  # Very short for testing
    config.stage2_iterations = 50
    config.log_interval = 10
    config.save_interval = 100  # Don't save during test
    config.checkpoint_dir = '/tmp/timegan_v6_test'

    model = TimeGANV6(config)
    trainer = TimeGANV6Trainer(model, config)

    print("\n=== Testing Stage 1 ===")
    stage1_metrics = trainer.train_stage1(train_loader)
    print(f"Stage 1 final metrics: {stage1_metrics}")
    assert 'loss_recon' in stage1_metrics
    print("Stage 1: PASS")

    print("\n=== Testing Stage 2 ===")
    stage2_metrics = trainer.train_stage2(train_loader)
    print(f"Stage 2 final metrics: {stage2_metrics}")
    assert 'd_loss' in stage2_metrics
    assert 'g_loss' in stage2_metrics
    print("Stage 2: PASS")

    print("\n=== Testing Generation After Training ===")
    conditions = torch.rand(4, 1)
    x_fake = model.generate(4, conditions, seq_len=50)
    print(f"Generated shape: {x_fake.shape}")
    assert x_fake.shape == (4, 50, 8)
    print("Generation: PASS")

    print("\n=== Testing Checkpoint Save/Load ===")
    trainer._save_checkpoint('test_checkpoint.pt', stage=2)
    checkpoint_path = os.path.join(config.checkpoint_dir, 'test_checkpoint.pt')
    assert os.path.exists(checkpoint_path)

    # Create new model and load
    model2 = TimeGANV6(config)
    trainer2 = TimeGANV6Trainer(model2, config)
    trainer2.load_checkpoint(checkpoint_path)
    print("Checkpoint: PASS")

    # Clean up
    import shutil
    if os.path.exists(config.checkpoint_dir):
        shutil.rmtree(config.checkpoint_dir)

    print("\n" + "=" * 60)
    print("ALL TRAINER TESTS PASSED")
    print("=" * 60)
