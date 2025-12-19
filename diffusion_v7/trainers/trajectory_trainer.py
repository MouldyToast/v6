"""
Trajectory Diffusion Trainer

Training loop for trajectory diffusion model with:
- Classifier-free guidance (CFG) training
- Smoke test mode (single-batch overfitting)
- Full training mode
- Logging and checkpointing

Adapts MotionDiffuse's DDPM trainer for trajectory generation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
import time
from pathlib import Path

from ..models.goal_conditioner import GoalConditioner
from ..models.trajectory_transformer import TrajectoryTransformer
from ..models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
from ..config_trajectory import TrajectoryDiffusionConfig


class TrajectoryDiffusionTrainer:
    """
    Trainer for trajectory diffusion model.

    Handles both smoke testing (single-batch overfitting) and full training.

    Args:
        config: TrajectoryDiffusionConfig instance
    """

    def __init__(self, config: TrajectoryDiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create models
        self.goal_conditioner = GoalConditioner(
            condition_dim=config.condition_dim,
            latent_dim=config.latent_dim,
            use_cfg=config.use_cfg,
            dropout=config.dropout
        ).to(self.device)

        self.model = TrajectoryTransformer(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_size=config.ff_size,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            activation=config.activation,
            goal_latent_dim=config.latent_dim  # Match goal conditioner output
        ).to(self.device)

        # Create diffusion process
        betas = get_named_beta_schedule(
            config.noise_schedule,
            config.diffusion_steps
        )

        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=self._get_model_mean_type(config.model_mean_type),
            model_var_type=self._get_model_var_type(config.model_var_type),
            loss_type=self._get_loss_type(config.loss_type)
        )

        # Timestep sampler
        self.timestep_sampler = create_named_schedule_sampler('uniform', self.diffusion)

        # Optimizer
        all_params = list(self.model.parameters()) + list(self.goal_conditioner.parameters())
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                all_params,
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                all_params,
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # Learning rate scheduler (optional)
        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.lr_decay_steps,
                eta_min=config.learning_rate * 0.1
            )

        # Logging
        self.writer = None
        if config.log_dir:
            log_path = Path(config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_path))

        # Checkpoint directory
        if config.save_dir:
            Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Model info
        self._print_model_info()

    def _get_model_mean_type(self, mean_type_str: str) -> ModelMeanType:
        """Convert string to ModelMeanType enum."""
        mapping = {
            'epsilon': ModelMeanType.EPSILON,
            'x0': ModelMeanType.START_X,
            'previous_x': ModelMeanType.PREVIOUS_X
        }
        return mapping.get(mean_type_str, ModelMeanType.EPSILON)

    def _get_model_var_type(self, var_type_str: str) -> ModelVarType:
        """Convert string to ModelVarType enum."""
        mapping = {
            'fixed_small': ModelVarType.FIXED_SMALL,
            'fixed_large': ModelVarType.FIXED_LARGE,
            'learned': ModelVarType.LEARNED,
            'learned_range': ModelVarType.LEARNED_RANGE
        }
        return mapping.get(var_type_str, ModelVarType.FIXED_SMALL)

    def _get_loss_type(self, loss_type_str: str) -> LossType:
        """Convert string to LossType enum."""
        mapping = {
            'mse': LossType.MSE,
            'l1': LossType.RESCALED_MSE,  # Using rescaled for smoother training
            'kl': LossType.KL
        }
        return mapping.get(loss_type_str, LossType.MSE)

    def _print_model_info(self):
        """Print model architecture and parameter count."""
        print("\n" + "=" * 70)
        print("TRAJECTORY DIFFUSION MODEL")
        print("=" * 70)

        # Count parameters
        model_params = sum(p.numel() for p in self.model.parameters())
        conditioner_params = sum(p.numel() for p in self.goal_conditioner.parameters())
        total_params = model_params + conditioner_params

        print(f"\n[Model Architecture]")
        print(f"  Transformer parameters: {model_params:,}")
        print(f"  Goal conditioner parameters: {conditioner_params:,}")
        print(f"  Total parameters: {total_params:,}")

        print(f"\n[Diffusion Settings]")
        print(f"  Timesteps: {self.config.diffusion_steps}")
        print(f"  Noise schedule: {self.config.noise_schedule}")
        print(f"  Model mean type: {self.config.model_mean_type}")

        print(f"\n[Training Settings]")
        print(f"  Optimizer: {self.config.optimizer}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient clipping: {self.config.grad_clip}")

        print(f"\n[Classifier-Free Guidance]")
        print(f"  Enabled: {self.config.use_cfg}")
        if self.config.use_cfg:
            print(f"  Dropout rate: {self.config.cfg_dropout}")
            print(f"  Guidance scale: {self.config.cfg_guidance_scale}")

        if self.config.smoke_test:
            print(f"\n[SMOKE TEST MODE]")
            print(f"  Single-batch overfitting test")
            print(f"  Steps: {self.config.smoke_test_steps}")
            print(f"  Batch size: {self.config.smoke_test_batch_size}")

        print("=" * 70 + "\n")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dict with 'motion', 'condition', 'length' keys

        Returns:
            Dict with loss values
        """
        self.model.train()
        self.goal_conditioner.train()

        # Move batch to device
        motion = batch['motion'].to(self.device)  # (B, T, 8)
        condition = batch['condition'].to(self.device)  # (B, 3)
        lengths = batch['length'].to(self.device)  # (B,)

        batch_size = motion.shape[0]

        # Sample timesteps
        t, _ = self.timestep_sampler.sample(batch_size, self.device)

        # Classifier-free guidance: randomly drop conditions
        cfg_mask = None
        if self.config.use_cfg:
            # Create mask: True = use condition, False = use null embedding
            cfg_mask = torch.rand(batch_size, device=self.device) > self.config.cfg_dropout

        # Embed goal conditions (with CFG masking)
        goal_embed = self.goal_conditioner(condition, mask=cfg_mask)

        # Diffusion training: predict noise
        # The model expects (x_noisy, t, goal_embed, lengths)
        # gaussian_diffusion.training_losses will handle the forward diffusion
        def model_fn(x, timesteps):
            """Wrapper for model call."""
            return self.model(x, timesteps, goal_embed, lengths)

        # Compute diffusion loss
        losses = self.diffusion.training_losses(
            model=model_fn,
            x_start=motion,
            t=t,
            model_kwargs={},  # We handle conditioning in model_fn
            noise=None  # Let diffusion generate noise
        )

        # Extract loss
        # gaussian_diffusion returns dict with 'mse', 'target', 'pred' keys
        loss_mse = losses['mse'].mean()

        # Total loss
        loss = loss_mse

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.goal_conditioner.parameters()),
                self.config.grad_clip
            )

        # Optimizer step
        self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Return loss values
        return {
            'loss': loss.item(),
            'loss_mse': loss_mse.item(),
        }

    def train_smoke_test(
        self,
        single_batch: Dict[str, torch.Tensor]
    ):
        """
        Smoke test: Overfit on a single batch.

        Goal: Verify architecture works by achieving near-zero loss.

        Args:
            single_batch: Fixed batch to overfit on
        """
        print("\n" + "=" * 70)
        print("SMOKE TEST: Single-Batch Overfitting")
        print("=" * 70)
        print(f"Target: Loss should go to near-zero within {self.config.smoke_test_steps} steps")
        print(f"If loss doesn't decrease, there's a bug in data loading or model architecture")
        print("=" * 70 + "\n")

        start_time = time.time()
        best_loss = float('inf')

        for step in range(self.config.smoke_test_steps):
            # Train on the same batch repeatedly
            metrics = self.train_step(single_batch)

            # Update best loss
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']

            # Log
            if (step + 1) % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {step+1}/{self.config.smoke_test_steps} | "
                      f"Loss: {metrics['loss']:.6f} | "
                      f"MSE: {metrics['loss_mse']:.6f} | "
                      f"Best: {best_loss:.6f} | "
                      f"Time: {elapsed:.1f}s")

                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('smoke_test/loss', metrics['loss'], step)
                    self.writer.add_scalar('smoke_test/loss_mse', metrics['loss_mse'], step)
                    self.writer.add_scalar('smoke_test/best_loss', best_loss, step)

        # Final report
        print("\n" + "=" * 70)
        print("SMOKE TEST COMPLETE")
        print("=" * 70)
        print(f"Final loss: {metrics['loss']:.6f}")
        print(f"Best loss: {best_loss:.6f}")

        if best_loss < 0.01:
            print("✓ SUCCESS: Loss went to near-zero")
            print("  Architecture is working correctly")
        elif best_loss < 0.1:
            print("⚠ PARTIAL SUCCESS: Loss decreased but didn't reach zero")
            print("  May need more steps or learning rate tuning")
        else:
            print("✗ FAILURE: Loss didn't decrease significantly")
            print("  Check data loading and model architecture")

        print("=" * 70 + "\n")

        # Save checkpoint
        self.save_checkpoint('smoke_test_final.pth')

        return best_loss

    def train_full(
        self,
        train_loader,
        val_loader=None
    ):
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Optional validation DataLoader
        """
        print("\n" + "=" * 70)
        print("STARTING FULL TRAINING")
        print("=" * 70)
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Steps per epoch: {len(train_loader)}")
        print(f"Total steps: {self.config.num_epochs * len(train_loader)}")
        print("=" * 70 + "\n")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Training
            train_metrics = self._train_epoch(train_loader)

            # Validation
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)

            # Logging
            epoch_time = time.time() - epoch_start
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)

            # Checkpointing
            if (epoch + 1) % (self.config.checkpoint_interval // len(train_loader)) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70 + "\n")

        # Save final checkpoint
        self.save_checkpoint('final.pth')

    def _train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.goal_conditioner.train()

        epoch_metrics = {
            'loss': 0.0,
            'loss_mse': 0.0,
        }

        for batch_idx, batch in enumerate(train_loader):
            metrics = self.train_step(batch)

            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]

            # Log step
            if (self.global_step + 1) % self.config.log_interval == 0:
                print(f"Epoch {self.epoch} | Step {self.global_step} | "
                      f"Loss: {metrics['loss']:.6f} | "
                      f"MSE: {metrics['loss_mse']:.6f}")

                if self.writer:
                    for key, value in metrics.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)

            self.global_step += 1

        # Average metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def _validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.goal_conditioner.eval()

        val_metrics = {
            'loss': 0.0,
            'loss_mse': 0.0,
        }

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                motion = batch['motion'].to(self.device)
                condition = batch['condition'].to(self.device)
                lengths = batch['length'].to(self.device)
                batch_size = motion.shape[0]

                # Sample timesteps
                t, _ = self.timestep_sampler.sample(batch_size, self.device)

                # No CFG masking during validation (use full conditioning)
                goal_embed = self.goal_conditioner(condition, mask=None)

                # Model forward
                def model_fn(x, timesteps):
                    return self.model(x, timesteps, goal_embed, lengths)

                # Compute loss
                losses = self.diffusion.training_losses(
                    model=model_fn,
                    x_start=motion,
                    t=t,
                    model_kwargs={},
                    noise=None
                )

                loss_mse = losses['mse'].mean()
                loss = loss_mse

                # Accumulate
                val_metrics['loss'] += loss.item()
                val_metrics['loss_mse'] += loss_mse.item()

        # Average
        num_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches

        return val_metrics

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        epoch_time: float
    ):
        """Log epoch results."""
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        if val_metrics:
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Time: {epoch_time:.1f}s")

        if self.writer:
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            if val_metrics:
                self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/time', epoch_time, epoch)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'goal_conditioner_state_dict': self.goal_conditioner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        save_path = Path(self.config.save_dir) / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        load_path = Path(self.config.save_dir) / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.goal_conditioner.load_state_dict(checkpoint['goal_conditioner_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint: {load_path}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Global step: {self.global_step}")


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing TrajectoryDiffusionTrainer...")
    print("=" * 70)

    # This will test initialization only (no actual training without data)
    from ..config_trajectory import get_smoke_test_config

    config = get_smoke_test_config()
    config.device = 'cpu'  # Use CPU for testing

    print("\n=== Creating Trainer ===")
    trainer = TrajectoryDiffusionTrainer(config)

    print("\n✓ Trainer initialization successful")
    print("=" * 70)
