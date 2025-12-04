"""
Optuna Hyperparameter Optimization for TimeGAN V6
Full integration with V6 two-stage training pipeline

Usage:
    # Stage 1 only
    python optuna_optimize_v6.py --stage 1 --n-trials 50

    # Stage 2 only (requires Stage 1 checkpoint)
    python optuna_optimize_v6.py --stage 2 --n-trials 100 --stage1-checkpoint checkpoints/best_stage1.pt

    # Full pipeline
    python optuna_optimize_v6.py --stage both --n-trials 50

    # Resume existing study
    python optuna_optimize_v6.py --stage 1 --resume

    # Analyze results
    python optuna_optimize_v6.py --analyze --study-name timegan_v6_stage1
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.exceptions import TrialPruned
from optuna import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import json
import traceback
import numpy as np
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# V6 imports
from timegan_v6.model_v6 import TimeGANV6
from timegan_v6.config_model_v6 import TimeGANV6Config
from timegan_v6.evaluation_v6 import TimeGANV6Evaluator
from data_loader_v6 import (
    create_stage1_loader,
    create_stage2_loader,
    create_dataloaders_v6,
    validate_v6_data,
    MouseTrajectoryDatasetV6
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptunaConfig:
    """Central configuration for optimization runs."""

    # Data
    data_dir: str = "./processed_data_v4"

    # Storage
    storage_path: str = "sqlite:///v6_optuna.db"
    tensorboard_dir: str = "./runs/optuna"
    checkpoint_dir: str = "./checkpoints/optuna"

    # Study settings
    stage1_trials: int = 50
    stage2_trials: int = 100

    # Training limits (reduced for Optuna trials)
    stage1_max_iterations: int = 5000   # Shorter than full training
    stage2_max_iterations: int = 10000  # Shorter than full training

    # Evaluation intervals
    stage1_eval_interval: int = 200
    stage2_eval_interval: int = 500

    # Resource limits
    timeout_per_trial: int = 3600  # 1 hour max per trial

    # Reproducibility
    seed: int = 42

    # Early stopping
    patience: int = 10  # Stop study if no improvement for N trials
    trial_patience: int = 10  # Early stopping within trial

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch size
    batch_size: int = 64
    num_workers: int = 0


# =============================================================================
# HELPER CLASSES
# =============================================================================

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


# =============================================================================
# STAGE 1: AUTOENCODER OPTIMIZATION
# =============================================================================

class Stage1Objective:
    """
    Multi-objective optimization for Stage 1 autoencoder.

    Objectives:
        1. Reconstruction MSE (minimize)
        2. Latent space quality/smoothness (minimize)
    """

    def __init__(self, config: OptunaConfig, dataloader, val_dataloader=None):
        self.config = config
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(config.device)

    def __call__(self, trial: optuna.Trial) -> Tuple[float, float]:
        """
        Returns tuple of objectives: (reconstruction_mse, latent_quality)
        """
        try:
            # Define search space and build model
            hparams = self._suggest_hyperparameters(trial)

            # Store config for analysis
            trial.set_user_attr("full_config", hparams)
            trial.set_user_attr("start_time", datetime.now().isoformat())

            # Build model
            model_config = self._build_config(hparams)
            model = TimeGANV6(model_config)
            model.to(self.device)

            # Log parameter count
            param_count = sum(p.numel() for p in model.parameters())
            trial.set_user_attr("param_count", param_count)

            # Training setup
            optimizer = self._build_optimizer(model, hparams)
            scheduler = self._build_scheduler(optimizer, hparams)

            # Training state
            loss_recon_avg = MovingAverage(100)
            loss_latent_avg = MovingAverage(100)
            best_val_loss = float('inf')
            patience_counter = 0

            max_iterations = self.config.stage1_max_iterations
            eval_interval = self.config.stage1_eval_interval

            # Infinite data iterator
            data_iter = self._infinite_loader()

            model.train()

            for iteration in range(1, max_iterations + 1):
                # Training step
                x_real, condition, lengths = next(data_iter)
                x_real = x_real.to(self.device)
                lengths = lengths.to(self.device)

                # Forward pass
                x_recon, losses = model.forward_autoencoder(x_real, lengths)

                # Backward pass
                optimizer.zero_grad()
                losses['total'].backward()

                # Gradient clipping
                if hparams.get('grad_clip', 1.0) > 0:
                    nn.utils.clip_grad_norm_(
                        model.get_autoencoder_parameters(),
                        hparams['grad_clip']
                    )

                optimizer.step()

                if scheduler is not None and hparams['lr_schedule'] != 'reduce_on_plateau':
                    scheduler.step()

                # Update metrics
                loss_recon_avg.update(losses['recon'].item())
                loss_latent_avg.update(losses['latent'].item())

                # Evaluation at intervals
                if iteration % eval_interval == 0:
                    val_mse, latent_quality = self._evaluate(model)

                    # Scheduler step for plateau
                    if scheduler is not None and hparams['lr_schedule'] == 'reduce_on_plateau':
                        scheduler.step(val_mse)

                    step = iteration // eval_interval

                    # Report for pruning
                    trial.report(val_mse, step)

                    # Log intermediate values
                    trial.set_user_attr(f"val_mse_step_{step}", val_mse)
                    trial.set_user_attr(f"latent_quality_step_{step}", latent_quality)

                    # Check pruning
                    if trial.should_prune():
                        raise TrialPruned()

                    # Early stopping within trial
                    if val_mse < best_val_loss:
                        best_val_loss = val_mse
                        patience_counter = 0
                        self._save_checkpoint(model, trial, hparams)
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.trial_patience:
                            trial.set_user_attr("stopped_early", True)
                            trial.set_user_attr("stopped_at_iteration", iteration)
                            break

            # Final evaluation
            final_mse, final_latent = self._evaluate(model)

            trial.set_user_attr("final_mse", final_mse)
            trial.set_user_attr("final_latent_quality", final_latent)
            trial.set_user_attr("end_time", datetime.now().isoformat())
            trial.set_user_attr("total_iterations", iteration)

            return (final_mse, final_latent)

        except Exception as e:
            trial.set_user_attr("error", str(e))
            trial.set_user_attr("traceback", traceback.format_exc())
            raise

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Stage 1 search space matching V6 architecture."""

        hparams = {}

        # === Architecture ===
        hparams['latent_dim'] = trial.suggest_categorical("latent_dim", [32, 48, 64, 96])
        hparams['summary_dim'] = trial.suggest_categorical("summary_dim", [64, 96, 128, 192])

        # Encoder/Decoder
        hparams['encoder_hidden_dim'] = trial.suggest_categorical("encoder_hidden_dim", [48, 64, 96, 128])
        hparams['encoder_num_layers'] = trial.suggest_int("encoder_num_layers", 2, 4)
        hparams['decoder_hidden_dim'] = trial.suggest_categorical("decoder_hidden_dim", [48, 64, 96, 128])
        hparams['decoder_num_layers'] = trial.suggest_int("decoder_num_layers", 2, 4)

        # Pooler
        hparams['pool_type'] = trial.suggest_categorical("pool_type", ["attention", "mean", "last", "hybrid"])
        hparams['pooler_dropout'] = trial.suggest_float("pooler_dropout", 0.0, 0.3)

        # Expander
        hparams['expand_type'] = trial.suggest_categorical("expand_type", ["lstm", "mlp", "repeat"])
        hparams['expander_hidden_dim'] = trial.suggest_categorical("expander_hidden_dim", [64, 128, 192, 256])
        hparams['expander_num_layers'] = trial.suggest_int("expander_num_layers", 1, 3)
        hparams['expander_dropout'] = trial.suggest_float("expander_dropout", 0.0, 0.3)

        # === Training ===
        hparams['optimizer'] = trial.suggest_categorical("optimizer", ["adam", "adamw", "radam"])
        hparams['lr'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        hparams['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # Learning rate schedule
        hparams['lr_schedule'] = trial.suggest_categorical(
            "lr_schedule", ["none", "cosine", "step", "reduce_on_plateau"]
        )

        if hparams['lr_schedule'] == 'cosine':
            hparams['warmup_steps'] = trial.suggest_int("warmup_steps", 100, 500)
        elif hparams['lr_schedule'] == 'step':
            hparams['lr_step_size'] = trial.suggest_int("lr_step_size", 500, 2000)
            hparams['lr_gamma'] = trial.suggest_float("lr_gamma", 0.1, 0.9)

        # === Loss Weights ===
        hparams['lambda_recon'] = trial.suggest_float("lambda_recon", 0.5, 2.0)
        hparams['lambda_latent'] = trial.suggest_float("lambda_latent", 0.1, 1.0)

        # === Regularization ===
        hparams['grad_clip'] = trial.suggest_float("grad_clip", 0.5, 2.0)

        return hparams

    def _build_config(self, hparams: Dict[str, Any]) -> TimeGANV6Config:
        """Build TimeGANV6Config from hyperparameters."""
        return TimeGANV6Config(
            # Architecture
            latent_dim=hparams['latent_dim'],
            summary_dim=hparams['summary_dim'],
            encoder_hidden_dim=hparams['encoder_hidden_dim'],
            encoder_num_layers=hparams['encoder_num_layers'],
            decoder_hidden_dim=hparams['decoder_hidden_dim'],
            decoder_num_layers=hparams['decoder_num_layers'],
            pool_type=hparams['pool_type'],
            pooler_dropout=hparams['pooler_dropout'],
            expand_type=hparams['expand_type'],
            expander_hidden_dim=hparams['expander_hidden_dim'],
            expander_num_layers=hparams['expander_num_layers'],
            expander_dropout=hparams['expander_dropout'],
            # Training
            lr_autoencoder=hparams['lr'],
            lambda_recon=hparams['lambda_recon'],
            lambda_latent=hparams['lambda_latent'],
            stage1_grad_clip=hparams['grad_clip'],
            stage1_weight_decay=hparams['weight_decay'],
            # Device
            device=self.config.device,
        )

    def _build_optimizer(self, model: TimeGANV6, hparams: Dict[str, Any]):
        """Build optimizer from hyperparameters."""
        opt_class = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "radam": torch.optim.RAdam,
        }[hparams['optimizer']]

        return opt_class(
            model.get_autoencoder_parameters(),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay']
        )

    def _build_scheduler(self, optimizer, hparams: Dict[str, Any]):
        """Build LR scheduler from hyperparameters."""
        schedule = hparams['lr_schedule']

        if schedule == 'none':
            return None
        elif schedule == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=hparams['warmup_steps']
            )
        elif schedule == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hparams['lr_step_size'],
                gamma=hparams['lr_gamma']
            )
        elif schedule == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )
        return None

    def _infinite_loader(self):
        """Create infinite iterator from DataLoader."""
        while True:
            for batch in self.dataloader:
                yield batch

    @torch.no_grad()
    def _evaluate(self, model: TimeGANV6) -> Tuple[float, float]:
        """
        Evaluate model quality.

        Returns:
            (reconstruction_mse, latent_quality)
        """
        model.eval()

        eval_loader = self.val_dataloader if self.val_dataloader else self.dataloader

        recon_losses = []
        latent_qualities = []

        for i, (x_real, condition, lengths) in enumerate(eval_loader):
            if i >= 10:  # Limit evaluation batches
                break

            x_real = x_real.to(self.device)
            lengths = lengths.to(self.device)

            # Get reconstruction loss
            x_recon, losses, intermediates = model.forward_autoencoder(
                x_real, lengths, return_intermediates=True
            )
            recon_losses.append(losses['recon'].item())

            # Latent space quality: measure smoothness
            # (variance of latent differences = how noisy is the latent space)
            z_summary = intermediates['z_summary']
            if z_summary.size(0) > 1:
                # Pairwise differences in latent space
                z_diffs = torch.cdist(z_summary, z_summary)
                latent_var = z_diffs.var().item()
                latent_qualities.append(latent_var)

        model.train()

        avg_recon = sum(recon_losses) / len(recon_losses) if recon_losses else float('inf')
        avg_latent = sum(latent_qualities) / len(latent_qualities) if latent_qualities else 1.0

        return avg_recon, avg_latent

    def _save_checkpoint(self, model: TimeGANV6, trial: optuna.Trial, hparams: Dict):
        """Save model checkpoint for this trial."""
        checkpoint_dir = Path(self.config.checkpoint_dir) / f"stage1_trial_{trial.number}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'hparams': hparams,
            'trial_number': trial.number,
        }, checkpoint_dir / "best_model.pt")


# =============================================================================
# STAGE 2: WGAN-GP OPTIMIZATION
# =============================================================================

class Stage2Objective:
    """
    Multi-objective optimization for Stage 2 WGAN-GP.

    Objectives:
        1. MMD between real and generated latent distributions (minimize)
        2. Discriminative score (minimize - closer to 0.5 is better)
    """

    def __init__(self, config: OptunaConfig, frozen_autoencoder_checkpoint: str,
                 dataloader, val_dataloader=None):
        self.config = config
        self.frozen_checkpoint = frozen_autoencoder_checkpoint
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(config.device)

        # Load base config from checkpoint
        checkpoint = torch.load(frozen_checkpoint, map_location=self.device)
        self.base_hparams = checkpoint.get('hparams', {})

    def __call__(self, trial: optuna.Trial) -> Tuple[float, float]:
        """
        Returns tuple of objectives: (mmd, discriminative_score)
        """
        try:
            hparams = self._suggest_hyperparameters(trial)
            trial.set_user_attr("full_config", hparams)
            trial.set_user_attr("start_time", datetime.now().isoformat())

            # Build model with Stage 1 frozen
            model = self._build_model(hparams)
            model.to(self.device)

            # Load and freeze autoencoder
            checkpoint = torch.load(self.frozen_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.freeze_autoencoder()

            # Build optimizers
            opt_G, opt_D = self._build_optimizers(model, hparams)

            # Training state
            d_loss_avg = MovingAverage(100)
            g_loss_avg = MovingAverage(100)
            wasserstein_avg = MovingAverage(100)

            max_iterations = self.config.stage2_max_iterations
            eval_interval = self.config.stage2_eval_interval

            data_iter = self._infinite_loader()
            training_stable = True

            model.train()

            for iteration in range(1, max_iterations + 1):
                # === Discriminator Updates ===
                for _ in range(hparams['n_critic']):
                    x_real, condition, lengths = next(data_iter)
                    x_real = x_real.to(self.device)
                    condition = condition.to(self.device)
                    lengths = lengths.to(self.device)

                    d_losses = model.forward_discriminator(x_real, condition, lengths)

                    opt_D.zero_grad()
                    d_losses['d_loss'].backward()

                    if hparams.get('grad_clip', 1.0) > 0:
                        nn.utils.clip_grad_norm_(
                            model.get_discriminator_parameters(),
                            hparams['grad_clip']
                        )

                    opt_D.step()

                d_loss_avg.update(d_losses['d_loss'].item())
                wasserstein_avg.update(d_losses['wasserstein'].item())

                # === Generator Update ===
                x_real, condition, lengths = next(data_iter)
                x_real = x_real.to(self.device)
                condition = condition.to(self.device)
                lengths = lengths.to(self.device)

                batch_size = x_real.size(0)

                g_losses = model.forward_generator(
                    batch_size, condition,
                    x_real if hparams.get('use_feature_matching', True) else None,
                    lengths if hparams.get('use_feature_matching', True) else None
                )

                opt_G.zero_grad()
                g_losses['g_loss'].backward()

                if hparams.get('grad_clip', 1.0) > 0:
                    nn.utils.clip_grad_norm_(
                        model.get_generator_parameters(),
                        hparams['grad_clip']
                    )

                opt_G.step()

                g_loss_avg.update(g_losses['g_loss'].item())

                # Check for NaN (training collapse)
                if torch.isnan(torch.tensor(d_loss_avg.get())) or torch.isnan(torch.tensor(g_loss_avg.get())):
                    trial.set_user_attr("training_collapsed", True)
                    trial.set_user_attr("collapse_iteration", iteration)
                    training_stable = False
                    break

                # Evaluation
                if iteration % eval_interval == 0:
                    mmd, disc_score = self._evaluate(model)

                    step = iteration // eval_interval

                    # Report for pruning (use MMD as primary metric)
                    trial.report(mmd, step)

                    trial.set_user_attr(f"mmd_step_{step}", mmd)
                    trial.set_user_attr(f"disc_score_step_{step}", disc_score)

                    if trial.should_prune():
                        raise TrialPruned()

            # Final evaluation
            if training_stable:
                final_mmd, final_disc = self._evaluate(model)
                self._save_checkpoint(model, trial, hparams)
            else:
                final_mmd = 10.0
                final_disc = 1.0

            trial.set_user_attr("final_mmd", final_mmd)
            trial.set_user_attr("final_disc_score", final_disc)
            trial.set_user_attr("end_time", datetime.now().isoformat())

            return (final_mmd, final_disc)

        except Exception as e:
            trial.set_user_attr("error", str(e))
            trial.set_user_attr("traceback", traceback.format_exc())
            raise

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Stage 2 search space."""

        hparams = {}

        # === Generator ===
        hparams['noise_dim'] = trial.suggest_categorical("noise_dim", [64, 128, 192, 256])
        hparams['generator_hidden_dims'] = trial.suggest_categorical(
            "generator_hidden_dims",
            [[128, 128], [256, 256], [256, 256, 256], [512, 256], [512, 512, 256]]
        )
        hparams['generator_use_residual'] = trial.suggest_categorical("generator_use_residual", [True, False])
        hparams['generator_type'] = trial.suggest_categorical("generator_type", ["standard", "film"])

        # === Discriminator ===
        hparams['discriminator_hidden_dims'] = trial.suggest_categorical(
            "discriminator_hidden_dims",
            [[128, 128], [256, 256], [256, 256, 256], [512, 256], [512, 512, 256]]
        )
        hparams['discriminator_use_spectral_norm'] = trial.suggest_categorical(
            "discriminator_use_spectral_norm", [True, False]
        )
        hparams['discriminator_type'] = trial.suggest_categorical(
            "discriminator_type", ["standard", "projection"]
        )

        # === Training ===
        hparams['lr_G'] = trial.suggest_float("lr_G", 1e-5, 1e-3, log=True)
        hparams['lr_D'] = trial.suggest_float("lr_D", 1e-5, 1e-3, log=True)
        hparams['n_critic'] = trial.suggest_int("n_critic", 1, 10)

        # WGAN-GP
        hparams['lambda_gp'] = trial.suggest_float("lambda_gp", 1.0, 50.0, log=True)

        # Feature matching
        hparams['use_feature_matching'] = trial.suggest_categorical("use_feature_matching", [True, False])
        if hparams['use_feature_matching']:
            hparams['lambda_fm'] = trial.suggest_float("lambda_fm", 0.1, 10.0, log=True)
        else:
            hparams['lambda_fm'] = 0.0

        # Optimizer
        hparams['beta1'] = trial.suggest_float("beta1", 0.0, 0.5)
        hparams['beta2'] = trial.suggest_float("beta2", 0.9, 0.999)

        # Regularization
        hparams['grad_clip'] = trial.suggest_float("grad_clip", 0.5, 2.0)

        return hparams

    def _build_model(self, hparams: Dict[str, Any]) -> TimeGANV6:
        """Build TimeGANV6 with Stage 2 hyperparameters."""

        # Merge with base hparams from Stage 1
        config = TimeGANV6Config(
            # From Stage 1 checkpoint
            latent_dim=self.base_hparams.get('latent_dim', 48),
            summary_dim=self.base_hparams.get('summary_dim', 96),
            encoder_hidden_dim=self.base_hparams.get('encoder_hidden_dim', 64),
            encoder_num_layers=self.base_hparams.get('encoder_num_layers', 3),
            decoder_hidden_dim=self.base_hparams.get('decoder_hidden_dim', 64),
            decoder_num_layers=self.base_hparams.get('decoder_num_layers', 3),
            pool_type=self.base_hparams.get('pool_type', 'attention'),
            expand_type=self.base_hparams.get('expand_type', 'lstm'),
            expander_hidden_dim=self.base_hparams.get('expander_hidden_dim', 128),
            expander_num_layers=self.base_hparams.get('expander_num_layers', 2),
            # Stage 2 hyperparameters
            noise_dim=hparams['noise_dim'],
            generator_hidden_dims=hparams['generator_hidden_dims'],
            generator_use_residual=hparams['generator_use_residual'],
            generator_type=hparams['generator_type'],
            discriminator_hidden_dims=hparams['discriminator_hidden_dims'],
            discriminator_use_spectral_norm=hparams['discriminator_use_spectral_norm'],
            discriminator_type=hparams['discriminator_type'],
            lr_generator=hparams['lr_G'],
            lr_discriminator=hparams['lr_D'],
            n_critic=hparams['n_critic'],
            lambda_gp=hparams['lambda_gp'],
            lambda_fm=hparams['lambda_fm'],
            use_feature_matching=hparams['use_feature_matching'],
            adam_beta1=hparams['beta1'],
            adam_beta2=hparams['beta2'],
            stage2_grad_clip=hparams['grad_clip'],
            device=self.config.device,
        )

        return TimeGANV6(config)

    def _build_optimizers(self, model: TimeGANV6, hparams: Dict[str, Any]):
        """Build generator and discriminator optimizers."""
        opt_G = torch.optim.Adam(
            model.get_generator_parameters(),
            lr=hparams['lr_G'],
            betas=(hparams['beta1'], hparams['beta2'])
        )
        opt_D = torch.optim.Adam(
            model.get_discriminator_parameters(),
            lr=hparams['lr_D'],
            betas=(hparams['beta1'], hparams['beta2'])
        )
        return opt_G, opt_D

    def _infinite_loader(self):
        """Create infinite iterator from DataLoader."""
        while True:
            for batch in self.dataloader:
                yield batch

    @torch.no_grad()
    def _evaluate(self, model: TimeGANV6) -> Tuple[float, float]:
        """
        Evaluate generator quality.

        Returns:
            (mmd, discriminative_score)
        """
        model.eval()

        eval_loader = self.val_dataloader if self.val_dataloader else self.dataloader

        # Collect real and fake latents
        z_reals = []
        z_fakes = []

        for i, (x_real, condition, lengths) in enumerate(eval_loader):
            if i >= 10:
                break

            x_real = x_real.to(self.device)
            condition = condition.to(self.device)
            lengths = lengths.to(self.device)

            # Real latents
            z_real = model.encode(x_real, lengths)
            z_reals.append(z_real)

            # Fake latents
            batch_size = x_real.size(0)
            noise = torch.randn(batch_size, model.config.noise_dim, device=self.device)
            z_fake = model.generator(noise, condition)
            z_fakes.append(z_fake)

        model.train()

        z_real_all = torch.cat(z_reals, dim=0)
        z_fake_all = torch.cat(z_fakes, dim=0)

        # Compute MMD
        mmd = self._compute_mmd(z_real_all, z_fake_all)

        # Simple discriminative proxy: mean discriminator output difference
        # (closer to 0 means D can't distinguish)
        disc_score = self._compute_disc_score(model, z_real_all, z_fake_all)

        return mmd, disc_score

    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Maximum Mean Discrepancy."""
        n, m = x.size(0), y.size(0)

        # RBF kernel with median heuristic
        xy = torch.cat([x, y], dim=0)
        dists = torch.cdist(xy, xy)
        median_dist = dists.median()
        gamma = 1.0 / (2 * median_dist ** 2 + 1e-8)

        xx = torch.exp(-gamma * torch.cdist(x, x) ** 2)
        yy = torch.exp(-gamma * torch.cdist(y, y) ** 2)
        xy_dist = torch.exp(-gamma * torch.cdist(x, y) ** 2)

        mmd_sq = xx.sum() / (n * n) + yy.sum() / (m * m) - 2 * xy_dist.sum() / (n * m)

        return max(0, mmd_sq.item()) ** 0.5

    def _compute_disc_score(self, model: TimeGANV6, z_real: torch.Tensor,
                            z_fake: torch.Tensor) -> float:
        """Compute discriminator-based score."""
        # Dummy condition for evaluation
        condition = torch.zeros(z_real.size(0), 1, device=self.device)

        d_real = model.discriminator(z_real, condition).mean().item()
        d_fake = model.discriminator(z_fake, condition[:z_fake.size(0)]).mean().item()

        # Return absolute difference (lower = can't distinguish)
        return abs(d_real - d_fake)

    def _save_checkpoint(self, model: TimeGANV6, trial: optuna.Trial, hparams: Dict):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir) / f"stage2_trial_{trial.number}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'hparams': hparams,
            'trial_number': trial.number,
        }, checkpoint_dir / "best_model.pt")


# =============================================================================
# STUDY MANAGER
# =============================================================================

class StudyManager:
    """Manages Optuna studies with best practices."""

    def __init__(self, config: OptunaConfig):
        self.config = config
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        optuna.logging.set_verbosity(optuna.logging.INFO)

        log_path = Path(self.config.tensorboard_dir) / "optuna.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        optuna.logging.get_logger("optuna").addHandler(file_handler)

    def create_stage1_study(self, study_name: str = "timegan_v6_stage1") -> optuna.Study:
        """Create multi-objective study for Stage 1."""

        sampler = TPESampler(
            seed=self.config.seed,
            n_startup_trials=10,
            multivariate=True,
        )

        pruner = HyperbandPruner(
            min_resource=3,
            max_resource=self.config.stage1_max_iterations // self.config.stage1_eval_interval,
            reduction_factor=3,
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=self.config.storage_path,
            sampler=sampler,
            pruner=pruner,
            directions=["minimize", "minimize"],  # (recon_mse, latent_quality)
            load_if_exists=True,
        )

        # Enqueue baseline
        self._enqueue_stage1_baseline(study)

        return study

    def create_stage2_study(self, study_name: str = "timegan_v6_stage2") -> optuna.Study:
        """Create multi-objective study for Stage 2."""

        sampler = TPESampler(
            seed=self.config.seed,
            n_startup_trials=15,
            multivariate=True,
        )

        pruner = HyperbandPruner(
            min_resource=2,
            max_resource=self.config.stage2_max_iterations // self.config.stage2_eval_interval,
            reduction_factor=3,
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=self.config.storage_path,
            sampler=sampler,
            pruner=pruner,
            directions=["minimize", "minimize"],  # (mmd, disc_score)
            load_if_exists=True,
        )

        self._enqueue_stage2_baseline(study)

        return study

    def _enqueue_stage1_baseline(self, study: optuna.Study):
        """Enqueue known good Stage 1 configurations."""
        if len([t for t in study.trials if t.state == TrialState.COMPLETE]) > 0:
            return

        baseline = {
            "latent_dim": 48,
            "summary_dim": 96,
            "encoder_hidden_dim": 64,
            "encoder_num_layers": 3,
            "decoder_hidden_dim": 64,
            "decoder_num_layers": 3,
            "pool_type": "attention",
            "pooler_dropout": 0.1,
            "expand_type": "lstm",
            "expander_hidden_dim": 128,
            "expander_num_layers": 2,
            "expander_dropout": 0.1,
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "lr_schedule": "none",
            "lambda_recon": 1.0,
            "lambda_latent": 0.5,
            "grad_clip": 1.0,
        }
        study.enqueue_trial(baseline)

    def _enqueue_stage2_baseline(self, study: optuna.Study):
        """Enqueue known good Stage 2 configurations."""
        if len([t for t in study.trials if t.state == TrialState.COMPLETE]) > 0:
            return

        baseline = {
            "noise_dim": 128,
            "generator_hidden_dims": [256, 256, 256],
            "generator_use_residual": True,
            "generator_type": "standard",
            "discriminator_hidden_dims": [256, 256, 256],
            "discriminator_use_spectral_norm": False,
            "discriminator_type": "standard",
            "lr_G": 1e-4,
            "lr_D": 1e-4,
            "n_critic": 5,
            "lambda_gp": 10.0,
            "use_feature_matching": True,
            "lambda_fm": 1.0,
            "beta1": 0.0,
            "beta2": 0.9,
            "grad_clip": 1.0,
        }
        study.enqueue_trial(baseline)

    def run_stage1(self, objective: Stage1Objective, n_trials: int = None,
                   study_name: str = "timegan_v6_stage1") -> optuna.Study:
        """Run Stage 1 optimization."""

        study = self.create_stage1_study(study_name)
        n_trials = n_trials or self.config.stage1_trials

        callbacks = [EarlyStoppingCallback(patience=self.config.patience)]

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_trial * n_trials,
            callbacks=callbacks,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        return study

    def run_stage2(self, objective: Stage2Objective, n_trials: int = None,
                   study_name: str = "timegan_v6_stage2") -> optuna.Study:
        """Run Stage 2 optimization."""

        study = self.create_stage2_study(study_name)
        n_trials = n_trials or self.config.stage2_trials

        callbacks = [EarlyStoppingCallback(patience=self.config.patience)]

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_trial * n_trials,
            callbacks=callbacks,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        return study


# =============================================================================
# CALLBACKS
# =============================================================================

class EarlyStoppingCallback:
    """Stop study if no improvement for `patience` trials."""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_values = None
        self.no_improvement_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != TrialState.COMPLETE:
            return

        current_values = trial.values

        if self.best_values is None:
            self.best_values = current_values
            return

        improved = any(c < b for c, b in zip(current_values, self.best_values))

        if improved:
            self.best_values = [min(c, b) for c, b in zip(current_values, self.best_values)]
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            study.stop()


# =============================================================================
# ANALYSIS
# =============================================================================

class StudyAnalyzer:
    """Comprehensive analysis of completed studies."""

    def __init__(self, study: optuna.Study):
        self.study = study

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print(f"Study: {self.study.study_name}")
        print("=" * 60)

        complete = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        pruned = [t for t in self.study.trials if t.state == TrialState.PRUNED]
        failed = [t for t in self.study.trials if t.state == TrialState.FAIL]

        print(f"Total trials: {len(self.study.trials)}")
        print(f"  Complete: {len(complete)}")
        print(f"  Pruned: {len(pruned)}")
        print(f"  Failed: {len(failed)}")

        if complete:
            pareto_trials = self.study.best_trials
            print(f"\nPareto front size: {len(pareto_trials)}")

            print("\nBest trials (Pareto front):")
            for i, trial in enumerate(pareto_trials[:5]):
                print(f"\n  Trial {trial.number}:")
                print(f"    Values: {trial.values}")
                print(f"    Key params:")
                for key in ['latent_dim', 'lr', 'pool_type', 'n_critic', 'lambda_gp']:
                    if key in trial.params:
                        print(f"      {key}: {trial.params[key]}")

    def plot_all(self, save_dir: str = "./optuna_plots"):
        """Generate visualization plots."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            import optuna.visualization as viz

            # Parameter importances
            try:
                fig = viz.plot_param_importances(self.study)
                fig.write_html(save_path / "param_importances.html")
            except Exception as e:
                print(f"Could not plot param importances: {e}")

            # Optimization history
            try:
                fig = viz.plot_optimization_history(self.study)
                fig.write_html(save_path / "optimization_history.html")
            except Exception as e:
                print(f"Could not plot optimization history: {e}")

            # Pareto front
            try:
                fig = viz.plot_pareto_front(self.study)
                fig.write_html(save_path / "pareto_front.html")
            except Exception as e:
                print(f"Could not plot Pareto front: {e}")

            # Parallel coordinate
            try:
                fig = viz.plot_parallel_coordinate(self.study)
                fig.write_html(save_path / "parallel_coordinate.html")
            except Exception as e:
                print(f"Could not plot parallel coordinate: {e}")

            print(f"\nPlots saved to {save_path}")

        except ImportError:
            print("optuna visualization requires plotly. Install with: pip install plotly")

    def export_results(self, filepath: str):
        """Export all trial results to JSON."""
        results = {
            "study_name": self.study.study_name,
            "directions": [str(d) for d in self.study.directions],
            "n_trials": len(self.study.trials),
            "best_trials": [],
            "all_trials": [],
        }

        for trial in self.study.best_trials:
            results["best_trials"].append({
                "number": trial.number,
                "values": trial.values,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            })

        for trial in self.study.trials:
            results["all_trials"].append({
                "number": trial.number,
                "state": str(trial.state),
                "values": trial.values if trial.values else None,
                "params": trial.params,
            })

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results exported to {filepath}")

    def get_best_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """Get path to best trial's checkpoint."""
        if not self.study.best_trials:
            return None

        best_trial = self.study.best_trials[0]
        stage = "stage1" if "stage1" in self.study.study_name else "stage2"
        checkpoint_path = Path(checkpoint_dir) / f"{stage}_trial_{best_trial.number}" / "best_model.pt"

        if checkpoint_path.exists():
            return str(checkpoint_path)
        return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main optimization pipeline."""

    parser = argparse.ArgumentParser(description="Optuna optimization for TimeGAN V6")
    parser.add_argument("--stage", type=str, default="1", choices=["1", "2", "both"],
                        help="Which stage to optimize")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--data-dir", type=str, default="./processed_data_v4",
                        help="Path to preprocessed data")
    parser.add_argument("--stage1-checkpoint", type=str, default=None,
                        help="Stage 1 checkpoint for Stage 2 optimization")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing study")
    parser.add_argument("--study-name", type=str, default=None, help="Study name for analysis")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Setup config
    config = OptunaConfig(
        data_dir=args.data_dir,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Validate data
    print(f"\nValidating data directory: {config.data_dir}")
    validate_v6_data(config.data_dir)

    manager = StudyManager(config)

    # Analysis mode
    if args.analyze:
        study_name = args.study_name or f"timegan_v6_stage{args.stage}"
        study = optuna.load_study(
            study_name=study_name,
            storage=config.storage_path,
        )
        analyzer = StudyAnalyzer(study)
        analyzer.print_summary()
        analyzer.plot_all(f"./plots/{study_name}")
        analyzer.export_results(f"./results/{study_name}_results.json")
        return

    # Stage 1 Optimization
    if args.stage in ["1", "both"]:
        print("\n" + "=" * 60)
        print("STAGE 1: Autoencoder Optimization")
        print("=" * 60)

        # Create dataloaders
        stage1_loader = create_stage1_loader(
            config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            augment=True,
            quality_filter=True
        )

        train_loader, val_loader, _ = create_dataloaders_v6(
            config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        stage1_objective = Stage1Objective(config, stage1_loader, val_loader)
        stage1_study = manager.run_stage1(stage1_objective, n_trials=args.n_trials)

        # Analyze
        analyzer1 = StudyAnalyzer(stage1_study)
        analyzer1.print_summary()

        # Get best checkpoint for Stage 2
        best_stage1_checkpoint = analyzer1.get_best_checkpoint(config.checkpoint_dir)
        print(f"\nBest Stage 1 checkpoint: {best_stage1_checkpoint}")

    # Stage 2 Optimization
    if args.stage in ["2", "both"]:
        print("\n" + "=" * 60)
        print("STAGE 2: WGAN-GP Optimization")
        print("=" * 60)

        # Get Stage 1 checkpoint
        if args.stage == "2":
            if not args.stage1_checkpoint:
                print("ERROR: Stage 2 requires --stage1-checkpoint")
                return
            stage1_checkpoint = args.stage1_checkpoint
        else:
            stage1_checkpoint = best_stage1_checkpoint

        if not stage1_checkpoint or not os.path.exists(stage1_checkpoint):
            print(f"ERROR: Stage 1 checkpoint not found: {stage1_checkpoint}")
            return

        print(f"Using Stage 1 checkpoint: {stage1_checkpoint}")

        # Create dataloaders
        stage2_loader = create_stage2_loader(
            config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            augment=False,
            quality_filter=True
        )

        _, val_loader, _ = create_dataloaders_v6(
            config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        stage2_objective = Stage2Objective(
            config, stage1_checkpoint, stage2_loader, val_loader
        )
        stage2_study = manager.run_stage2(stage2_objective, n_trials=args.n_trials)

        # Analyze
        analyzer2 = StudyAnalyzer(stage2_study)
        analyzer2.print_summary()

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
