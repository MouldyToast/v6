"""
Inpainting Reference Implementation for V8 Trajectory Diffusion

This file provides complete, copy-pasteable code for:
1. Inpainting-based trajectory generation
2. Training without goal conditioning
3. Preprocessing for 2D features

Reference: Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" (ICML 2022)

Usage:
    This is a REFERENCE file. The actual implementation should be in:
    - diffusion_v8/sampling/inpainting_sampler.py
    - diffusion_v8/trainers/trajectory_trainer.py
    - preprocess_V8.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# INPAINTING SAMPLER
# =============================================================================

class InpaintingSampler:
    """
    Generates trajectories using endpoint inpainting.

    At each denoising step:
    1. Predict clean trajectory from noisy input
    2. Replace start position with (0, 0)
    3. Replace end position with target endpoint
    4. Add noise for next step (except final step)

    This guarantees the trajectory reaches the target endpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion,  # GaussianDiffusion instance
        max_seq_len: int = 200,
        device: str = 'cuda'
    ):
        self.model = model
        self.diffusion = diffusion
        self.max_seq_len = max_seq_len
        self.device = device

    def generate(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        length: int,
        ddim_steps: int = 50,
        eta: float = 0.0,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate a single trajectory from start_pos to end_pos.

        Args:
            start_pos: Starting position (typically (0, 0))
            end_pos: Target endpoint (normalized coordinates)
            length: Trajectory length (number of timesteps)
            ddim_steps: Number of DDIM sampling steps
            eta: DDIM eta (0 = deterministic)
            show_progress: Show progress bar

        Returns:
            trajectory: (length, 2) array of positions
        """
        self.model.eval()

        with torch.no_grad():
            # Convert to tensors
            start = torch.tensor(start_pos, device=self.device, dtype=torch.float32)
            end = torch.tensor(end_pos, device=self.device, dtype=torch.float32)
            length_tensor = torch.tensor([length], device=self.device)

            # Start from pure noise
            x = torch.randn(1, self.max_seq_len, 2, device=self.device)

            # Get DDIM timestep schedule
            timesteps = self._get_ddim_timesteps(ddim_steps)

            # Iteratively denoise with inpainting
            iterator = reversed(list(enumerate(timesteps)))
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Generating")

            for i, t in iterator:
                t_batch = torch.tensor([t], device=self.device)

                # 1. Predict noise
                noise_pred = self.model(x, t_batch, length_tensor)

                # 2. DDIM step to get x_0 prediction
                x_0_pred = self._predict_x0(x, t, noise_pred)

                # 3. INPAINTING: Replace endpoints
                x_0_pred[0, 0, :] = start
                x_0_pred[0, length - 1, :] = end

                # 4. Add noise for next step (except final)
                if i > 0:
                    t_prev = timesteps[len(timesteps) - i]
                    x = self._add_noise(x_0_pred, t_prev, eta)
                else:
                    x = x_0_pred

            # Extract actual trajectory (remove padding)
            trajectory = x[0, :length, :].cpu().numpy()

        return trajectory

    def generate_batch(
        self,
        endpoints: np.ndarray,
        lengths: np.ndarray,
        ddim_steps: int = 50
    ) -> np.ndarray:
        """
        Generate multiple trajectories in parallel.

        Args:
            endpoints: (batch, 2) array of target endpoints
            lengths: (batch,) array of trajectory lengths
            ddim_steps: DDIM steps

        Returns:
            trajectories: (batch, max_len, 2) array
        """
        batch_size = len(endpoints)
        self.model.eval()

        with torch.no_grad():
            # Convert to tensors
            ends = torch.tensor(endpoints, device=self.device, dtype=torch.float32)
            lens = torch.tensor(lengths, device=self.device)
            starts = torch.zeros(batch_size, 2, device=self.device)

            # Start from noise
            x = torch.randn(batch_size, self.max_seq_len, 2, device=self.device)

            timesteps = self._get_ddim_timesteps(ddim_steps)

            for i, t in enumerate(reversed(timesteps)):
                t_batch = torch.full((batch_size,), t, device=self.device)

                noise_pred = self.model(x, t_batch, lens)
                x_0_pred = self._predict_x0(x, t, noise_pred)

                # Inpaint each trajectory in batch
                for b in range(batch_size):
                    x_0_pred[b, 0, :] = starts[b]
                    x_0_pred[b, lengths[b] - 1, :] = ends[b]

                if i < len(timesteps) - 1:
                    t_prev = timesteps[len(timesteps) - i - 2]
                    x = self._add_noise(x_0_pred, t_prev, eta=0.0)
                else:
                    x = x_0_pred

        return x.cpu().numpy()

    def _get_ddim_timesteps(self, num_steps: int) -> list:
        """Get evenly spaced timesteps for DDIM."""
        total_steps = self.diffusion.num_timesteps
        step_size = total_steps // num_steps
        timesteps = list(range(0, total_steps, step_size))[:num_steps]
        return timesteps

    def _predict_x0(self, x_t: torch.Tensor, t: int, noise_pred: torch.Tensor) -> torch.Tensor:
        """Predict clean sample from noisy sample and predicted noise."""
        alpha_t = self.diffusion.alphas_cumprod[t]
        sqrt_alpha = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

        x_0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        return x_0

    def _add_noise(self, x_0: torch.Tensor, t: int, eta: float = 0.0) -> torch.Tensor:
        """Add noise for timestep t (DDIM forward)."""
        alpha_t = self.diffusion.alphas_cumprod[t]
        sqrt_alpha = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

        noise = torch.randn_like(x_0) if eta > 0 else torch.zeros_like(x_0)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t


# =============================================================================
# TRAINING (NO GOAL CONDITIONING)
# =============================================================================

def train_step_v8(
    model: nn.Module,
    diffusion,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    device: str = 'cuda'
) -> float:
    """
    Single V8 training step - NO goal conditioning.

    Args:
        model: TrajectoryTransformer
        diffusion: GaussianDiffusion
        optimizer: Adam optimizer
        batch: Dict with 'positions' (B, L, 2) and 'lengths' (B,)
        device: Device

    Returns:
        loss: Training loss value
    """
    model.train()

    # Get data
    positions = batch['positions'].to(device)  # (B, max_len, 2)
    lengths = batch['lengths'].to(device)      # (B,)
    batch_size = positions.shape[0]

    # Sample random timesteps
    t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)

    # Sample noise
    noise = torch.randn_like(positions)

    # Forward diffusion: add noise to clean data
    x_noisy = diffusion.q_sample(positions, t, noise)

    # Predict noise (NO goal conditioning)
    noise_pred = model(x_noisy, t, lengths)

    # Compute loss with length masking
    mask = create_length_mask(lengths, positions.shape[1], device)
    loss = masked_mse_loss(noise_pred, noise, mask)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return loss.item()


def create_length_mask(lengths: torch.Tensor, max_len: int, device: str) -> torch.Tensor:
    """Create mask: 1 for valid positions, 0 for padding."""
    batch_size = lengths.shape[0]
    mask = torch.zeros(batch_size, max_len, 1, device=device)
    for i, length in enumerate(lengths):
        mask[i, :length, :] = 1.0
    return mask


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss only on valid (non-padded) positions."""
    sq_error = (pred - target) ** 2
    masked_error = sq_error * mask
    return masked_error.sum() / mask.sum()


# =============================================================================
# PREPROCESSING (2D FEATURES)
# =============================================================================

def preprocess_trajectory_v8(
    x: np.ndarray,
    y: np.ndarray,
    position_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a single trajectory for V8.

    Args:
        x: Raw x coordinates
        y: Raw y coordinates
        position_scale: Normalization scale factor

    Returns:
        positions: (L, 2) normalized positions relative to start
        endpoint: (2,) the final position (for evaluation)
    """
    # Make relative to start
    x_rel = x - x[0]
    y_rel = y - y[0]

    # Store endpoint before normalization (for evaluation)
    endpoint = np.array([x_rel[-1], y_rel[-1]])

    # Normalize to approximately [-1, 1]
    x_norm = x_rel / position_scale
    y_norm = y_rel / position_scale

    # Stack into (L, 2)
    positions = np.stack([x_norm, y_norm], axis=1)

    return positions, endpoint


def compute_position_scale(trajectories: list) -> float:
    """
    Compute normalization scale from training data.

    Uses max displacement across all trajectories.
    """
    max_displacement = 0
    for traj in trajectories:
        x, y = traj['x'], traj['y']
        x_rel = np.array(x) - x[0]
        y_rel = np.array(y) - y[0]
        displacement = np.sqrt(x_rel**2 + y_rel**2).max()
        max_displacement = max(max_displacement, displacement)

    # Add small margin
    return max_displacement * 1.1


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("V8 Inpainting Reference Implementation")
    print("=" * 50)
    print()
    print("This file provides reference code for:")
    print("  1. InpaintingSampler - Generate trajectories to target endpoints")
    print("  2. train_step_v8 - Training without goal conditioning")
    print("  3. preprocess_trajectory_v8 - Convert raw data to 2D positions")
    print()
    print("Key differences from V7:")
    print("  - No CFG (Classifier-Free Guidance)")
    print("  - No GoalConditioner")
    print("  - 2D features (x, y) instead of 8D")
    print("  - Direction controlled via endpoint inpainting")
    print()
    print("See DIFFUSION_V7_ARCHITECTURE.md for full documentation.")
