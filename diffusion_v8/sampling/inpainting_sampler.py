"""
Inpainting Sampler for V8 Trajectory Diffusion

Generates trajectories using endpoint inpainting instead of CFG.

At each denoising step:
1. Predict clean trajectory from noisy input
2. Replace start position with (0, 0)
3. Replace end position with target endpoint
4. Add noise for next step (except final step)

This guarantees the trajectory reaches the target endpoint exactly.

Reference: Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" (ICML 2022)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm


class InpaintingSampler:
    """
    Generates trajectories using endpoint inpainting.

    V8 Key Difference from V7:
    - No CFG (classifier-free guidance)
    - No goal embedding
    - Direction controlled by replacing endpoint at each denoising step

    Args:
        model: TrajectoryTransformer (V8)
        diffusion: GaussianDiffusion instance
        max_seq_len: Maximum trajectory length
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion,
        max_seq_len: int = 200,
        device: str = 'cuda'
    ):
        self.model = model
        self.diffusion = diffusion
        self.max_seq_len = max_seq_len
        self.device = torch.device(device) if isinstance(device, str) else device

    def generate(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        length: int,
        ddim_steps: int = 50,
        eta: float = 1.0,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate a single trajectory from start_pos to end_pos.

        Args:
            start_pos: Starting position (typically (0, 0))
            end_pos: Target endpoint (normalized coordinates)
            length: Trajectory length (number of timesteps)
            ddim_steps: Number of DDIM sampling steps
            eta: Noise scale for re-noising (default 1.0, must be >0 for inpainting)
            show_progress: Show progress bar

        Returns:
            trajectory: (length, 2) array of positions
        """
        self.model.eval()

        with torch.no_grad():
            # Convert to tensors
            start = torch.tensor(start_pos, device=self.device, dtype=torch.float32)
            end = torch.tensor(end_pos, device=self.device, dtype=torch.float32)
            length_tensor = torch.tensor([length], device=self.device, dtype=torch.long)

            # Start from pure noise
            x = torch.randn(1, self.max_seq_len, 2, device=self.device)

            # Get DDIM timestep schedule
            timesteps = self._get_ddim_timesteps(ddim_steps)

            # Iteratively denoise with inpainting
            iterator = list(enumerate(timesteps))
            if show_progress:
                iterator = tqdm(iterator, desc="Generating")

            for i, t in iterator:
                t_batch = torch.tensor([t], device=self.device, dtype=torch.long)

                # 1. Predict noise
                noise_pred = self.model(x, t_batch, length_tensor)

                # 2. DDIM step to get x_0 prediction
                x_0_pred = self._predict_x0(x, t, noise_pred)

                # 3. INPAINTING: Replace endpoints
                x_0_pred[0, 0, :] = start
                x_0_pred[0, length - 1, :] = end

                # 4. Add noise for next step (except final)
                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    x = self._add_noise(x_0_pred, t_next, eta)
                else:
                    x = x_0_pred

            # Extract actual trajectory (remove padding)
            trajectory = x[0, :length, :].cpu().numpy()

        return trajectory

    def generate_batch(
        self,
        endpoints: np.ndarray,
        lengths: np.ndarray,
        ddim_steps: int = 50,
        eta: float = 1.0,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate multiple trajectories in parallel.

        Args:
            endpoints: (batch, 2) array of target endpoints
            lengths: (batch,) array of trajectory lengths
            ddim_steps: DDIM steps
            eta: DDIM eta (0 = deterministic)
            show_progress: Show progress bar

        Returns:
            trajectories: (batch, max_len, 2) array
        """
        batch_size = len(endpoints)
        self.model.eval()

        with torch.no_grad():
            # Convert to tensors
            ends = torch.tensor(endpoints, device=self.device, dtype=torch.float32)
            lens = torch.tensor(lengths, device=self.device, dtype=torch.long)
            starts = torch.zeros(batch_size, 2, device=self.device)

            # Start from noise
            x = torch.randn(batch_size, self.max_seq_len, 2, device=self.device)

            timesteps = self._get_ddim_timesteps(ddim_steps)

            iterator = list(enumerate(timesteps))
            if show_progress:
                iterator = tqdm(iterator, desc="Generating batch")

            for i, t in iterator:
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

                noise_pred = self.model(x, t_batch, lens)
                x_0_pred = self._predict_x0(x, t, noise_pred)

                # Inpaint each trajectory in batch
                for b in range(batch_size):
                    length_b = int(lens[b].item())  # Use tensor, convert to int
                    x_0_pred[b, 0, :] = starts[b]
                    x_0_pred[b, length_b - 1, :] = ends[b]

                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    x = self._add_noise(x_0_pred, t_next, eta)
                else:
                    x = x_0_pred

        return x.cpu().numpy()

    def generate_single(
        self,
        target_x: float,
        target_y: float,
        length: int,
        ddim_steps: int = 50,
        eta: float = 1.0,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Convenience method to generate a single trajectory.

        Args:
            target_x: Normalized target x position
            target_y: Normalized target y position
            length: Trajectory length
            ddim_steps: DDIM steps
            eta: DDIM eta
            show_progress: Show progress bar

        Returns:
            trajectory: (length, 2) array
        """
        return self.generate(
            start_pos=(0.0, 0.0),
            end_pos=(target_x, target_y),
            length=length,
            ddim_steps=ddim_steps,
            eta=eta,
            show_progress=show_progress
        )

    def _get_ddim_timesteps(self, num_steps: int) -> List[int]:
        """
        Get evenly spaced timesteps for DDIM (reversed for denoising).

        Args:
            num_steps: Number of DDIM steps

        Returns:
            timesteps: List of timesteps in descending order
        """
        total_steps = self.diffusion.num_timesteps
        step_size = total_steps // num_steps
        timesteps = list(range(total_steps - 1, -1, -step_size))[:num_steps]
        return timesteps

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        t: int,
        noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict clean sample from noisy sample and predicted noise.

        x_0 = (x_t - sqrt(1 - alpha_t) * noise) / sqrt(alpha_t)

        Args:
            x_t: Noisy sample
            t: Timestep
            noise_pred: Predicted noise

        Returns:
            x_0: Predicted clean sample
        """
        # Get alpha values (handle both tensor and numpy array)
        alpha_t = self.diffusion.alphas_cumprod[t]
        if not isinstance(alpha_t, torch.Tensor):
            alpha_t = torch.tensor(alpha_t, device=self.device, dtype=torch.float32)

        sqrt_alpha = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

        x_0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        return x_0

    def _add_noise(
        self,
        x_0: torch.Tensor,
        t: int,
        eta: float = 1.0
    ) -> torch.Tensor:
        """
        Re-noise x_0 to timestep t for inpainting.

        For inpainting, we MUST add noise after modifying x_0.
        With eta=0 and zero noise, we'd just get scaled x_0, which is wrong.

        x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * eta * noise

        Args:
            x_0: Clean sample (after inpainting)
            t: Target timestep
            eta: Noise scale (default 1.0 for full noise, NOT 0)

        Returns:
            x_t: Noisy sample at timestep t
        """
        # Get alpha values (handle both tensor and numpy array)
        alpha_t = self.diffusion.alphas_cumprod[t]
        if not isinstance(alpha_t, torch.Tensor):
            alpha_t = torch.tensor(alpha_t, device=self.device, dtype=torch.float32)

        sqrt_alpha = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

        # For inpainting, we need fresh noise to properly re-noise
        # eta scales the noise (1.0 = standard, <1.0 = less noise, >1.0 = more)
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * eta * noise
        return x_t


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing InpaintingSampler V8...")
    print("=" * 70)

    print("\nInpaintingSampler class defined successfully")
    print("\nKey methods:")
    print("  - generate(start_pos, end_pos, length) -> single trajectory")
    print("  - generate_batch(endpoints, lengths) -> batch of trajectories")
    print("  - generate_single(target_x, target_y, length) -> convenience method")
    print("\nUsage:")
    print("  sampler = InpaintingSampler(model, diffusion, device='cuda')")
    print("  trajectory = sampler.generate_single(target_x=0.5, target_y=0.3, length=100)")
    print("=" * 70)
