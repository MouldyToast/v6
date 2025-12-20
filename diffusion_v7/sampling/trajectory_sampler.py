"""
Trajectory Generation with DDIM Sampling

Provides fast trajectory generation using DDIM (50 steps vs 1000 DDPM steps).
Supports classifier-free guidance for stronger goal conditioning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm


class TrajectoryGenerator:
    """
    Generate trajectories using trained diffusion model.

    Supports:
    - DDIM sampling (fast, 50 steps)
    - DDPM sampling (slow, 1000 steps)
    - Classifier-free guidance (CFG)
    - Batch generation
    - Variable sequence lengths

    Args:
        model: Trained TrajectoryTransformer
        goal_conditioner: Trained GoalConditioner
        diffusion: GaussianDiffusion instance
        device: torch device
    """

    def __init__(
        self,
        model,
        goal_conditioner,
        diffusion,
        device: torch.device
    ):
        self.model = model
        self.goal_conditioner = goal_conditioner
        self.diffusion = diffusion
        self.device = device

        # Set models to eval mode
        self.model.eval()
        self.goal_conditioner.eval()

    @torch.no_grad()
    def generate(
        self,
        conditions: torch.Tensor,
        lengths: torch.Tensor,
        num_samples: int = 1,
        method: str = 'ddim',
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        cfg_scale: float = 2.0,
        use_cfg: bool = True,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate trajectories from goal conditions.

        Args:
            conditions: (batch, 3) - goal conditions [distance_norm, cos(angle), sin(angle)]
            lengths: (batch,) - desired sequence lengths
            num_samples: Number of samples per condition (for diversity)
            method: 'ddim' or 'ddpm'
            ddim_steps: Number of DDIM steps (only for method='ddim')
            ddim_eta: DDIM eta parameter (0=deterministic, 1=stochastic)
            cfg_scale: Classifier-free guidance scale (1.0=no guidance, >1.0=stronger)
            use_cfg: Whether to use CFG
            show_progress: Show progress bar

        Returns:
            trajectories: (batch * num_samples, max_length, 8) - generated trajectories
        """
        batch_size = conditions.shape[0]
        max_length = lengths.max().item()

        # Expand for num_samples
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
            lengths = lengths.repeat_interleave(num_samples, dim=0)

        total_batch = batch_size * num_samples

        # Move to device
        conditions = conditions.to(self.device)
        lengths = lengths.to(self.device)

        # Shape for generation
        shape = (total_batch, max_length, 8)

        # Create model wrapper for diffusion
        if use_cfg and self.goal_conditioner.use_cfg:
            # CFG: Run model twice (conditional + unconditional)
            def model_fn(x, t):
                # Conditional prediction
                goal_embed_cond = self.goal_conditioner(conditions, mask=None)
                pred_cond = self.model(x, t, goal_embed_cond, lengths)

                # Unconditional prediction
                goal_embed_uncond = self.goal_conditioner.get_null_embedding(
                    total_batch, self.device
                )
                pred_uncond = self.model(x, t, goal_embed_uncond, lengths)

                # Combine with guidance scale
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                return pred
        else:
            # No CFG: Just conditional
            def model_fn(x, t):
                goal_embed = self.goal_conditioner(conditions, mask=None)
                return self.model(x, t, goal_embed, lengths)

        # Generate based on method
        if method == 'ddim':
            # DDIM sampling (fast)
            # Create timestep schedule for DDIM
            timesteps = self._make_ddim_timesteps(
                ddim_num_steps=ddim_steps,
                total_timesteps=self.diffusion.num_timesteps
            )

            trajectories = self._ddim_sample(
                model_fn=model_fn,
                shape=shape,
                timesteps=timesteps,
                eta=ddim_eta,
                show_progress=show_progress
            )
        elif method == 'ddpm':
            # DDPM sampling (slow but thorough)
            trajectories = self.diffusion.p_sample_loop(
                model=model_fn,
                shape=shape,
                clip_denoised=False,
                progress=show_progress
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ddim' or 'ddpm'")

        # Mask out padding based on lengths
        trajectories = self._mask_trajectories(trajectories, lengths)

        return trajectories

    def _make_ddim_timesteps(
        self,
        ddim_num_steps: int,
        total_timesteps: int
    ) -> np.ndarray:
        """
        Create timestep schedule for DDIM.

        Args:
            ddim_num_steps: Number of DDIM steps (e.g., 50)
            total_timesteps: Total diffusion timesteps (e.g., 1000)

        Returns:
            timesteps: Array of timesteps to use for DDIM
        """
        # Linear spacing
        c = total_timesteps // ddim_num_steps
        timesteps = np.asarray(list(range(0, total_timesteps, c)))

        # Reverse (we denoise from T to 0)
        timesteps = timesteps[::-1].copy()

        return timesteps

    def _ddim_sample(
        self,
        model_fn,
        shape: tuple,
        timesteps: np.ndarray,
        eta: float = 0.0,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        DDIM sampling loop.

        Args:
            model_fn: Model function (x, t) -> noise_prediction
            shape: Output shape (batch, seq_len, 8)
            timesteps: Timesteps to use for DDIM
            eta: DDIM eta (0=deterministic, 1=stochastic)
            show_progress: Show progress bar

        Returns:
            sample: Generated trajectories
        """
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(*shape, device=self.device)

        # DDIM denoising loop
        iterator = tqdm(timesteps, desc="DDIM sampling") if show_progress else timesteps

        for i, t in enumerate(iterator):
            # Current timestep
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise
            pred_noise = model_fn(x, t_batch)

            # Get next timestep
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else 0

            # DDIM update step
            x = self._ddim_step(
                x=x,
                pred_noise=pred_noise,
                t=t,
                t_next=t_next,
                eta=eta
            )

        return x

    def _ddim_step(
        self,
        x: torch.Tensor,
        pred_noise: torch.Tensor,
        t: int,
        t_next: int,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Single DDIM denoising step.

        DDIM update formula:
            x_0_pred = (x_t - sqrt(1 - alpha_t) * noise) / sqrt(alpha_t)
            direction = sqrt(1 - alpha_next) * noise
            x_next = sqrt(alpha_next) * x_0_pred + direction

        Args:
            x: Current noisy sample
            pred_noise: Predicted noise from model
            t: Current timestep
            t_next: Next timestep (smaller)
            eta: Stochasticity (0=deterministic)

        Returns:
            x_next: Less noisy sample
        """
        # Get alpha values
        alpha_t = self.diffusion.alphas_cumprod[t]
        alpha_next = self.diffusion.alphas_cumprod[t_next] if t_next > 0 else 1.0

        # Predict x_0 from x_t and noise
        sqrt_alpha_t = np.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = np.sqrt(1 - alpha_t)

        x_0_pred = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

        # Compute direction pointing to x_t
        sqrt_alpha_next = np.sqrt(alpha_next)
        sqrt_one_minus_alpha_next = np.sqrt(1 - alpha_next)

        # DDIM direction
        direction = sqrt_one_minus_alpha_next * pred_noise

        # Add stochasticity (eta > 0)
        if eta > 0:
            sigma = eta * np.sqrt((1 - alpha_next) / (1 - alpha_t)) * np.sqrt(1 - alpha_t / alpha_next)
            noise = torch.randn_like(x)
            direction = direction + sigma * noise

        # Compute x_next
        x_next = sqrt_alpha_next * x_0_pred + direction

        return x_next

    def _mask_trajectories(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Zero out padding positions based on sequence lengths.

        Args:
            trajectories: (batch, max_seq_len, 8)
            lengths: (batch,)

        Returns:
            masked: (batch, max_seq_len, 8) with padding zeroed
        """
        batch_size, max_len, _ = trajectories.shape

        # Create mask
        mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (batch, max_len, 1)

        # Apply mask
        masked = trajectories * mask

        return masked

    def generate_single(
        self,
        distance: float,
        angle: float,
        length: int,
        method: str = 'ddim',
        ddim_steps: int = 50,
        cfg_scale: float = 2.0,
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        Generate a single trajectory from distance and angle.

        Convenience function for generating one trajectory.

        Args:
            distance: Target distance (will be normalized)
            angle: Target angle in radians
            length: Desired trajectory length
            method: 'ddim' or 'ddpm'
            ddim_steps: DDIM steps
            cfg_scale: CFG guidance scale
            show_progress: Show progress bar

        Returns:
            trajectory: (length, 8) - single generated trajectory
        """
        # Create condition
        # Note: distance should be normalized to [-1, 1] using normalization_params
        # For now, assume it's already normalized or in valid range
        condition = torch.tensor([
            [distance, np.cos(angle), np.sin(angle)]
        ], dtype=torch.float32)

        lengths = torch.tensor([length], dtype=torch.long)

        # Generate
        trajectories = self.generate(
            conditions=condition,
            lengths=lengths,
            num_samples=1,
            method=method,
            ddim_steps=ddim_steps,
            cfg_scale=cfg_scale,
            show_progress=show_progress
        )

        # Return single trajectory
        return trajectories[0, :length]


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing TrajectoryGenerator...")
    print("=" * 70)

    # Note: This requires a trained model to test fully
    # Here we just test the class can be instantiated

    print("\nTrajectoryGenerator class defined successfully")
    print("To test, load a trained model and use:")
    print("  generator = TrajectoryGenerator(model, goal_conditioner, diffusion, device)")
    print("  trajectories = generator.generate(conditions, lengths)")
    print("=" * 70)
