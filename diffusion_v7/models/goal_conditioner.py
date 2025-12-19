"""
Goal Conditioner for Trajectory Diffusion

Embeds 3D goal conditioning vector [distance_norm, cos(angle), sin(angle)]
into the model's hidden dimension.

Supports Classifier-Free Guidance (CFG) for stronger conditioning control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalConditioner(nn.Module):
    """
    Embeds 3D goal condition to latent dimension for transformer conditioning.

    Architecture:
        3D input → 128D → 256D → latent_dim
        With SiLU activations and optional dropout

    Classifier-Free Guidance (CFG):
        During training, randomly replaces condition with learned null embedding
        to enable unconditional generation. During inference, interpolates between
        conditional and unconditional predictions for stronger conditioning.

    Args:
        condition_dim: Input condition dimension (default: 3 for distance + cos/sin angle)
        latent_dim: Output dimension matching transformer hidden dim
        use_cfg: Enable classifier-free guidance
        dropout: Dropout probability
    """

    def __init__(
        self,
        condition_dim: int = 3,
        latent_dim: int = 512,
        use_cfg: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.use_cfg = use_cfg

        # Embedding network: 3D → latent_dim
        self.embed = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(256, latent_dim),
        )

        # Classifier-free guidance: learnable null embedding
        # This represents "no condition" and is used during training when we drop the condition
        if use_cfg:
            self.null_embedding = nn.Parameter(torch.zeros(latent_dim))
            nn.init.normal_(self.null_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter('null_embedding', None)

    def forward(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with optional CFG masking.

        Args:
            condition: (batch, 3) - goal conditions [distance_norm, cos(angle), sin(angle)]
            mask: (batch,) - boolean mask where True = use condition, False = use null embedding
                  If None, uses all conditions (inference mode)

        Returns:
            embedded: (batch, latent_dim) - embedded condition vectors

        Example:
            # Training with CFG
            condition = torch.rand(64, 3)
            mask = torch.rand(64) > 0.1  # 10% dropout
            embedded = conditioner(condition, mask)

            # Inference (no mask)
            embedded = conditioner(condition)
        """
        batch_size = condition.shape[0]
        device = condition.device

        # Validate input
        assert condition.shape == (batch_size, self.condition_dim), \
            f"Expected condition shape ({batch_size}, {self.condition_dim}), got {condition.shape}"

        # Embed all conditions
        embedded = self.embed(condition)  # (batch, latent_dim)

        # Apply CFG masking if provided (training mode)
        if mask is not None and self.use_cfg:
            assert mask.shape == (batch_size,), \
                f"Expected mask shape ({batch_size},), got {mask.shape}"

            # Expand null embedding to batch
            null_embed = self.null_embedding.unsqueeze(0).expand(batch_size, -1)  # (batch, latent_dim)

            # Replace masked positions with null embedding
            # mask=True → use condition, mask=False → use null
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, 1)
            embedded = mask_expanded * embedded + (1 - mask_expanded) * null_embed

        return embedded

    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get null embedding for unconditional generation.

        Used during CFG sampling to get unconditional predictions.

        Args:
            batch_size: Number of samples
            device: Target device

        Returns:
            null_embed: (batch_size, latent_dim) - null embeddings
        """
        if not self.use_cfg:
            raise ValueError("Classifier-free guidance not enabled")

        return self.null_embedding.unsqueeze(0).expand(batch_size, -1).to(device)

    def sample_with_cfg(
        self,
        condition: torch.Tensor,
        model_forward_fn,
        guidance_scale: float = 2.0,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Sample with classifier-free guidance.

        Runs model twice (conditional + unconditional) and combines predictions:
            pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

        Args:
            condition: (batch, 3) - goal conditions
            model_forward_fn: Function that takes embedded condition and returns prediction
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
            **model_kwargs: Additional arguments for model forward

        Returns:
            prediction: Model output with CFG applied

        Example:
            def model_fn(cond_embed):
                return model(x_noisy, t, cond_embed, lengths)

            prediction = conditioner.sample_with_cfg(
                condition=goal_conditions,
                model_forward_fn=model_fn,
                guidance_scale=2.0
            )
        """
        if not self.use_cfg:
            # No CFG, just use conditional prediction
            cond_embed = self.forward(condition)
            return model_forward_fn(cond_embed, **model_kwargs)

        batch_size = condition.shape[0]
        device = condition.device

        # Conditional prediction
        cond_embed = self.forward(condition)
        cond_pred = model_forward_fn(cond_embed, **model_kwargs)

        # Unconditional prediction
        uncond_embed = self.get_null_embedding(batch_size, device)
        uncond_pred = model_forward_fn(uncond_embed, **model_kwargs)

        # Combine with guidance scale
        # guidance_scale = 1.0 → cond_pred
        # guidance_scale > 1.0 → extrapolate beyond cond_pred
        prediction = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

        return prediction


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == '__main__':
    print("Testing GoalConditioner...")
    print("=" * 70)

    # Test parameters
    batch_size = 16
    condition_dim = 3
    latent_dim = 128

    # Create conditioner
    print("\n=== Test 1: Basic Forward Pass ===")
    conditioner = GoalConditioner(
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        use_cfg=True,
        dropout=0.1
    )
    print(f"Created GoalConditioner")
    print(f"  Parameters: {sum(p.numel() for p in conditioner.parameters()):,}")

    # Test input
    condition = torch.randn(batch_size, condition_dim)
    condition[:, 0] = torch.tanh(condition[:, 0])  # distance_norm in [-1, 1]
    condition[:, 1:] = torch.tanh(condition[:, 1:])  # cos/sin in [-1, 1]

    print(f"\nInput condition shape: {condition.shape}")
    print(f"  Distance range: [{condition[:, 0].min():.3f}, {condition[:, 0].max():.3f}]")
    print(f"  Cos(angle) range: [{condition[:, 1].min():.3f}, {condition[:, 1].max():.3f}]")
    print(f"  Sin(angle) range: [{condition[:, 2].min():.3f}, {condition[:, 2].max():.3f}]")

    # Forward pass without mask (inference mode)
    embedded = conditioner(condition)
    print(f"\nOutput embedding shape: {embedded.shape}")
    print(f"  Range: [{embedded.min():.3f}, {embedded.max():.3f}]")
    assert embedded.shape == (batch_size, latent_dim)
    print("✓ Forward pass without mask passed")

    # Test 2: CFG masking
    print("\n=== Test 2: Classifier-Free Guidance Masking ===")
    cfg_dropout = 0.2
    mask = torch.rand(batch_size) > cfg_dropout  # True = keep condition
    n_dropped = (~mask).sum().item()
    print(f"CFG dropout: {cfg_dropout} → {n_dropped}/{batch_size} conditions dropped")

    embedded_masked = conditioner(condition, mask=mask)
    print(f"Output shape: {embedded_masked.shape}")
    assert embedded_masked.shape == (batch_size, latent_dim)

    # Check that masked positions use null embedding
    null_embed = conditioner.get_null_embedding(batch_size, condition.device)
    print(f"Null embedding shape: {null_embed.shape}")

    # Verify masked positions match null embedding
    for i in range(batch_size):
        if not mask[i]:
            diff = (embedded_masked[i] - null_embed[i]).abs().max().item()
            assert diff < 1e-5, f"Masked position {i} should use null embedding, got diff={diff}"

    print("✓ CFG masking passed")

    # Test 3: No CFG
    print("\n=== Test 3: Without CFG ===")
    conditioner_no_cfg = GoalConditioner(
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        use_cfg=False
    )
    embedded_no_cfg = conditioner_no_cfg(condition)
    print(f"Output shape: {embedded_no_cfg.shape}")
    assert embedded_no_cfg.shape == (batch_size, latent_dim)
    print("✓ No-CFG mode passed")

    # Test 4: Gradient flow
    print("\n=== Test 4: Gradient Flow ===")
    conditioner.train()
    embedded = conditioner(condition, mask=mask)
    loss = embedded.mean()
    loss.backward()

    # Check gradients
    for name, param in conditioner.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        else:
            print(f"  {name}: no gradient")

    print("✓ Gradient flow passed")

    # Test 5: CFG Sampling Mock
    print("\n=== Test 5: CFG Sampling (Mock) ===")
    conditioner.eval()

    # Mock model forward function
    def mock_model_fn(cond_embed, x_dummy):
        # Simulate model output: just return something dependent on condition
        return cond_embed.mean(dim=-1, keepdim=True) + x_dummy

    x_dummy = torch.randn(batch_size, 1)

    # Sample with CFG
    guidance_scales = [1.0, 1.5, 2.0, 3.0]
    for scale in guidance_scales:
        pred = conditioner.sample_with_cfg(
            condition=condition,
            model_forward_fn=lambda c: mock_model_fn(c, x_dummy),
            guidance_scale=scale
        )
        print(f"  Guidance scale={scale}: output shape={pred.shape}, "
              f"range=[{pred.min():.3f}, {pred.max():.3f}]")

    print("✓ CFG sampling passed")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
