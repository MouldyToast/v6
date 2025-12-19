#!/usr/bin/env python3
"""
Architecture Test Script

Tests that all components can be imported and initialized without errors.
This verifies the code is syntactically correct before attempting training.
"""

import sys
import torch

print("=" * 70)
print("DIFFUSION V7 ARCHITECTURE TEST")
print("=" * 70)

# Test 1: Config
print("\n[Test 1] Importing configuration...")
try:
    from diffusion_v7.config_trajectory import (
        TrajectoryDiffusionConfig,
        get_smoke_test_config,
        get_medium_config,
        get_full_config
    )
    print("✓ Configuration imports successful")

    config = get_smoke_test_config()
    config.device = 'cpu'
    print(f"✓ Smoke test config created: {config.latent_dim}D, {config.num_layers} layers")
except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    sys.exit(1)

# Test 2: GoalConditioner
print("\n[Test 2] Testing GoalConditioner...")
try:
    from diffusion_v7.models import GoalConditioner

    conditioner = GoalConditioner(
        condition_dim=3,
        latent_dim=128,
        use_cfg=True
    )
    print(f"✓ GoalConditioner created: {sum(p.numel() for p in conditioner.parameters()):,} params")

    # Test forward pass
    condition = torch.randn(16, 3)
    goal_embed = conditioner(condition)
    print(f"✓ Forward pass: {condition.shape} → {goal_embed.shape}")

    # Test CFG masking
    mask = torch.rand(16) > 0.1
    goal_embed_masked = conditioner(condition, mask=mask)
    print(f"✓ CFG masking: {mask.sum().item()}/{16} conditions kept")

except Exception as e:
    print(f"✗ GoalConditioner test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: TrajectoryTransformer
print("\n[Test 3] Testing TrajectoryTransformer...")
try:
    from diffusion_v7.models import TrajectoryTransformer

    model = TrajectoryTransformer(
        input_dim=8,
        latent_dim=128,
        num_layers=4,
        num_heads=4,
        ff_size=256,
        max_seq_len=200,
        goal_latent_dim=128
    )
    print(f"✓ TrajectoryTransformer created: {sum(p.numel() for p in model.parameters()):,} params")

    # Test forward pass
    batch_size = 4
    seq_len = 50
    x = torch.randn(batch_size, seq_len, 8)
    t = torch.randint(0, 1000, (batch_size,))
    goal_embed = torch.randn(batch_size, 128)
    lengths = torch.tensor([50, 30, 45, 20])

    output = model(x, t, goal_embed, lengths)
    print(f"✓ Forward pass: {x.shape} → {output.shape}")

    # Test gradient flow
    loss = output.mean()
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = len(list(model.parameters()))
    print(f"✓ Gradients: {has_grad}/{total_params} parameters have gradients")

except Exception as e:
    print(f"✗ TrajectoryTransformer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: GaussianDiffusion
print("\n[Test 4] Testing GaussianDiffusion...")
try:
    from diffusion_v7.models.gaussian_diffusion import (
        GaussianDiffusion,
        get_named_beta_schedule,
        ModelMeanType,
        ModelVarType,
        LossType
    )

    betas = get_named_beta_schedule('linear', 1000)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE
    )
    print(f"✓ GaussianDiffusion created: {diffusion.num_timesteps} timesteps")

except Exception as e:
    print(f"✗ GaussianDiffusion test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: TrajectoryDataset
print("\n[Test 5] Testing TrajectoryDataset (without actual data)...")
try:
    from diffusion_v7.datasets import TrajectoryDataset

    print("✓ TrajectoryDataset class imported")
    print("  (Actual data loading test requires processed_data_v6/)")

except Exception as e:
    print(f"✗ TrajectoryDataset test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Trainer
print("\n[Test 6] Testing TrajectoryDiffusionTrainer...")
try:
    from diffusion_v7.trainers import TrajectoryDiffusionTrainer

    config.device = 'cpu'
    trainer = TrajectoryDiffusionTrainer(config)
    print(f"✓ Trainer created successfully")

    # Count total parameters
    total_params = (
        sum(p.numel() for p in trainer.model.parameters()) +
        sum(p.numel() for p in trainer.goal_conditioner.parameters())
    )
    print(f"✓ Total model parameters: {total_params:,}")

except Exception as e:
    print(f"✗ Trainer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Integration Test
print("\n[Test 7] Integration test (synthetic data)...")
try:
    # Create synthetic batch
    batch = {
        'motion': torch.randn(16, 50, 8),
        'condition': torch.randn(16, 3),
        'length': torch.randint(30, 50, (16,))
    }

    # Single training step
    metrics = trainer.train_step(batch)
    print(f"✓ Training step completed")
    print(f"  Loss: {metrics['loss']:.6f}")
    print(f"  MSE: {metrics['loss_mse']:.6f}")

except Exception as e:
    print(f"✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 70)
print("ALL TESTS PASSED ✅")
print("=" * 70)
print("\nArchitecture is ready for training!")
print("\nNext steps:")
print("  1. Prepare data: Ensure processed_data_v6/ exists")
print("  2. Run smoke test: python train_diffusion_v7.py --mode smoke_test --data_dir processed_data_v6")
print("  3. If smoke test passes, run full training")
print("=" * 70 + "\n")
