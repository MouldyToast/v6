#!/usr/bin/env python3
"""
================================================================================
TimeGAN V6 Master Usage Script
================================================================================

Change RUN_MODE below to control what this script does.
All configurations are in the CONFIG section.

AVAILABLE MODES:
----------------
    "train"              - Full training (Stage 1 + Stage 2) with optimized loaders
    "train_stage1"       - Train only Stage 1 (Autoencoder)
    "train_stage2"       - Train only Stage 2 (WGAN-GP) from checkpoint
    "resume"             - Resume training from checkpoint
    "generate"           - Generate synthetic trajectories
    "evaluate"           - Evaluate trained model
    "visualize"          - Visualize real vs generated trajectories
    "quick_test"         - Quick smoke test with synthetic data
    "validate_data"      - Validate preprocessed data directory
    "inspect_model"      - Print model architecture and parameter counts
    "inspect_checkpoint" - Inspect a saved checkpoint
    "export_samples"     - Generate and export samples to file

QUICK START:
------------
1. Set RUN_MODE to your desired mode
2. Update CONFIG paths (data_dir, checkpoint paths, etc.)
3. Run: python run_v6.py

================================================================================
"""

import os
import sys
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# RUN MODE - Change this to control what the script does
# ==============================================================================

RUN_MODE = "quick_test"  # <-- CHANGE THIS (see options above)


# ==============================================================================
# CONFIGURATION - Adjust these settings as needed
# ==============================================================================

CONFIG = {
    # ─── Data Paths ───────────────────────────────────────────────────────────
    "data_dir": "processed_data_v4",          # Directory with preprocessed V4 data

    # ─── Checkpoint Paths ─────────────────────────────────────────────────────
    # WINDOWS USERS: Use forward slashes or raw strings for paths!
    #   Good: "D:/V6/checkpoints/v6"  or  r"D:\V6\checkpoints\v6"
    #   Bad:  "D:\V6\checkpoints\v6"  (backslash = escape character)
    # NOTE: Checkpoints must be .pt FILES, not directories!
    #   Good: "checkpoints/v6/final.pt"
    #   Bad:  "checkpoints/v6"
    "checkpoint_dir": "./checkpoints/v6",     # Where to save checkpoints
    "stage1_checkpoint": None,                # .pt file for Stage 2 training (e.g., "checkpoints/v6/stage1_final.pt")
    "resume_checkpoint": None,                # .pt file to resume from (e.g., "checkpoints/v6/stage2_iter5000.pt")
    "eval_checkpoint": None,                  # .pt file for eval/generate (e.g., "checkpoints/v6/final.pt")

    # ─── Device ───────────────────────────────────────────────────────────────
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ─── Training Parameters ──────────────────────────────────────────────────
    "batch_size": 64,
    "num_workers": 0,                         # Set to 4+ for faster loading on GPU

    # Stage 1 (Autoencoder)
    "stage1_iterations": 15000,
    "lr_autoencoder": 1e-3,
    "stage1_threshold": 0.05,                 # Convergence threshold

    # Stage 2 (WGAN-GP)
    "stage2_iterations": 30000,
    "lr_generator": 1e-4,
    "lr_discriminator": 1e-4,
    "n_critic": 5,                            # D updates per G update

    # ─── Generation Parameters ────────────────────────────────────────────────
    "n_samples": 100,                         # Number of samples to generate
    "seq_len": 100,                           # Sequence length for generation
    "output_file": "generated_samples.npy",   # Output file for exports

    # ─── Visualization Parameters ─────────────────────────────────────────────
    "viz_n_samples": 5,                       # Number of trajectories to visualize
    "viz_output_file": "trajectory_comparison.png",  # Output image file
    "viz_show_plot": True,                    # Show plot interactively (set False for headless)

    # ─── Quick Test Parameters ────────────────────────────────────────────────
    "quick_test_iterations": 50,              # Very short training for testing

    # ─── Logging ──────────────────────────────────────────────────────────────
    "log_interval": 100,
    "eval_interval": 1000,
    "save_interval": 5000,

    # ─── Model Architecture (usually leave as default) ────────────────────────
    "config_preset": "default",               # "default", "fast", or "large"
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_config():
    """Get TimeGANV6Config based on preset and CONFIG overrides."""
    from timegan_v6 import TimeGANV6Config, get_default_config, get_fast_config, get_large_config

    preset = CONFIG["config_preset"]
    if preset == "fast":
        config = get_fast_config()
    elif preset == "large":
        config = get_large_config()
    else:
        config = get_default_config()

    # Override with CONFIG values
    config.device = CONFIG["device"]
    config.batch_size = CONFIG["batch_size"]
    config.num_workers = CONFIG["num_workers"]
    config.stage1_iterations = CONFIG["stage1_iterations"]
    config.lr_autoencoder = CONFIG["lr_autoencoder"]
    config.stage1_threshold = CONFIG["stage1_threshold"]
    config.stage2_iterations = CONFIG["stage2_iterations"]
    config.lr_generator = CONFIG["lr_generator"]
    config.lr_discriminator = CONFIG["lr_discriminator"]
    config.n_critic = CONFIG["n_critic"]
    config.log_interval = CONFIG["log_interval"]
    config.eval_interval = CONFIG["eval_interval"]
    config.save_interval = CONFIG["save_interval"]
    config.checkpoint_dir = CONFIG["checkpoint_dir"]

    return config


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_config():
    """Print current configuration."""
    print("Configuration:")
    print("-" * 40)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()


def validate_checkpoint_path(path, config_key):
    """
    Validate that a checkpoint path is a valid .pt file.

    Returns (is_valid, error_message)
    """
    if path is None:
        return False, f"ERROR: {config_key} must be set in CONFIG"

    if not os.path.exists(path):
        return False, f"ERROR: File not found: {path}"

    if os.path.isdir(path):
        # Common mistake: pointing to directory instead of file
        pt_files = [f for f in os.listdir(path) if f.endswith('.pt')]
        if pt_files:
            suggestion = os.path.join(path, pt_files[0])
            return False, (f"ERROR: {path} is a directory, not a file!\n"
                          f"  Did you mean: {suggestion}\n"
                          f"  Available .pt files: {pt_files}")
        return False, f"ERROR: {path} is a directory, not a .pt file"

    if not path.endswith('.pt'):
        return False, f"WARNING: {path} doesn't have .pt extension (may still work)"

    return True, None


# ==============================================================================
# MODE: train - Full Training Pipeline
# ==============================================================================

def run_train():
    """Run full V6 training with optimized data loaders."""
    print_header("TimeGAN V6 - Full Training")
    print_config()

    from timegan_v6 import train_v6_optimized

    config = get_config()

    model, metrics = train_v6_optimized(
        data_dir=CONFIG["data_dir"],
        config=config,
        device=CONFIG["device"]
    )

    print("\n" + "=" * 70)
    print(" Training Complete!")
    print("=" * 70)
    print(f"\nStage 1 Results:")
    print(f"  Reconstruction Loss: {metrics['stage1'].get('loss_recon', 'N/A'):.4f}")
    print(f"  Converged: {metrics['stage1'].get('converged', 'N/A')}")
    print(f"\nStage 2 Results:")
    print(f"  Wasserstein Distance: {metrics['stage2'].get('wasserstein', 'N/A'):.4f}")
    print(f"\nCheckpoints saved to: {CONFIG['checkpoint_dir']}")

    return model, metrics


# ==============================================================================
# MODE: train_stage1 - Train Only Autoencoder
# ==============================================================================

def run_train_stage1():
    """Train only Stage 1 (Autoencoder)."""
    print_header("TimeGAN V6 - Stage 1 Training (Autoencoder)")
    print_config()

    from timegan_v6 import TimeGANV6, TimeGANV6Trainer
    from data_loader_v6 import create_stage1_loader

    config = get_config()

    # Create model and trainer
    model = TimeGANV6(config)
    trainer = TimeGANV6Trainer(model, config)

    # Create Stage 1 loader
    stage1_loader = create_stage1_loader(
        CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        augment=True,
        quality_filter=True
    )

    # Train
    metrics = trainer.train_stage1(stage1_loader)

    print(f"\nStage 1 Complete!")
    print(f"  Final Reconstruction Loss: {metrics['loss_recon']:.4f}")
    print(f"  Checkpoint: {CONFIG['checkpoint_dir']}/stage1_final.pt")

    return model, metrics


# ==============================================================================
# MODE: train_stage2 - Train Only WGAN-GP (from checkpoint)
# ==============================================================================

def run_train_stage2():
    """Train only Stage 2 (WGAN-GP) using pretrained autoencoder."""
    print_header("TimeGAN V6 - Stage 2 Training (WGAN-GP)")

    if CONFIG["stage1_checkpoint"] is None:
        print("ERROR: stage1_checkpoint must be set in CONFIG for Stage 2 training")
        print("Set CONFIG['stage1_checkpoint'] = 'path/to/stage1_final.pt'")
        return None, None

    print_config()

    from timegan_v6 import TimeGANV6, TimeGANV6Trainer
    from data_loader_v6 import create_stage2_loader

    config = get_config()

    # Create model and trainer
    model = TimeGANV6(config)
    trainer = TimeGANV6Trainer(model, config)

    # Load Stage 1 checkpoint
    trainer.load_checkpoint(CONFIG["stage1_checkpoint"])

    # Create Stage 2 loader
    stage2_loader = create_stage2_loader(
        CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        augment=False,
        quality_filter=True
    )

    # Train
    metrics = trainer.train_stage2(stage2_loader)

    print(f"\nStage 2 Complete!")
    print(f"  Final Wasserstein: {metrics['wasserstein']:.4f}")
    print(f"  Checkpoint: {CONFIG['checkpoint_dir']}/stage2_final.pt")

    return model, metrics


# ==============================================================================
# MODE: resume - Resume Training from Checkpoint
# ==============================================================================

def run_resume():
    """Resume training from a checkpoint."""
    print_header("TimeGAN V6 - Resume Training")

    if CONFIG["resume_checkpoint"] is None:
        print("ERROR: resume_checkpoint must be set in CONFIG")
        print("Set CONFIG['resume_checkpoint'] = 'path/to/checkpoint.pt'")
        return None, None

    print_config()

    from timegan_v6 import TimeGANV6, TimeGANV6Trainer
    from data_loader_v6 import create_stage1_loader, create_stage2_loader

    config = get_config()

    # Create model and trainer
    model = TimeGANV6(config)
    trainer = TimeGANV6Trainer(model, config)

    # Load checkpoint
    trainer.load_checkpoint(CONFIG["resume_checkpoint"])

    # Determine which stage to continue
    if trainer.stage1_completed and not trainer.stage2_completed:
        print("Resuming Stage 2 training...")
        stage2_loader = create_stage2_loader(
            CONFIG["data_dir"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"]
        )
        metrics = trainer.train_stage2(stage2_loader)
    elif not trainer.stage1_completed:
        print("Resuming Stage 1 training...")
        stage1_loader = create_stage1_loader(
            CONFIG["data_dir"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"]
        )
        metrics = trainer.train_stage1(stage1_loader)
    else:
        print("Training already complete!")
        metrics = trainer.metrics

    return model, metrics


# ==============================================================================
# MODE: generate - Generate Synthetic Trajectories
# ==============================================================================

def run_generate():
    """Generate synthetic trajectories from trained model."""
    print_header("TimeGAN V6 - Generate Trajectories")

    # Validate checkpoint path
    is_valid, error_msg = validate_checkpoint_path(CONFIG["eval_checkpoint"], "eval_checkpoint")
    if not is_valid:
        print(error_msg)
        print("\nSet CONFIG['eval_checkpoint'] = 'path/to/final.pt'")
        return None

    print_config()

    from timegan_v6 import TimeGANV6, TimeGANV6Config

    config = get_config()
    model = TimeGANV6(config)

    # Load checkpoint
    checkpoint = torch.load(CONFIG["eval_checkpoint"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG["device"])
    model.eval()

    # Generate random conditions
    n_samples = CONFIG["n_samples"]
    seq_len = CONFIG["seq_len"]
    conditions = torch.rand(n_samples, 1, device=CONFIG["device"])

    print(f"Generating {n_samples} trajectories with seq_len={seq_len}...")

    with torch.no_grad():
        x_fake = model.generate(n_samples, conditions, seq_len=seq_len)

    print(f"\nGenerated shape: {x_fake.shape}")
    print(f"Value range: [{x_fake.min().item():.4f}, {x_fake.max().item():.4f}]")

    # Show sample statistics
    print(f"\nSample statistics:")
    print(f"  Mean: {x_fake.mean().item():.4f}")
    print(f"  Std:  {x_fake.std().item():.4f}")

    return x_fake.cpu().numpy(), conditions.cpu().numpy()


# ==============================================================================
# MODE: evaluate - Evaluate Trained Model
# ==============================================================================

def run_evaluate():
    """Evaluate trained model on test data."""
    print_header("TimeGAN V6 - Evaluation")

    if CONFIG["eval_checkpoint"] is None:
        print("ERROR: eval_checkpoint must be set in CONFIG")
        print("Set CONFIG['eval_checkpoint'] = 'path/to/final.pt'")
        return None

    print_config()

    from timegan_v6 import TimeGANV6, TimeGANV6Evaluator, print_evaluation_report
    from data_loader_v6 import load_v6_data

    config = get_config()
    model = TimeGANV6(config)

    # Load checkpoint
    checkpoint = torch.load(CONFIG["eval_checkpoint"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG["device"])

    # Load test data
    print("Loading test data...")
    data = load_v6_data(CONFIG["data_dir"])

    X_test = torch.FloatTensor(data['X_test'])
    C_test = torch.FloatTensor(data['C_test']).unsqueeze(-1)
    L_test = torch.LongTensor(data['L_test'])

    print(f"Test data: {X_test.shape[0]} samples")

    # Evaluate
    evaluator = TimeGANV6Evaluator(model, device=CONFIG["device"])
    metrics = evaluator.evaluate(
        X_test, C_test, L_test,
        n_generated=min(500, X_test.shape[0])
    )

    # Print report
    print_evaluation_report(metrics)

    return metrics


# ==============================================================================
# MODE: quick_test - Quick Smoke Test
# ==============================================================================

def run_quick_test():
    """Quick smoke test with synthetic data."""
    print_header("TimeGAN V6 - Quick Smoke Test")

    import shutil

    # Create temporary test data
    test_dir = '/tmp/v6_quick_test'
    os.makedirs(test_dir, exist_ok=True)

    print("Creating synthetic test data...")
    np.random.seed(42)
    n_samples = 100
    seq_len = 50
    feature_dim = 8

    X = np.random.randn(n_samples, seq_len, feature_dim).astype(np.float32) * 0.5
    X = np.clip(X, -1, 1)
    C = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    L = np.random.randint(20, seq_len, n_samples).astype(np.int32)

    # Save splits
    for split, (start, end) in [('train', (0, 70)), ('val', (70, 85)), ('test', (85, 100))]:
        np.save(os.path.join(test_dir, f'X_{split}.npy'), X[start:end])
        np.save(os.path.join(test_dir, f'C_{split}.npy'), C[start:end])
        np.save(os.path.join(test_dir, f'L_{split}.npy'), L[start:end])

    print(f"  Created {n_samples} synthetic samples")

    # Quick training
    from timegan_v6 import TimeGANV6, TimeGANV6Config, TimeGANV6Trainer, get_fast_config
    from data_loader_v6 import create_stage_loaders

    config = get_fast_config()
    config.device = CONFIG["device"]
    config.stage1_iterations = CONFIG["quick_test_iterations"]
    config.stage2_iterations = CONFIG["quick_test_iterations"]
    config.log_interval = 10
    config.save_interval = 10000  # Don't save
    config.checkpoint_dir = os.path.join(test_dir, 'checkpoints')

    model = TimeGANV6(config)
    trainer = TimeGANV6Trainer(model, config)

    # Create loaders
    stage1_loader, stage2_loader = create_stage_loaders(test_dir, batch_size=16)

    print("\n--- Stage 1 (Autoencoder) ---")
    stage1_metrics = trainer.train_stage1(stage1_loader)

    print("\n--- Stage 2 (WGAN-GP) ---")
    stage2_metrics = trainer.train_stage2(stage2_loader)

    # Generate
    print("\n--- Generation ---")
    conditions = torch.rand(10, 1, device=CONFIG["device"])
    with torch.no_grad():
        x_fake = model.generate(10, conditions, seq_len=30)
    print(f"Generated: {x_fake.shape}")
    print(f"Range: [{x_fake.min().item():.3f}, {x_fake.max().item():.3f}]")

    # Cleanup
    shutil.rmtree(test_dir)
    print("\nCleaned up test data.")

    print("\n" + "=" * 70)
    print(" QUICK TEST PASSED!")
    print("=" * 70)

    return True


# ==============================================================================
# MODE: validate_data - Validate Data Directory
# ==============================================================================

def run_validate_data():
    """Validate preprocessed data directory."""
    print_header("TimeGAN V6 - Validate Data")
    print_config()

    from data_loader_v6 import validate_v6_data, load_v6_data

    try:
        validate_v6_data(CONFIG["data_dir"])

        # Load and show statistics
        data = load_v6_data(CONFIG["data_dir"])

        print("\nData Statistics:")
        print("-" * 40)

        for split in ['train', 'val', 'test']:
            X = data[f'X_{split}']
            C = data[f'C_{split}']
            L = data[f'L_{split}']

            print(f"\n{split.upper()}:")
            print(f"  Samples: {len(X)}")
            print(f"  Shape: {X.shape}")
            print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
            print(f"  C range: [{C.min():.3f}, {C.max():.3f}]")
            print(f"  L range: [{L.min()}, {L.max()}] (mean: {L.mean():.1f})")

        print("\n" + "=" * 70)
        print(" DATA VALIDATION PASSED!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        return False


# ==============================================================================
# MODE: inspect_model - Inspect Model Architecture
# ==============================================================================

def run_inspect_model():
    """Print model architecture and parameter counts."""
    print_header("TimeGAN V6 - Model Inspection")
    print_config()

    from timegan_v6 import TimeGANV6

    config = get_config()
    model = TimeGANV6(config)

    # Print summary
    print(model.summary())

    # Detailed component info
    print("\n" + "-" * 50)
    print("Component Details:")
    print("-" * 50)

    components = [
        ('Encoder', model.encoder),
        ('Decoder', model.decoder),
        ('Pooler', model.pooler),
        ('Expander', model.expander),
        ('Generator', model.generator),
        ('Discriminator', model.discriminator),
    ]

    for name, component in components:
        print(f"\n{name}:")
        print(f"  {component.__class__.__name__}")
        n_params = sum(p.numel() for p in component.parameters())
        n_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,} (trainable: {n_trainable:,})")

    return model


# ==============================================================================
# MODE: inspect_checkpoint - Inspect Saved Checkpoint
# ==============================================================================

def run_inspect_checkpoint():
    """Inspect a saved checkpoint."""
    print_header("TimeGAN V6 - Checkpoint Inspection")

    checkpoint_path = CONFIG.get("eval_checkpoint") or CONFIG.get("resume_checkpoint")

    if checkpoint_path is None:
        print("ERROR: Set eval_checkpoint or resume_checkpoint in CONFIG")
        return None

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None

    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 50)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\nKeys in checkpoint:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  {key}: {len(checkpoint[key])} tensors")
        elif key == 'config':
            print(f"  {key}: TimeGANV6Config")
        elif isinstance(checkpoint[key], dict):
            print(f"  {key}: {checkpoint[key]}")
        else:
            print(f"  {key}: {checkpoint[key]}")

    if 'metrics' in checkpoint:
        print(f"\nTraining Metrics:")
        metrics = checkpoint['metrics']
        if 'stage1' in metrics:
            print(f"  Stage 1: {metrics['stage1']}")
        if 'stage2' in metrics:
            print(f"  Stage 2: {metrics['stage2']}")

    if 'config' in checkpoint:
        print(f"\nConfiguration (from checkpoint):")
        config = checkpoint['config']
        for section, values in config.items():
            print(f"  {section}: {values}")

    return checkpoint


# ==============================================================================
# MODE: export_samples - Generate and Export Samples
# ==============================================================================

def run_export_samples():
    """Generate samples and export to file."""
    print_header("TimeGAN V6 - Export Samples")

    if CONFIG["eval_checkpoint"] is None:
        print("ERROR: eval_checkpoint must be set in CONFIG")
        return None

    print_config()

    # Generate
    samples, conditions = run_generate()

    if samples is None:
        return None

    # Export
    output_file = CONFIG["output_file"]
    np.savez(
        output_file,
        samples=samples,
        conditions=conditions,
        seq_len=CONFIG["seq_len"],
        n_samples=CONFIG["n_samples"]
    )

    print(f"\nExported to: {output_file}")
    print(f"  samples shape: {samples.shape}")
    print(f"  conditions shape: {conditions.shape}")

    return output_file


# ==============================================================================
# MODE: visualize - Visualize Real vs Generated Trajectories
# ==============================================================================

def run_visualize():
    """Visualize real vs generated trajectories side by side."""
    print_header("TimeGAN V6 - Visualization: Real vs Generated")

    # Validate checkpoint
    is_valid, error_msg = validate_checkpoint_path(CONFIG["eval_checkpoint"], "eval_checkpoint")
    if not is_valid:
        print(error_msg)
        print("\nSet CONFIG['eval_checkpoint'] = 'path/to/final.pt'")
        return None

    print_config()

    try:
        import matplotlib
        if not CONFIG["viz_show_plot"]:
            matplotlib.use('Agg')  # Headless mode
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return None

    from timegan_v6 import TimeGANV6
    from data_loader_v6 import load_v6_data

    config = get_config()
    model = TimeGANV6(config)

    # Load checkpoint
    print("Loading model...")
    checkpoint = torch.load(CONFIG["eval_checkpoint"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG["device"])
    model.eval()

    # Load real data
    print("Loading real data...")
    data = load_v6_data(CONFIG["data_dir"])
    X_real = data['X_test']
    C_real = data['C_test']
    L_real = data['L_test']

    n_viz = min(CONFIG["viz_n_samples"], len(X_real))

    # Select random samples
    indices = np.random.choice(len(X_real), n_viz, replace=False)

    # Generate fake samples with matching conditions
    conditions = torch.FloatTensor(C_real[indices]).unsqueeze(-1).to(CONFIG["device"])

    print(f"Generating {n_viz} synthetic trajectories...")
    with torch.no_grad():
        X_fake = model.generate(n_viz, conditions, seq_len=X_real.shape[1])
    X_fake = X_fake.cpu().numpy()

    # Feature names for V4 format
    feature_names = ['dx', 'dy', 'speed', 'accel', 'sin_h', 'cos_h', 'ang_vel', 'dt']

    # Create visualization
    fig, axes = plt.subplots(n_viz, 4, figsize=(16, 4 * n_viz))
    if n_viz == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        length = L_real[idx]
        real = X_real[idx, :length, :]
        fake = X_fake[i, :length, :]
        condition = C_real[idx]

        # Plot 1: XY trajectory (reconstructed from dx, dy)
        ax = axes[i, 0]
        real_x = np.cumsum(real[:, 0])
        real_y = np.cumsum(real[:, 1])
        fake_x = np.cumsum(fake[:, 0])
        fake_y = np.cumsum(fake[:, 1])
        ax.plot(real_x, real_y, 'b-', label='Real', linewidth=2, alpha=0.8)
        ax.plot(fake_x, fake_y, 'r--', label='Generated', linewidth=2, alpha=0.8)
        ax.scatter([real_x[0]], [real_y[0]], c='green', s=100, marker='o', zorder=5, label='Start')
        ax.scatter([real_x[-1]], [real_y[-1]], c='blue', s=100, marker='x', zorder=5)
        ax.scatter([fake_x[-1]], [fake_y[-1]], c='red', s=100, marker='x', zorder=5)
        ax.set_title(f'Sample {i+1} - XY Trajectory (cond={condition:.2f})')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.legend(loc='best')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        # Plot 2: Speed over time
        ax = axes[i, 1]
        time = np.arange(length)
        ax.plot(time, real[:, 2], 'b-', label='Real', linewidth=2, alpha=0.8)
        ax.plot(time, fake[:, 2], 'r--', label='Generated', linewidth=2, alpha=0.8)
        ax.set_title('Speed over Time')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Speed (normalized)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Plot 3: Feature distributions
        ax = axes[i, 2]
        features_to_compare = [0, 1, 2, 3]  # dx, dy, speed, accel
        positions = np.arange(len(features_to_compare))
        width = 0.35
        real_means = [real[:, f].mean() for f in features_to_compare]
        fake_means = [fake[:, f].mean() for f in features_to_compare]
        real_stds = [real[:, f].std() for f in features_to_compare]
        fake_stds = [fake[:, f].std() for f in features_to_compare]
        ax.bar(positions - width/2, real_means, width, yerr=real_stds, label='Real', color='blue', alpha=0.7, capsize=3)
        ax.bar(positions + width/2, fake_means, width, yerr=fake_stds, label='Generated', color='red', alpha=0.7, capsize=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([feature_names[f] for f in features_to_compare])
        ax.set_title('Feature Comparison (mean ± std)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Heading (sin_h, cos_h as angle)
        ax = axes[i, 3]
        real_angle = np.arctan2(real[:, 4], real[:, 5])  # atan2(sin, cos)
        fake_angle = np.arctan2(fake[:, 4], fake[:, 5])
        ax.plot(time, np.degrees(real_angle), 'b-', label='Real', linewidth=2, alpha=0.8)
        ax.plot(time, np.degrees(fake_angle), 'r--', label='Generated', linewidth=2, alpha=0.8)
        ax.set_title('Heading Angle over Time')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Angle (degrees)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = CONFIG["viz_output_file"]
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")

    # Show if requested
    if CONFIG["viz_show_plot"]:
        print("Displaying plot (close window to continue)...")
        plt.show()
    else:
        plt.close()

    print("\n" + "=" * 70)
    print(" VISUALIZATION COMPLETE!")
    print("=" * 70)

    return output_file


# ==============================================================================
# MAIN DISPATCHER
# ==============================================================================

MODES = {
    "train": run_train,
    "train_stage1": run_train_stage1,
    "train_stage2": run_train_stage2,
    "resume": run_resume,
    "generate": run_generate,
    "evaluate": run_evaluate,
    "visualize": run_visualize,
    "quick_test": run_quick_test,
    "validate_data": run_validate_data,
    "inspect_model": run_inspect_model,
    "inspect_checkpoint": run_inspect_checkpoint,
    "export_samples": run_export_samples,
}


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print(f" TimeGAN V6 - Mode: {RUN_MODE}")
    print("=" * 70)

    if RUN_MODE not in MODES:
        print(f"\nERROR: Unknown mode '{RUN_MODE}'")
        print(f"\nAvailable modes:")
        for mode in MODES:
            print(f"  - {mode}")
        sys.exit(1)

    # Run selected mode
    start_time = time.time()
    result = MODES[RUN_MODE]()
    elapsed = time.time() - start_time

    print(f"\nElapsed time: {elapsed:.1f}s")

    return result


if __name__ == "__main__":
    main()
