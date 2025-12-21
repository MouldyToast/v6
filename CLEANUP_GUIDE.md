# Cleanup Guide - Safe Files to Delete

Quick reference for cleaning up the v6 project.

---

## ✅ KEEP - Essential Files

### Core Package (Never Delete)
```
diffusion_v7/
├── __init__.py
├── config_trajectory.py
├── models/
│   ├── __init__.py
│   ├── goal_conditioner.py
│   ├── trajectory_transformer.py
│   └── gaussian_diffusion.py
├── datasets/
│   ├── __init__.py
│   └── trajectory_dataset.py
├── trainers/
│   ├── __init__.py
│   └── trajectory_trainer.py
├── sampling/
│   ├── __init__.py
│   └── trajectory_sampler.py
└── evaluation/
    ├── __init__.py
    ├── metrics.py
    └── visualize.py
```

### Main Scripts (Never Delete)
- `preprocess_V6.py` - Data preprocessing
- `train_diffusion_v7.py` - Training CLI
- `generate_diffusion_v7.py` - Generation CLI

### Data & Models (Never Delete)
- `processed_data_v6/` - All .npy files
- `checkpoints_diffusion_v7/best.pth` - Best model
- `checkpoints_diffusion_v7/final.pth` - Final model

### Documentation (Keep)
- `DIFFUSION_V7_ARCHITECTURE.md` - Master document
- `DIFFUSION_RECOMMENDATION.md` - Design decisions
- `CLEANUP_GUIDE.md` - This file
- Any README.md files

---

## ⚠️ KEEP - Useful Tools

### Testing & Validation
- `test_diffusion_v7_architecture.py` - Validates model works

### Diagnostic Scripts (Keep During Development)
- `visualize_generated.py` - Trajectory visualization
- `compare_real_vs_generated.py` - Real vs generated comparison
- `diagnose_distance_error.py` - Distance accuracy diagnostics
- `check_norm_params.py` - Normalization parameter inspector

---

## 🗑️ DELETE - Safe to Remove

### Temporary Files
```bash
# Python cache
rm -rf __pycache__/
rm -rf diffusion_v7/__pycache__/
rm -rf diffusion_v7/*/__pycache__/
find . -name "*.pyc" -delete

# Temporary checkpoints (keep best.pth and final.pth)
rm checkpoints_diffusion_v7/checkpoint_epoch_*.pth
```

### Redundant Utilities
- `check_normalization.py` - Simple version, redundant with `check_norm_params.py`

### Temporary Outputs (After Analysis)
- `condition_distribution.png`
- `trajectories_by_angle.png`
- `individual_with_targets.png`
- `endpoint_scatter.png`
- `straightness_comparison.png`
- `distance_diagnostic.png`
- `real_vs_generated_examples.png`

### Old Results (If Re-generated)
```bash
# Old generation results (after you've analyzed them)
rm -rf results/diffusion_v7_old/
rm -rf results/test_runs/
```

---

## 🤔 REVIEW BEFORE DELETING

### If You're Fully Migrating to Diffusion (from V6 GAN)

**Can Archive/Delete:**
- `embedder_v6.py`
- `recovery_v6.py`
- `model_v6.py`
- `pooler_v6.py`
- `train_v6.py` (or similar V6 GAN training scripts)
- `checkpoints_v6/` (if you're not using V6 GAN anymore)

**KEEP:**
- `preprocess_V6.py` - Still needed! Diffusion uses the same preprocessing

### MotionDiffuse Files

**Only Keep:**
- Code already integrated into `diffusion_v7/models/gaussian_diffusion.py`

**Delete:**
- Any other standalone MotionDiffuse files not in your `diffusion_v7/` package
- MotionDiffuse checkpoints (if you downloaded any)
- MotionDiffuse datasets (if you downloaded any)

---

## 📊 Disk Space Recovery

### Expected Savings

**Cache files:** ~10-50 MB
```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

**Old checkpoints:** ~100-500 MB per checkpoint
```bash
# Keep only best.pth and final.pth
# Delete checkpoint_epoch_*.pth files
```

**Temporary diagnostic images:** ~5-10 MB
```bash
rm *.png  # Only if you've analyzed them
```

**Old results directories:** Varies
```bash
# Review and delete old result directories
```

---

## 🎯 Recommended Cleanup Workflow

### Step 1: Clean Python Cache
```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Step 2: Clean Old Checkpoints
```bash
cd checkpoints_diffusion_v7/
# Keep best.pth and final.pth
# Delete: checkpoint_epoch_*.pth (except maybe the last one)
ls -lh  # Review before deleting
# rm checkpoint_epoch_*.pth  # Uncomment to delete
```

### Step 3: Archive Diagnostic Images
```bash
mkdir -p archive/diagnostics/
mv *.png archive/diagnostics/  # Move instead of delete
```

### Step 4: Clean Old Results
```bash
# Review results/ directory
ls -lh results/
# Archive or delete old runs
```

### Step 5: Review V6 GAN Files (If Migrating)
```bash
# List V6 GAN related files
ls -lh embedder_v6.py recovery_v6.py model_v6.py pooler_v6.py
# Archive if you might need them later
mkdir -p archive/v6_gan/
# mv embedder_v6.py recovery_v6.py model_v6.py pooler_v6.py archive/v6_gan/
```

---

## ⚡ Quick Cleanup Commands

### Minimal Cleanup (Safe)
```bash
# Just Python cache
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Moderate Cleanup
```bash
# Python cache + temporary images
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
mkdir -p archive/
mv *.png archive/ 2>/dev/null || true
```

### Aggressive Cleanup (Review First!)
```bash
# Cache + images + old checkpoints
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
mkdir -p archive/
mv *.png archive/ 2>/dev/null || true
cd checkpoints_diffusion_v7/
ls checkpoint_epoch_*.pth | head -n -1 | xargs rm  # Keep last epoch checkpoint
```

---

## 🚫 NEVER Delete

**Critical Data:**
- `processed_data_v6/*.npy` - Regenerating requires raw data
- `checkpoints_diffusion_v7/best.pth` - Best trained model
- `trajectories/` - Original raw data (if you still have it)

**Core Code:**
- Anything in `diffusion_v7/` package
- `preprocess_V6.py`, `train_diffusion_v7.py`, `generate_diffusion_v7.py`

**Configuration:**
- `.git/` directory
- `.gitignore`
- Any environment configuration files

---

## 📝 Notes

### Before Deleting Anything
1. Commit current work to git
2. Make a backup if unsure
3. Test that everything still works after cleanup

### After Cleanup
```bash
# Verify essential files still exist
python -c "import diffusion_v7; print('Package OK')"
python train_diffusion_v7.py --help  # Should work
python generate_diffusion_v7.py --help  # Should work
```

### Disk Usage Check
```bash
# Before cleanup
du -sh .

# After cleanup
du -sh .

# Check specific directories
du -sh diffusion_v7/ processed_data_v6/ checkpoints_diffusion_v7/
```

---

Last updated: 2025-12-21
