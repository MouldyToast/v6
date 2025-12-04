"""
Evaluation Suite for TimeGAN V6

Comprehensive evaluation metrics for generated trajectories:

1. Discriminative Score: Train classifier to distinguish real vs fake
   - Lower is better (closer to 0.5 = random)

2. Predictive Score: Train predictor on real, test on fake
   - Lower is better (fake is predictable like real)

3. Distribution Metrics:
   - MMD (Maximum Mean Discrepancy)
   - Correlation analysis
   - Feature statistics comparison

4. Latent Space Analysis:
   - t-SNE visualization
   - Latent distribution statistics
   - Condition fidelity

Usage:
    from timegan_v6.evaluation_v6 import TimeGANV6Evaluator

    evaluator = TimeGANV6Evaluator(model)
    metrics = evaluator.evaluate(real_data, conditions, n_samples=1000)
    print(metrics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List
import numpy as np

from .model_v6 import TimeGANV6


# =============================================================================
# Helper Networks for Evaluation
# =============================================================================

class SequenceClassifier(nn.Module):
    """
    LSTM classifier for discriminative score.
    Classifies sequences as real (1) or fake (0).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) optional

        Returns:
            prob: (batch, 1) probability of being real
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # Use last layer hidden state
        h_last = h_n[-1]  # (batch, hidden_dim)
        return self.classifier(h_last)


class SequencePredictor(nn.Module):
    """
    LSTM predictor for predictive score.
    Predicts next timestep from current context.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        self.predictor = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            pred: (batch, seq_len, input_dim) predictions for next timestep
        """
        lstm_out, _ = self.lstm(x)
        return self.predictor(lstm_out)


# =============================================================================
# Main Evaluator Class
# =============================================================================

class TimeGANV6Evaluator:
    """
    Comprehensive evaluation for TimeGAN V6 generated trajectories.
    """

    def __init__(self, model: TimeGANV6, device: torch.device = None):
        """
        Args:
            model: Trained TimeGANV6 model
            device: Device for evaluation
        """
        self.model = model
        self.device = device or model.get_device()
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, real_data: torch.Tensor, conditions: torch.Tensor,
                 lengths: torch.Tensor = None, n_generated: int = None,
                 seq_len: int = None) -> Dict[str, float]:
        """
        Run full evaluation suite.

        Args:
            real_data: (n_samples, seq_len, feature_dim) - real trajectories
            conditions: (n_samples, condition_dim) - conditions for generation
            lengths: (n_samples,) - sequence lengths (optional)
            n_generated: Number of fake samples to generate (defaults to len(real))
            seq_len: Sequence length for generation (defaults to real_data.size(1))

        Returns:
            Dict with all evaluation metrics
        """
        if n_generated is None:
            n_generated = real_data.size(0)
        if seq_len is None:
            seq_len = real_data.size(1)
        if lengths is None:
            lengths = torch.full((real_data.size(0),), seq_len)

        real_data = real_data.to(self.device)
        conditions = conditions.to(self.device)
        lengths = lengths.to(self.device)

        # Generate fake data (no grad needed for generation)
        print("Generating synthetic trajectories...")
        with torch.no_grad():
            fake_data = self._generate_fake(n_generated, conditions[:n_generated], seq_len)

        metrics = {}

        # Distribution metrics
        print("Computing distribution metrics...")
        dist_metrics = self.compute_distribution_metrics(real_data, fake_data, lengths)
        metrics.update(dist_metrics)

        # Discriminative score
        print("Computing discriminative score...")
        disc_score = self.compute_discriminative_score(real_data, fake_data, lengths)
        metrics['discriminative_score'] = disc_score

        # Predictive score
        print("Computing predictive score...")
        pred_score = self.compute_predictive_score(real_data, fake_data, lengths)
        metrics['predictive_score'] = pred_score

        # Latent space metrics
        print("Computing latent space metrics...")
        latent_metrics = self.compute_latent_metrics(real_data, fake_data, conditions, lengths)
        metrics.update(latent_metrics)

        return metrics

    def _generate_fake(self, n_samples: int, conditions: torch.Tensor,
                       seq_len: int) -> torch.Tensor:
        """Generate fake samples."""
        # Generate in batches to avoid memory issues
        batch_size = min(64, n_samples)
        fake_batches = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_conditions = conditions[i:batch_end]
            batch_fake = self.model.generate(
                batch_end - i, batch_conditions, seq_len=seq_len
            )
            fake_batches.append(batch_fake)

        return torch.cat(fake_batches, dim=0)

    # =========================================================================
    # Distribution Metrics
    # =========================================================================

    def compute_distribution_metrics(self, real: torch.Tensor, fake: torch.Tensor,
                                     lengths: torch.Tensor) -> Dict[str, float]:
        """
        Compute distribution comparison metrics.

        Args:
            real: (n, seq_len, feature_dim) - real data
            fake: (n, seq_len, feature_dim) - fake data
            lengths: (n,) - sequence lengths

        Returns:
            Dict with MMD, mean/std differences
        """
        # Flatten to (n, seq_len * feature_dim) for MMD
        real_flat = real.reshape(real.size(0), -1)
        fake_flat = fake.reshape(fake.size(0), -1)

        # MMD with Gaussian kernel
        mmd = self._compute_mmd(real_flat, fake_flat)

        # Feature-wise statistics
        real_mean = real.mean(dim=(0, 1))
        fake_mean = fake.mean(dim=(0, 1))
        mean_diff = (real_mean - fake_mean).abs().mean().item()

        real_std = real.std(dim=(0, 1))
        fake_std = fake.std(dim=(0, 1))
        std_diff = (real_std - fake_std).abs().mean().item()

        # Temporal statistics
        real_diff = (real[:, 1:] - real[:, :-1]).abs().mean().item()
        fake_diff = (fake[:, 1:] - fake[:, :-1]).abs().mean().item()
        temporal_diff = abs(real_diff - fake_diff)

        return {
            'mmd': mmd,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'temporal_diff': temporal_diff
        }

    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor,
                     kernel: str = 'rbf') -> float:
        """
        Compute Maximum Mean Discrepancy.

        Args:
            x: (n, d) - samples from distribution P
            y: (m, d) - samples from distribution Q
            kernel: 'rbf' or 'linear'

        Returns:
            MMD estimate
        """
        n, m = x.size(0), y.size(0)

        if kernel == 'rbf':
            # Use median heuristic for bandwidth
            with torch.no_grad():
                xy = torch.cat([x, y], dim=0)
                dists = torch.cdist(xy, xy)
                median_dist = dists.median()
                gamma = 1.0 / (2 * median_dist ** 2 + 1e-8)

            # Compute kernel matrices
            xx = torch.exp(-gamma * torch.cdist(x, x) ** 2)
            yy = torch.exp(-gamma * torch.cdist(y, y) ** 2)
            xy = torch.exp(-gamma * torch.cdist(x, y) ** 2)
        else:
            # Linear kernel
            xx = x @ x.T
            yy = y @ y.T
            xy = x @ y.T

        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        mmd_sq = xx.sum() / (n * n) + yy.sum() / (m * m) - 2 * xy.sum() / (n * m)

        return max(0, mmd_sq.item()) ** 0.5

    # =========================================================================
    # Discriminative Score
    # =========================================================================

    def compute_discriminative_score(self, real: torch.Tensor, fake: torch.Tensor,
                                     lengths: torch.Tensor, n_epochs: int = 20) -> float:
        """
        Train classifier to distinguish real vs fake.
        Lower score (closer to 0.5) is better.

        Args:
            real: (n, seq_len, feature_dim) - real data
            fake: (n, seq_len, feature_dim) - fake data
            lengths: (n,) - sequence lengths
            n_epochs: Training epochs

        Returns:
            Classification accuracy (0.5 = indistinguishable)
        """
        # Detach and clone to ensure clean gradient state for new classifier
        real = real.detach().clone()
        fake = fake.detach().clone()
        lengths = lengths.detach().clone()

        feature_dim = real.size(2)
        n_samples = real.size(0)

        # Create labels
        real_labels = torch.ones(n_samples, 1, device=self.device)
        fake_labels = torch.zeros(n_samples, 1, device=self.device)

        # Combine data
        all_data = torch.cat([real, fake], dim=0)
        all_labels = torch.cat([real_labels, fake_labels], dim=0)
        all_lengths = torch.cat([lengths, lengths], dim=0)

        # Shuffle
        perm = torch.randperm(all_data.size(0))
        all_data = all_data[perm]
        all_labels = all_labels[perm]
        all_lengths = all_lengths[perm]

        # Split train/test
        split = int(0.8 * all_data.size(0))
        train_data, test_data = all_data[:split], all_data[split:]
        train_labels, test_labels = all_labels[:split], all_labels[split:]
        train_lengths, test_lengths = all_lengths[:split], all_lengths[split:]

        # Create classifier
        classifier = SequenceClassifier(feature_dim).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # Train
        classifier.train()
        batch_size = 32

        for epoch in range(n_epochs):
            perm = torch.randperm(train_data.size(0))
            for i in range(0, train_data.size(0), batch_size):
                idx = perm[i:i + batch_size]
                batch_data = train_data[idx]
                batch_labels = train_labels[idx]
                batch_lengths = train_lengths[idx]

                optimizer.zero_grad()
                pred = classifier(batch_data, batch_lengths)
                loss = criterion(pred, batch_labels)
                loss.backward()
                optimizer.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            test_pred = classifier(test_data, test_lengths)
            test_pred_binary = (test_pred > 0.5).float()
            accuracy = (test_pred_binary == test_labels).float().mean().item()

        return accuracy

    # =========================================================================
    # Predictive Score
    # =========================================================================

    def compute_predictive_score(self, real: torch.Tensor, fake: torch.Tensor,
                                 lengths: torch.Tensor, n_epochs: int = 20) -> float:
        """
        Train predictor on real data, evaluate on fake.
        Lower score means fake is as predictable as real.

        Args:
            real: (n, seq_len, feature_dim) - real data
            fake: (n, seq_len, feature_dim) - fake data
            lengths: (n,) - sequence lengths
            n_epochs: Training epochs

        Returns:
            MSE on fake data (lower = better)
        """
        # Detach and clone to ensure clean gradient state for new predictor
        real = real.detach().clone()
        fake = fake.detach().clone()

        feature_dim = real.size(2)

        # Create predictor
        predictor = SequencePredictor(feature_dim).to(self.device)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Train on real data
        predictor.train()
        batch_size = 32

        for epoch in range(n_epochs):
            perm = torch.randperm(real.size(0))
            for i in range(0, real.size(0), batch_size):
                idx = perm[i:i + batch_size]
                batch_data = real[idx]

                # Input: x[:-1], Target: x[1:]
                x_input = batch_data[:, :-1]
                x_target = batch_data[:, 1:]

                optimizer.zero_grad()
                pred = predictor(x_input)
                loss = criterion(pred, x_target)
                loss.backward()
                optimizer.step()

        # Evaluate on fake data
        predictor.eval()
        with torch.no_grad():
            fake_input = fake[:, :-1]
            fake_target = fake[:, 1:]
            fake_pred = predictor(fake_input)
            fake_mse = criterion(fake_pred, fake_target).item()

            # Also compute on real for reference
            real_input = real[:, :-1]
            real_target = real[:, 1:]
            real_pred = predictor(real_input)
            real_mse = criterion(real_pred, real_target).item()

        # Return ratio (closer to 1 is better)
        return fake_mse / (real_mse + 1e-8)

    # =========================================================================
    # Latent Space Metrics
    # =========================================================================

    def compute_latent_metrics(self, real: torch.Tensor, fake: torch.Tensor,
                               conditions: torch.Tensor,
                               lengths: torch.Tensor) -> Dict[str, float]:
        """
        Analyze latent space quality.

        Args:
            real: Real trajectories
            fake: Fake trajectories
            conditions: Conditions
            lengths: Sequence lengths

        Returns:
            Dict with latent space metrics
        """
        # Encode real and fake to latent space
        with torch.no_grad():
            z_real = self.model.encode(real, lengths)
            z_fake = self.model.encode(fake, lengths[:fake.size(0)])

        # Latent MMD
        latent_mmd = self._compute_mmd(z_real, z_fake)

        # Latent statistics
        z_real_mean = z_real.mean(dim=0)
        z_fake_mean = z_fake.mean(dim=0)
        latent_mean_diff = (z_real_mean - z_fake_mean).abs().mean().item()

        z_real_std = z_real.std(dim=0)
        z_fake_std = z_fake.std(dim=0)
        latent_std_diff = (z_real_std - z_fake_std).abs().mean().item()

        # Coverage: what fraction of real latent space is covered by fake
        # Simple proxy: compare ranges
        z_real_min = z_real.min(dim=0)[0]
        z_real_max = z_real.max(dim=0)[0]
        z_fake_min = z_fake.min(dim=0)[0]
        z_fake_max = z_fake.max(dim=0)[0]

        # Coverage = intersection / real_range
        intersection_min = torch.max(z_real_min, z_fake_min)
        intersection_max = torch.min(z_real_max, z_fake_max)
        intersection = (intersection_max - intersection_min).clamp(min=0)
        real_range = (z_real_max - z_real_min).clamp(min=1e-8)
        coverage = (intersection / real_range).mean().item()

        return {
            'latent_mmd': latent_mmd,
            'latent_mean_diff': latent_mean_diff,
            'latent_std_diff': latent_std_diff,
            'latent_coverage': coverage
        }

    # =========================================================================
    # Visualization Utilities
    # =========================================================================

    @torch.no_grad()
    def get_latent_embeddings(self, data: torch.Tensor,
                              lengths: torch.Tensor) -> np.ndarray:
        """
        Get latent embeddings for visualization (e.g., t-SNE).

        Args:
            data: (n, seq_len, feature_dim) - trajectories
            lengths: (n,) - sequence lengths

        Returns:
            embeddings: (n, summary_dim) numpy array
        """
        data = data.to(self.device)
        lengths = lengths.to(self.device)

        z = self.model.encode(data, lengths)
        return z.cpu().numpy()

    @torch.no_grad()
    def visualize_latent_space(self, real: torch.Tensor, fake: torch.Tensor,
                               lengths: torch.Tensor,
                               method: str = 'tsne') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D projections of latent space for visualization.

        Args:
            real: Real trajectories
            fake: Fake trajectories
            lengths: Sequence lengths
            method: 'tsne' or 'pca'

        Returns:
            (real_2d, fake_2d) numpy arrays of shape (n, 2)
        """
        z_real = self.get_latent_embeddings(real, lengths)
        z_fake = self.get_latent_embeddings(fake, lengths[:fake.size(0)])

        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z_all = np.concatenate([z_real, z_fake], axis=0)
            z_2d = pca.fit_transform(z_all)
            real_2d = z_2d[:len(z_real)]
            fake_2d = z_2d[len(z_real):]
        elif method == 'tsne':
            try:
                from sklearn.manifold import TSNE
                z_all = np.concatenate([z_real, z_fake], axis=0)
                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                z_2d = tsne.fit_transform(z_all)
                real_2d = z_2d[:len(z_real)]
                fake_2d = z_2d[len(z_real):]
            except ImportError:
                print("sklearn not available, falling back to PCA")
                return self.visualize_latent_space(real, fake, lengths, method='pca')
        else:
            raise ValueError(f"Unknown method: {method}")

        return real_2d, fake_2d


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_evaluate(model: TimeGANV6, real_data: torch.Tensor,
                   conditions: torch.Tensor, lengths: torch.Tensor = None,
                   n_samples: int = 500) -> Dict[str, float]:
    """
    Quick evaluation with default settings.

    Args:
        model: Trained TimeGANV6 model
        real_data: Real trajectories
        conditions: Conditions
        lengths: Sequence lengths
        n_samples: Number of samples to use

    Returns:
        Evaluation metrics
    """
    evaluator = TimeGANV6Evaluator(model)

    # Subsample if needed
    if real_data.size(0) > n_samples:
        idx = torch.randperm(real_data.size(0))[:n_samples]
        real_data = real_data[idx]
        conditions = conditions[idx]
        if lengths is not None:
            lengths = lengths[idx]

    return evaluator.evaluate(real_data, conditions, lengths)


def print_evaluation_report(metrics: Dict[str, float]):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("TimeGAN V6 Evaluation Report")
    print("=" * 60)

    print("\n--- Distribution Metrics ---")
    print(f"  MMD (↓ better):              {metrics.get('mmd', 'N/A'):.4f}")
    print(f"  Mean Difference (↓ better):  {metrics.get('mean_diff', 'N/A'):.4f}")
    print(f"  Std Difference (↓ better):   {metrics.get('std_diff', 'N/A'):.4f}")
    print(f"  Temporal Diff (↓ better):    {metrics.get('temporal_diff', 'N/A'):.4f}")

    print("\n--- Quality Scores ---")
    disc = metrics.get('discriminative_score', 'N/A')
    if isinstance(disc, float):
        disc_quality = "Excellent" if disc < 0.6 else "Good" if disc < 0.7 else "Fair" if disc < 0.8 else "Poor"
        print(f"  Discriminative (→0.5 better): {disc:.4f} ({disc_quality})")
    else:
        print(f"  Discriminative (→0.5 better): {disc}")

    pred = metrics.get('predictive_score', 'N/A')
    if isinstance(pred, float):
        pred_quality = "Excellent" if pred < 1.5 else "Good" if pred < 2.0 else "Fair" if pred < 3.0 else "Poor"
        print(f"  Predictive (→1.0 better):     {pred:.4f} ({pred_quality})")
    else:
        print(f"  Predictive (→1.0 better):     {pred}")

    print("\n--- Latent Space ---")
    print(f"  Latent MMD (↓ better):        {metrics.get('latent_mmd', 'N/A'):.4f}")
    print(f"  Latent Mean Diff (↓ better):  {metrics.get('latent_mean_diff', 'N/A'):.4f}")
    print(f"  Latent Std Diff (↓ better):   {metrics.get('latent_std_diff', 'N/A'):.4f}")
    print(f"  Latent Coverage (↑ better):   {metrics.get('latent_coverage', 'N/A'):.4f}")

    print("\n" + "=" * 60)


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == '__main__':
    print("Testing TimeGAN V6 Evaluation Suite...")
    print("=" * 60)

    # Create dummy model
    from timegan_v6.config_model_v6 import get_fast_config
    from timegan_v6.model_v6 import TimeGANV6

    config = get_fast_config()
    config.device = 'cpu'
    model = TimeGANV6(config)

    # Create dummy data
    n_samples = 100
    seq_len = 50
    feature_dim = config.feature_dim

    real_data = torch.randn(n_samples, seq_len, feature_dim).clamp(-1, 1)
    conditions = torch.rand(n_samples, config.condition_dim)
    lengths = torch.randint(30, seq_len + 1, (n_samples,))

    # Create evaluator
    print("\n=== Creating Evaluator ===")
    evaluator = TimeGANV6Evaluator(model)
    print("Evaluator created: PASS")

    # Test distribution metrics
    print("\n=== Testing Distribution Metrics ===")
    fake_data = model.generate(n_samples, conditions, seq_len=seq_len)
    dist_metrics = evaluator.compute_distribution_metrics(real_data, fake_data, lengths)
    print(f"MMD: {dist_metrics['mmd']:.4f}")
    print(f"Mean diff: {dist_metrics['mean_diff']:.4f}")
    print(f"Std diff: {dist_metrics['std_diff']:.4f}")
    print("Distribution metrics: PASS")

    # Test discriminative score (quick version)
    print("\n=== Testing Discriminative Score ===")
    disc_score = evaluator.compute_discriminative_score(
        real_data[:50], fake_data[:50], lengths[:50], n_epochs=5
    )
    print(f"Discriminative score: {disc_score:.4f}")
    assert 0 <= disc_score <= 1
    print("Discriminative score: PASS")

    # Test predictive score (quick version)
    print("\n=== Testing Predictive Score ===")
    pred_score = evaluator.compute_predictive_score(
        real_data[:50], fake_data[:50], lengths[:50], n_epochs=5
    )
    print(f"Predictive score: {pred_score:.4f}")
    assert pred_score > 0
    print("Predictive score: PASS")

    # Test latent metrics
    print("\n=== Testing Latent Metrics ===")
    latent_metrics = evaluator.compute_latent_metrics(
        real_data[:50], fake_data[:50], conditions[:50], lengths[:50]
    )
    print(f"Latent MMD: {latent_metrics['latent_mmd']:.4f}")
    print(f"Latent coverage: {latent_metrics['latent_coverage']:.4f}")
    print("Latent metrics: PASS")

    # Test latent embeddings
    print("\n=== Testing Latent Embeddings ===")
    embeddings = evaluator.get_latent_embeddings(real_data[:20], lengths[:20])
    print(f"Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (20, config.summary_dim)
    print("Latent embeddings: PASS")

    # Test full evaluation (quick)
    print("\n=== Testing Full Evaluation (Quick) ===")
    quick_metrics = quick_evaluate(model, real_data[:30], conditions[:30], lengths[:30], n_samples=30)
    print_evaluation_report(quick_metrics)
    print("Full evaluation: PASS")

    print("\n" + "=" * 60)
    print("ALL EVALUATION TESTS PASSED")
    print("=" * 60)
