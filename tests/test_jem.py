import os

import numpy as np
import polars as pl
import pytest
import torch

from src.model.jem.config import JEMConfig
from src.model.jem.diagnostics import JEMDiagnostics
from src.model.jem.data_utils import CreditRiskDataset, calculate_gini_stability
from src.model.jem.model import TabularJEM
from src.model.jem.scaler import TorchStandardScaler

# We will import SGLDReplayBuffer here, but it's not implemented yet.
try:
    from src.model.jem.sampler import SGLDReplayBuffer
except ImportError:
    pass


def test_torch_standard_scaler():
    """Test TorchStandardScaler components for Task 1.2."""
    scaler = TorchStandardScaler(num_features=5)

    x = torch.randn(10, 5) * 2.0 + 5.0
    x[:, 0] = 1.0  # Constant feature to test zero-variance logic

    # Needs fit before transform
    with pytest.raises(RuntimeError):
        scaler.transform(x)

    scaler.fit(x)
    assert scaler.is_fitted.item() is True

    x_scaled = scaler.transform(x)
    assert x_scaled.shape == (10, 5)

    # Check that scaled data has zero mean and unit variance (roughly)
    # The constant feature (idx 0) should be near zero, standard deviation should be near 1.0 for others
    means = x_scaled.mean(dim=0)
    stds = x_scaled.std(dim=0, unbiased=True)

    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    # the constant feature std will be exactly 0, not 1, because values are all exactly the mean.
    assert stds[0] == 0.0


def test_tabular_jem_architecture():
    """Test TabularJEM correctly processes inputs and outputs energies for Task 1.3."""
    config = JEMConfig(hidden_dims=[32, 16], spectral_norm=True)
    model = TabularJEM(input_dim=10, num_classes=2, config=config)

    x = torch.randn(8, 10)

    # Test forward pass (logits)
    logits = model(x)
    assert logits.shape == (8, 2)

    # Test compute_energy
    energy = model.compute_energy(x)
    assert energy.shape == (8,)


def test_sgld_replay_buffer_initialization():
    """Test SGLDReplayBuffer for Task 2.1 (TDD)."""
    # This should fail if SGLDReplayBuffer is not implemented yet.
    if "SGLDReplayBuffer" not in globals():
        pytest.fail("SGLDReplayBuffer has not been implemented yet!")

    buffer_size = 100
    feature_dim = 10
    buffer = SGLDReplayBuffer(buffer_size=buffer_size, feature_dim=feature_dim)

    assert buffer.buffer.shape == (buffer_size, feature_dim)
    assert buffer.pointer == 0
    assert buffer.is_full is False


def test_sgld_replay_buffer_sample_and_update():
    """Test SGLDReplayBuffer correctly samples (95% buffer, 5% noise) for Task 2.1."""
    if "SGLDReplayBuffer" not in globals():
        pytest.fail("SGLDReplayBuffer has not been implemented yet!")

    buffer_size = 100
    feature_dim = 10
    buffer = SGLDReplayBuffer(buffer_size=buffer_size, feature_dim=feature_dim)

    # Seed buffer artificially
    fake_samples = torch.ones(50, feature_dim)
    buffer.update(fake_samples)
    assert buffer.pointer == 50
    assert buffer.is_full is False

    # Now sample 20 items.
    # 95% should be from previous 50 (which are ones), 5% should be uniform noise [-1, 1]
    # In a batch of 20, 1 will be noise, 19 will be from buffer.
    samples = buffer.sample(batch_size=20)
    assert samples.shape == (20, feature_dim)


def test_sgld_sampler_generate():
    """Test SGLDSampler generates fake samples with proper gradient flows for Task 2.2."""
    try:
        from src.model.jem.sampler import SGLDSampler
    except ImportError:
        pytest.fail("SGLDSampler has not been implemented yet!")

    input_dim = 10
    num_classes = 2
    buffer_size = 100
    batch_size = 8

    config = JEMConfig(
        hidden_dims=[32], sgld_steps=5, sgld_step_size=0.1, sgld_sigma=0.01
    )
    model = TabularJEM(input_dim=input_dim, num_classes=num_classes, config=config)
    buffer = SGLDReplayBuffer(buffer_size=buffer_size, feature_dim=input_dim)
    sampler = SGLDSampler(config=config)

    # Run sampler
    # Note: sampler should move init_samples to model device
    x_fake = sampler.generate(model, buffer, batch_size=batch_size)

    assert x_fake.shape == (batch_size, input_dim)
    # Ensure x_fake is detached (no gradients)
    assert x_fake.grad_fn is None
    # Buffer should have been updated
    assert buffer.pointer == batch_size


def test_jem_loss_calculation():
    """Test JEMLoss formulation for Task 2.3 (TDD)."""
    try:
        from src.model.jem.loss import JEMLoss
    except ImportError:
        pytest.fail("JEMLoss has not been implemented yet!")

    input_dim = 10
    num_classes = 2
    batch_size = 4

    config = JEMConfig(l2_energy_weight=0.1)
    model = TabularJEM(input_dim=input_dim, num_classes=num_classes, config=config)
    criterion = JEMLoss(config=config)

    x_real = torch.randn(batch_size, input_dim)
    y_real = torch.randint(0, num_classes, (batch_size,))
    x_fake = torch.randn(batch_size, input_dim)

    loss_dict = criterion(model, x_real, y_real, x_fake)

    assert "total_loss" in loss_dict
    assert "clf_loss" in loss_dict
    assert "gen_loss" in loss_dict
    assert "l2_loss" in loss_dict

    assert loss_dict["total_loss"].item() > 0
    assert not torch.isnan(loss_dict["total_loss"])


def test_gini_stability_metric():
    """Test Gini Stability Metric for Task 3.1."""
    # Create fake labels and predictions
    # Suppose we have 3 weeks: week 0, week 1, week 2
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # Perfect predictions for week 0, slightly worse for week 1, even worse for week 2
    y_pred = np.array(
        [
            0.1,
            0.9,
            0.2,
            0.8,  # week 0: AUC 1.0 -> Gini 1.0
            0.3,
            0.7,
            0.4,
            0.6,  # week 1: AUC 1.0 -> Gini 1.0
            0.5,
            0.5,
            0.4,
            0.6,  # week 2: AUC depends, let's just make it valid
        ]
    )
    week_nums = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    results = calculate_gini_stability(y_true, y_pred, week_nums)

    assert "stability_metric" in results
    assert "falling_rate" in results
    assert "weekly_ginis" in results
    assert len(results["weekly_ginis"]) == 3
    # Falling rate should be <= 0
    assert results["falling_rate"] <= 0.0


def test_credit_risk_dataset():
    """Test the dataset logic correctly scales features for Task 3.1."""
    df = pl.DataFrame(
        {
            "case_id": [1, 2, 3],
            "MONTH": [1, 1, 1],
            "WEEK_NUM": [0, 0, 1],
            "target": [0, 1, 0],
            "feat_1": [10.0, 20.0, 30.0],
            "feat_2": [100.0, 200.0, 300.0],
        }
    )

    scaler = TorchStandardScaler(num_features=2)
    dataset = CreditRiskDataset(
        df, feature_cols=["feat_1", "feat_2"], scaler=scaler, is_train=True
    )

    # Assert scaling was applied
    assert dataset.scaler.is_fitted.item() is True
    assert dataset.features.shape == (3, 2)

    # Check item getters
    x, y, w = dataset[0]
    assert x.shape == (2,)
    assert y.item() == 0
    assert w.item() == 0


def test_jem_diagnostics():
    """Test JEMDiagnostics plots code for Task 3.2."""
    if "SGLDReplayBuffer" not in globals():
        pytest.skip("SGLDReplayBuffer has not been implemented yet!")

    e_real = [1.0, 0.8, 0.6]
    e_fake = [5.0, 4.0, 3.0]
    e_in_dist = np.array([1.2, 0.9, 1.1])
    e_out_dist = np.array([8.0, 9.1, 7.5])

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.3, 0.8])

    buffer = SGLDReplayBuffer(buffer_size=10, feature_dim=5)
    buffer.update(torch.randn(10, 5))
    x_real = torch.randn(15, 5)

    # Run all methods to ensure no exceptions are raised during plotting
    JEMDiagnostics.plot_energy_ranges(e_real, e_fake, save_path="test_ranges.png")
    JEMDiagnostics.plot_energy_density(
        e_in_dist, e_out_dist, save_path="test_density.png"
    )
    JEMDiagnostics.plot_reliability_diagram(
        y_true, y_prob, num_bins=2, save_path="test_rel.png"
    )
    JEMDiagnostics.inspect_replay_buffer(
        buffer, x_real, save_path="test_buffer.png", n_samples=5
    )

    # Assert files are created and delete them
    for fname in [
        "test_ranges.png",
        "test_density.png",
        "test_rel.png",
        "test_buffer.png",
    ]:
        assert os.path.exists(fname)
        os.remove(fname)
