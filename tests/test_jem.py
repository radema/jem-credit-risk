import torch
import pytest
from src.model.jem.config import JEMConfig
from src.model.jem.scaler import TorchStandardScaler
from src.model.jem.model import TabularJEM

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
