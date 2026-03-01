import pytest
import torch

from src.model.jem.scaler import TorchStandardScaler


def test_fit_updates_buffers():
    """Verify that fit() correctly computes and updates mean and variance buffers."""
    num_features = 3
    scaler = TorchStandardScaler(num_features=num_features)

    # Input data: 4 samples, 3 features
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )

    # Expected mean and variance (unbiased)
    # mean: [5.5, 6.5, 7.5]
    # var:
    #   f1: ((1-5.5)^2 + (4-5.5)^2 + (7-5.5)^2 + (10-5.5)^2) / 3
    #       = ((-4.5)^2 + (-1.5)^2 + (1.5)^2 + (4.5)^2) / 3
    #       = (20.25 + 2.25 + 2.25 + 20.25) / 3 = 45 / 3 = 15.0
    expected_mean = torch.tensor([5.5, 6.5, 7.5])
    expected_var = torch.tensor([15.0, 15.0, 15.0])

    scaler.fit(x)

    assert torch.allclose(scaler.mean, expected_mean)
    assert torch.allclose(scaler.var, expected_var)
    assert scaler.is_fitted.item() is True


def test_fit_returns_self():
    """Verify that fit() returns the scaler instance for chaining."""
    scaler = TorchStandardScaler(num_features=2)
    x = torch.randn(5, 2)
    result = scaler.fit(x)
    assert result is scaler


def test_fit_invalid_dimensions():
    """Verify ValueError is raised for non-2D tensors."""
    scaler = TorchStandardScaler(num_features=2)

    # 1D tensor
    with pytest.raises(ValueError, match="Expected 2D tensor"):
        scaler.fit(torch.randn(5))

    # 3D tensor
    with pytest.raises(ValueError, match="Expected 2D tensor"):
        scaler.fit(torch.randn(5, 2, 1))


def test_fit_invalid_feature_count():
    """Verify ValueError is raised for incorrect feature count."""
    scaler = TorchStandardScaler(num_features=5)

    # Input with 3 features instead of 5
    with pytest.raises(ValueError, match="Expected 5 features, got 3"):
        scaler.fit(torch.randn(10, 3))


def test_transform_unfitted_raises_error():
    """Verify RuntimeError is raised if transform is called before fit."""
    scaler = TorchStandardScaler(num_features=2)
    x = torch.randn(5, 2)

    with pytest.raises(RuntimeError, match="Scaler has not been fitted yet"):
        scaler.transform(x)


def test_transform_scaling():
    """Verify transform() correctly scales data using fitted parameters."""
    num_features = 2
    scaler = TorchStandardScaler(num_features=num_features)

    # Data with mean=[2, 10] and std=[1, 5] (var=[1, 25])
    x_train = torch.tensor([[1.0, 5.0], [3.0, 15.0]])

    # Mean: [2.0, 10.0]
    # Var: [((1-2)^2 + (3-2)^2)/1, ((5-10)^2 + (15-10)^2)/1] = [2.0, 50.0]

    scaler.fit(x_train)

    x_test = torch.tensor(
        [
            [2.0, 10.0],  # Should become [0, 0]
            [2.0 + (2.0**0.5), 10.0 + (50.0**0.5)],  # Should become [1, 1]
        ]
    )

    x_scaled = scaler.transform(x_test)

    expected_scaled = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    assert torch.allclose(x_scaled, expected_scaled)


def test_transform_zero_variance():
    """Verify handling of zero-variance features using eps."""
    num_features = 2
    eps = 1e-8
    scaler = TorchStandardScaler(num_features=num_features, eps=eps)

    # Feature 0 is constant
    x = torch.tensor([[10.0, 1.0], [10.0, 3.0]])

    scaler.fit(x)
    assert scaler.var[0] < eps

    # Transform
    x_scaled = scaler.transform(x)

    # Feature 0: (10 - 10) / 1.0 = 0.0
    # Feature 1: mean=2, var=2, std=sqrt(2). (1-2)/sqrt(2) = -0.7071, (3-2)/sqrt(2) = 0.7071
    assert torch.allclose(x_scaled[0, 0], torch.tensor(0.0))
    assert torch.allclose(x_scaled[1, 0], torch.tensor(0.0))

    expected_f1_0 = (torch.tensor(1.0) - 2.0) / (2.0**0.5)
    expected_f1_1 = (torch.tensor(3.0) - 2.0) / (2.0**0.5)
    assert torch.allclose(x_scaled[0, 1], expected_f1_0)
    assert torch.allclose(x_scaled[1, 1], expected_f1_1)


def test_fit_transform():
    """Verify fit_transform matches fit followed by transform."""
    scaler = TorchStandardScaler(num_features=2)
    x = torch.randn(10, 2)

    # Using fit_transform
    x_scaled_1 = scaler.fit_transform(x)

    # Manual fit and transform
    scaler2 = TorchStandardScaler(num_features=2)
    scaler2.fit(x)
    x_scaled_2 = scaler2.transform(x)

    assert torch.allclose(x_scaled_1, x_scaled_2)
    assert scaler.is_fitted.item() is True
