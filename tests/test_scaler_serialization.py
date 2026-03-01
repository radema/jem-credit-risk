import torch
import shutil
from pathlib import Path
from src.model.jem.scaler import TorchStandardScaler


def test_scaler_serialization():
    # Arrange
    num_features = 10
    batch_size = 50
    scaler = TorchStandardScaler(num_features=num_features)

    # Create random dummy data
    dummy_data = torch.randn(batch_size, num_features) * 5.0 + 2.0

    # Fit scaler
    scaler.fit(dummy_data)

    # Transform with original
    original_transformed = scaler.transform(dummy_data)

    # Save scaler
    artifact_dir = Path("models/test_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = artifact_dir / "scaler.pt"

    scaler.save(scaler_path)

    # Load scaler
    loaded_scaler = TorchStandardScaler.load(scaler_path)

    # Transform with loaded scaler
    loaded_transformed = loaded_scaler.transform(dummy_data)

    # Assert
    assert torch.allclose(original_transformed, loaded_transformed), (
        "Transformed outputs do not match!"
    )
    assert scaler.num_features == loaded_scaler.num_features, (
        "Num features don't match!"
    )
    assert scaler.eps == loaded_scaler.eps, "Eps doesn't match!"
    assert torch.allclose(scaler.mean, loaded_scaler.mean), "Mean doesn't match!"
    assert torch.allclose(scaler.var, loaded_scaler.var), "Var doesn't match!"

    print("Scaler serialization test passed successfully!")

    # Cleanup
    shutil.rmtree(artifact_dir)


if __name__ == "__main__":
    test_scaler_serialization()
