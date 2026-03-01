import torch
import torch.nn as nn

from src.utils.callbacks import EarlyStopping


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.ones(1))


def test_early_stopping_min():
    model = MockModel()
    early_stopping = EarlyStopping(patience=3, mode="min")

    # 1. First call sets best score
    assert early_stopping(10.0, model) is False
    assert early_stopping.best_score == 10.0
    assert early_stopping.counter == 0

    # 2. Improvement
    assert early_stopping(8.0, model) is False
    assert early_stopping.best_score == 8.0
    assert early_stopping.counter == 0

    # 3. No improvement
    assert early_stopping(9.0, model) is False
    assert early_stopping.counter == 1

    # 4. Another no improvement
    assert early_stopping(8.5, model) is False
    assert early_stopping.counter == 2

    # 5. Improvement resets counter
    assert early_stopping(7.0, model) is False
    assert early_stopping.best_score == 7.0
    assert early_stopping.counter == 0

    # 6. Patience run out
    early_stopping(7.1, model)
    early_stopping(7.2, model)
    assert early_stopping(7.3, model) is True
    assert early_stopping.early_stop is True


def test_early_stopping_max():
    model = MockModel()
    early_stopping = EarlyStopping(patience=2, mode="max")

    # 1. First call sets best score
    early_stopping(0.5, model)
    assert early_stopping.best_score == 0.5

    # 2. Improvement
    early_stopping(0.6, model)
    assert early_stopping.best_score == 0.6
    assert early_stopping.counter == 0

    # 3. No improvement
    early_stopping(0.55, model)
    assert early_stopping.counter == 1

    # 4. Trigger
    assert early_stopping(0.4, model) is True


def test_restore_best_weights():
    model = MockModel()
    early_stopping = EarlyStopping(patience=2, mode="min")

    # Initial state
    model.param.data = torch.tensor([1.0])
    early_stopping(1.0, model)

    # Degrade
    model.param.data = torch.tensor([2.0])
    early_stopping(2.0, model)

    # Restore
    early_stopping.restore_best_weights(model)
    assert model.param.item() == 1.0


def test_min_delta():
    model = MockModel()
    # Improvement must be at least 0.5
    early_stopping = EarlyStopping(patience=2, mode="min", min_delta=0.5)

    early_stopping(10.0, model)

    # 9.6 is less than 10.0 but not by 0.5
    early_stopping(9.6, model)
    assert early_stopping.counter == 1
    assert early_stopping.best_score == 10.0

    # 9.4 is less than 10.0 by more than 0.5
    early_stopping(9.4, model)
    assert early_stopping.counter == 0
    assert early_stopping.best_score == 9.4
