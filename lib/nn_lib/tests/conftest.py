"""Shared test fixtures for nn_lib tests."""

import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_input(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Batch of 8 samples with 4 features."""
    return rng.standard_normal((8, 4))


@pytest.fixture
def sample_targets_onehot(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """One-hot encoded targets for 3 classes, batch size 8."""
    targets = np.zeros((8, 3))
    classes = rng.integers(0, 3, size=8)
    targets[np.arange(8), classes] = 1.0
    return targets
