"""Data preparation stage: load, split, preprocess, and save artifacts."""

import logging
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import yaml

from nn_lib import Preprocessor
from nn_train._paths import ARTIFACTS_DIR, DATA_PATH, PARAMS_PATH

logger = logging.getLogger(__name__)

_N_FEATURES = 4


class PrepareParams(TypedDict):
    """Typed parameters for the prepare stage from params.yaml."""

    test_split: float
    random_seed: int


def _load_params() -> PrepareParams:
    """Load the prepare stage parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        all_params: dict[str, PrepareParams] = yaml.safe_load(f)
    return all_params["prepare"]


def load_and_split(
    data_path: Path,
    test_split: float,
    random_seed: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Load iris dataset and split into train/validation sets.

    Args:
        data_path: Path to the iris.dat file (4 features + 3 one-hot labels).
        test_split: Fraction of data to reserve for validation.
        random_seed: Seed for reproducible shuffling.

    Returns:
        Tuple of (x_train, x_val, y_train, y_val).
    """
    logger.info("Loading data from %s", data_path)
    data: npt.NDArray[np.float64] = np.loadtxt(data_path)
    rng = np.random.default_rng(seed=random_seed)
    indices = rng.permutation(data.shape[0])
    data = data[indices]

    split_idx = int(data.shape[0] * (1 - test_split))
    x_train: npt.NDArray[np.float64] = data[:split_idx, :_N_FEATURES]
    y_train: npt.NDArray[np.float64] = data[:split_idx, _N_FEATURES:]
    x_val: npt.NDArray[np.float64] = data[split_idx:, :_N_FEATURES]
    y_val: npt.NDArray[np.float64] = data[split_idx:, _N_FEATURES:]
    logger.info("Split: %d train, %d val", x_train.shape[0], x_val.shape[0])
    return x_train, x_val, y_train, y_val


def _fit_and_apply_preprocessor(
    x_train: npt.NDArray[np.float64],
    x_val: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Preprocessor]:
    """Fit preprocessor on training data and apply to both splits."""
    logger.info("Fitting preprocessor on %d training samples", x_train.shape[0])
    preprocessor = Preprocessor(x_train)
    return preprocessor.apply(x_train), preprocessor.apply(x_val), preprocessor


def _save_artifacts(
    x_train: npt.NDArray[np.float64],
    x_val: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.float64],
    preprocessor: Preprocessor,
) -> None:
    """Save prepared data arrays and fitted preprocessor to artifacts/."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / "train_x.npy", x_train)
    np.save(ARTIFACTS_DIR / "val_x.npy", x_val)
    np.save(ARTIFACTS_DIR / "train_y.npy", y_train)
    np.save(ARTIFACTS_DIR / "val_y.npy", y_val)
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Saved artifacts to %s", ARTIFACTS_DIR)


def main() -> None:
    """Orchestrate the prepare stage."""
    # Setup
    params = _load_params()
    test_split = params["test_split"]
    random_seed = params["random_seed"]

    # Process
    x_train, x_val, y_train, y_val = load_and_split(DATA_PATH, test_split, random_seed)
    x_train, x_val, preprocessor = _fit_and_apply_preprocessor(x_train, x_val)

    # Save
    _save_artifacts(x_train, x_val, y_train, y_val, preprocessor)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
