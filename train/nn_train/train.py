"""Training stage: build network from params, train, save model artifact."""

import logging
from typing import TypedDict, get_args

import numpy as np
import numpy.typing as npt
import yaml

from nn_lib import MultiLayerNetwork, Trainer, TrainerHyperparams
from nn_lib.network import ActivationType
from nn_lib.trainer import LossType
from nn_train._paths import ARTIFACTS_DIR, PARAMS_PATH

logger = logging.getLogger(__name__)


class TrainParams(TypedDict):
    """Combined network architecture and trainer params from params.yaml."""

    neurons: list[int]
    activations: list[str]
    batch_size: int
    epochs: int
    learning_rate: float
    loss_fun: str
    shuffle: bool


def _load_params() -> TrainParams:
    """Load the train stage parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        all_params: dict[str, TrainParams] = yaml.safe_load(f)
    return all_params["train"]


def _load_training_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load the preprocessed training arrays from artifacts/."""
    logger.info("Loading training data from %s", ARTIFACTS_DIR)
    x_train: npt.NDArray[np.float64] = np.load(ARTIFACTS_DIR / "train_x.npy")
    y_train: npt.NDArray[np.float64] = np.load(ARTIFACTS_DIR / "train_y.npy")
    return x_train, y_train


_VALID_ACTIVATIONS: frozenset[str] = frozenset(get_args(ActivationType))
_VALID_LOSSES: frozenset[str] = frozenset(get_args(LossType))


def _validate_activations(raw: list[str]) -> list[ActivationType]:
    """Validate activation names from params.yaml against known literals."""
    for name in raw:
        if name not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {name!r} in params.yaml. "
                f"Choose from {sorted(_VALID_ACTIVATIONS)}"
            )
    # Safe after validation — each element is a valid ActivationType literal
    return raw  # type: ignore[return-value]


def _validate_loss(raw: str) -> LossType:
    """Validate a loss name from params.yaml against known literals."""
    if raw not in _VALID_LOSSES:
        raise ValueError(
            f"Unknown loss {raw!r} in params.yaml. Choose from {sorted(_VALID_LOSSES)}"
        )
    return raw  # type: ignore[return-value]


def _build_and_train(
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    params: TrainParams,
) -> MultiLayerNetwork:
    """Construct the network and train it.

    Args:
        x_train: Preprocessed training features.
        y_train: Training target labels.
        params: Train stage parameters from params.yaml.

    Returns:
        The trained network.
    """
    input_dim = x_train.shape[1]
    activations = _validate_activations(params["activations"])
    loss_fun = _validate_loss(params["loss_fun"])

    network = MultiLayerNetwork(
        input_dim=input_dim,
        neurons=params["neurons"],
        activations=activations,
    )

    hyperparams: TrainerHyperparams = {
        "batch_size": params["batch_size"],
        "epochs": params["epochs"],
        "learning_rate": params["learning_rate"],
        "loss": loss_fun,
        "shuffle": params["shuffle"],
    }

    trainer = Trainer(network=network, hyperparams=hyperparams)

    logger.info(
        "Training: %d epochs, batch_size=%d, lr=%s",
        params["epochs"],
        params["batch_size"],
        params["learning_rate"],
    )
    trainer.train(x_train, y_train)

    train_loss = trainer.eval_loss(x_train, y_train)
    logger.info("Final training loss: %.6f", train_loss)
    return network


def _save_model(network: MultiLayerNetwork) -> None:
    """Serialize the trained model to artifacts/."""
    model_path = ARTIFACTS_DIR / "model.pkl"
    network.save(model_path)
    logger.info("Model saved to %s", model_path)


def main() -> None:
    """Orchestrate the train stage."""
    # Setup
    params = _load_params()
    x_train, y_train = _load_training_data()

    # Train
    network = _build_and_train(x_train, y_train, params)

    # Save
    _save_model(network)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
