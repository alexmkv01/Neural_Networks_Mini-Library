"""Training stage: build network from params, train, save model artifact."""

import logging
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import yaml

from nn_lib import MultiLayerNetwork, Trainer, TrainerHyperparams
from nn_lib.network import ActivationType
from nn_lib.trainer import LossType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


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
    activations = cast(list[ActivationType], params["activations"])

    network = MultiLayerNetwork(
        input_dim=input_dim,
        neurons=params["neurons"],
        activations=activations,
    )

    hyperparams: TrainerHyperparams = {
        "batch_size": params["batch_size"],
        "nb_epoch": params["epochs"],
        "learning_rate": params["learning_rate"],
        "loss_fun": cast(LossType, params["loss_fun"]),
        "shuffle_flag": params["shuffle"],
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
