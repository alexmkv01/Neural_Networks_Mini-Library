"""Evaluation stage: load model, compute metrics, write evaluate-metrics.json."""

import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nn_lib import MultiLayerNetwork
from nn_lib.losses import CrossEntropyLoss

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _load_model_and_data() -> tuple[
    MultiLayerNetwork,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Load the trained model and validation data from artifacts/."""
    logger.info("Loading model and validation data from %s", ARTIFACTS_DIR)
    model = MultiLayerNetwork.load(ARTIFACTS_DIR / "model.pkl")
    x_val: npt.NDArray[np.float64] = np.load(ARTIFACTS_DIR / "val_x.npy")
    y_val: npt.NDArray[np.float64] = np.load(ARTIFACTS_DIR / "val_y.npy")
    return model, x_val, y_val


def compute_accuracy(
    predictions: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
) -> float:
    """Compute classification accuracy from one-hot predictions and targets."""
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    return float(np.mean(pred_classes == true_classes))


def evaluate(
    model: MultiLayerNetwork,
    x_val: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Run evaluation and return metrics dict.

    Args:
        model: Trained neural network.
        x_val: Preprocessed validation features.
        y_val: Validation target labels (one-hot).

    Returns:
        Dictionary with val_loss and val_accuracy.
    """
    predictions = model.forward(x_val)

    loss_fn = CrossEntropyLoss()
    val_loss = loss_fn.forward(predictions, y_val)
    accuracy = compute_accuracy(predictions, y_val)

    logger.info("Validation loss: %.6f", val_loss)
    logger.info("Validation accuracy: %.2f%%", accuracy * 100)

    return {
        "val_loss": round(val_loss, 6),
        "val_accuracy": round(accuracy, 4),
    }


def _save_metrics(metrics: dict[str, float]) -> None:
    """Write metrics dict to artifacts/evaluate-metrics.json."""
    metrics_path = ARTIFACTS_DIR / "evaluate-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)


def main() -> None:
    """Orchestrate the evaluate stage."""
    # Setup
    model, x_val, y_val = _load_model_and_data()

    # Evaluate
    metrics = evaluate(model, x_val, y_val)

    # Save
    _save_metrics(metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
