"""Tests for the Trainer class."""

import numpy as np
import pytest

from nn_lib.network import MultiLayerNetwork
from nn_lib.trainer import Trainer, TrainerHyperparams


class TestTrainer:
    def test_training_reduces_loss(self) -> None:
        """Training for multiple epochs should reduce the loss."""
        np.random.seed(42)
        net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"])
        x = np.random.default_rng(0).standard_normal((20, 4))
        y = np.zeros((20, 3))
        y[np.arange(20), np.random.default_rng(0).integers(0, 3, 20)] = 1.0

        hyperparams: TrainerHyperparams = {
            "batch_size": 8,
            "nb_epoch": 1,
            "learning_rate": 0.01,
            "loss_fun": "cross_entropy",
            "shuffle_flag": False,
        }
        trainer = Trainer(network=net, hyperparams=hyperparams)
        loss_before = trainer.eval_loss(x, y)

        trainer.nb_epoch = 200
        trainer.train(x, y)
        loss_after = trainer.eval_loss(x, y)

        assert loss_after < loss_before

    def test_mse_loss_training(self) -> None:
        """Verify that MSE loss mode works and reduces loss."""
        np.random.seed(42)
        net = MultiLayerNetwork(2, [8, 1], ["sigmoid", "identity"])
        x = np.random.default_rng(0).standard_normal((16, 2))
        y = np.random.default_rng(0).standard_normal((16, 1))

        hyperparams: TrainerHyperparams = {
            "batch_size": 4,
            "nb_epoch": 100,
            "learning_rate": 0.01,
            "loss_fun": "mse",
            "shuffle_flag": True,
        }
        trainer = Trainer(network=net, hyperparams=hyperparams)
        loss_before = trainer.eval_loss(x, y)
        trainer.train(x, y)
        loss_after = trainer.eval_loss(x, y)

        assert loss_after < loss_before

    def test_unknown_loss_raises(self) -> None:
        net = MultiLayerNetwork(4, [3], ["identity"])
        hyperparams: TrainerHyperparams = {
            "batch_size": 8,
            "nb_epoch": 1,
            "learning_rate": 0.01,
            "loss_fun": "bad_loss",  # type: ignore[typeddict-item]
            "shuffle_flag": False,
        }
        with pytest.raises(ValueError, match="Unknown loss"):
            Trainer(network=net, hyperparams=hyperparams)

    def test_eval_loss_does_not_change_weights(self) -> None:
        np.random.seed(42)
        net = MultiLayerNetwork(4, [3], ["identity"])
        x = np.random.default_rng(0).standard_normal((4, 4))
        y = np.zeros((4, 3))
        y[np.arange(4), [0, 1, 2, 0]] = 1.0

        hyperparams: TrainerHyperparams = {
            "batch_size": 4,
            "nb_epoch": 1,
            "learning_rate": 0.01,
            "loss_fun": "cross_entropy",
            "shuffle_flag": False,
        }
        trainer = Trainer(network=net, hyperparams=hyperparams)

        output_before = net.forward(x).copy()
        trainer.eval_loss(x, y)
        output_after = net.forward(x)

        np.testing.assert_array_equal(output_before, output_after)
