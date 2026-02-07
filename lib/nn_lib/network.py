"""Multi-layer feedforward neural network."""

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from nn_lib.activations import IdentityLayer, ReluLayer, SigmoidLayer, TanhLayer
from nn_lib.base import Layer
from nn_lib.layers import LinearLayer

ActivationType = Literal["relu", "sigmoid", "tanh", "identity"]

_ACTIVATION_MAP: dict[str, type[Layer]] = {
    "relu": ReluLayer,
    "sigmoid": SigmoidLayer,
    "tanh": TanhLayer,
    "identity": IdentityLayer,
}


class MultiLayerNetwork:
    """A feedforward neural network composed of linear layers and activations.

    Each entry in `neurons` and `activations` defines one layer group:
    a LinearLayer followed by an activation layer.

    Example:
        >>> net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"])
        >>> output = net(input_data)
    """

    def __init__(
        self,
        input_dim: int,
        neurons: list[int],
        activations: list[ActivationType],
    ) -> None:
        if len(neurons) != len(activations):
            raise ValueError(
                f"neurons ({len(neurons)}) and activations ({len(activations)}) "
                f"must have the same length"
            )
        self._layers = _build_layers(input_dim, neurons, activations)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward pass through all layers sequentially."""
        output = x
        for layer_group in self._layers:
            for layer in layer_group:
                output = layer.forward(output)
        return output

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Backward pass through all layers in reverse order."""
        grad = grad_z
        for layer_group in reversed(self._layers):
            for layer in reversed(layer_group):
                grad = layer.backward(grad)
        return grad

    def update_params(self, learning_rate: float) -> None:
        """Update parameters of all learnable layers."""
        for layer_group in self._layers:
            for layer in layer_group:
                layer.update_params(learning_rate)

    def save(self, path: Path) -> None:
        """Serialize the network to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "MultiLayerNetwork":
        """Deserialize a network from a pickle file."""
        with open(path, "rb") as f:
            network: MultiLayerNetwork = pickle.load(f)
        return network

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.forward(x)


def _build_layers(
    input_dim: int,
    neurons: list[int],
    activations: list[ActivationType],
) -> list[list[Layer]]:
    """Construct layer groups from architecture specification.

    Each group is [LinearLayer, ActivationLayer].
    """
    layers: list[list[Layer]] = []
    prev_dim = input_dim
    for n_out, act_name in zip(neurons, activations, strict=True):
        linear = LinearLayer(prev_dim, n_out)
        activation = _create_activation(act_name)
        layers.append([linear, activation])
        prev_dim = n_out
    return layers


def _create_activation(name: ActivationType) -> Layer:
    """Instantiate an activation layer by name."""
    cls = _ACTIVATION_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown activation: {name!r}. Choose from {list(_ACTIVATION_MAP)}")
    return cls()
