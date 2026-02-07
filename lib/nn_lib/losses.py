"""Loss function implementations for neural network training."""

import numpy as np
import numpy.typing as npt


class MSELoss:
    """Mean Squared Error loss for regression tasks.

    L = (1/n) * sum((y_pred - y_target)^2)
    """

    def __init__(self) -> None:
        self._cache: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None

    def forward(
        self,
        y_pred: npt.NDArray[np.float64],
        y_target: npt.NDArray[np.float64],
    ) -> float:
        self._cache = (y_pred, y_target)
        n = y_pred.shape[0]
        return float(np.sum((y_pred - y_target) ** 2) / n)

    def backward(self) -> npt.NDArray[np.float64]:
        if self._cache is None:
            raise ValueError("forward() must be called before backward()")
        y_pred, y_target = self._cache
        n = y_pred.shape[0]
        result: npt.NDArray[np.float64] = 2.0 * (y_pred - y_target) / n
        return result


class CrossEntropyLoss:
    """Cross-entropy loss with built-in softmax for multi-class classification.

    Applies softmax to predictions, then computes:
    L = -(1/n) * sum(y_target * log(softmax(y_pred)))
    """

    def __init__(self) -> None:
        self._cache: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None

    @staticmethod
    def _softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Numerically stable softmax: subtract max per row before exp."""
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        result: npt.NDArray[np.float64] = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return result

    def forward(
        self,
        y_pred: npt.NDArray[np.float64],
        y_target: npt.NDArray[np.float64],
    ) -> float:
        if y_pred.shape[0] != y_target.shape[0]:
            raise ValueError(
                f"Batch size mismatch: predictions {y_pred.shape[0]}, targets {y_target.shape[0]}"
            )
        probs = self._softmax(y_pred)
        self._cache = (probs, y_target)
        n = y_pred.shape[0]
        eps = 1e-8
        return float(-np.sum(y_target * np.log(probs + eps)) / n)

    def backward(self) -> npt.NDArray[np.float64]:
        if self._cache is None:
            raise ValueError("forward() must be called before backward()")
        probs, y_target = self._cache
        n = probs.shape[0]
        result: npt.NDArray[np.float64] = (probs - y_target) / n
        return result
