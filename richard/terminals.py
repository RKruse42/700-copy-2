"""
The GP terminal set: the 264 shared one-hot indicators plus ephemeral random
constants. Nothing else - GP receives exactly the inputs the GA and the
neurocontroller receive (see features.py).
"""
import numpy as np

from richard.features import FEATURE_COUNT, FEATURE_NAMES

# ephemeral random constants are drawn uniformly from this range
CONST_LOW, CONST_HIGH = -1.0, 1.0


class Terminal:
    """A leaf: either 'read indicator i' or a constant fixed when the leaf was created."""

    __slots__ = ("feature_index", "const_value")

    def __init__(self, feature_index: int | None = None, const_value: float | None = None):
        if (feature_index is None) == (const_value is None):
            raise ValueError("Terminal needs exactly one of feature_index or const_value")
        self.feature_index = feature_index
        self.const_value = const_value

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """X: (n_samples, FEATURE_COUNT). Returns (n_samples,)."""
        if self.feature_index is not None:
            return X[:, self.feature_index]
        return np.full(X.shape[0], self.const_value, dtype=np.float64)

    def __repr__(self):
        if self.feature_index is not None:
            return FEATURE_NAMES[self.feature_index]
        return f"{self.const_value:.3f}"


def random_terminal(rng: np.random.Generator) -> Terminal:
    """Equal chance of an indicator or a fresh ephemeral constant."""
    if rng.random() < 0.5:
        return Terminal(feature_index=int(rng.integers(0, FEATURE_COUNT)))
    return Terminal(const_value=float(rng.uniform(CONST_LOW, CONST_HIGH)))