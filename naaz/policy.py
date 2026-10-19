"""Policy wrapper for the GA-trained Scoring Function.

Adapts the evolved weights into a PolicyFn callable by environment/play.py
and environment/evaluate.py (raw 25-cell 5x5 windows in, scores out).
"""
from __future__ import annotations

import os
import numpy as np

from simon.encoding import one_hot


def make_ga_policy(weights: np.ndarray, bias: float):
    """Returns a policy_fn(raw_features) -> scores compatible with environment.play.

    Args:
        weights: 1D array of 264 feature weights.
        bias: Float scalar intercept.

    Returns:
        policy: Callable taking (n_cells, 25) raw window features and returning
                (n_cells,) desirability scores.

    Note on polarity:
        In environment.play.choose_move(), the agent selects best_index = np.argmax(scores).
        Since our model predicts P(mine), we return -P(mine) so that
        the safest cell (lowest mine probability) receives the highest score.
    """
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    bias = float(bias)

    def policy(raw_features: np.ndarray) -> np.ndarray:
        # If input has 25 cells (full 5x5 window), drop center cell (index 12) -> 24 cells
        if raw_features.ndim == 2 and raw_features.shape[1] == 25:
            raw_features = np.delete(raw_features, 12, axis=1)
        elif raw_features.ndim == 1 and len(raw_features) == 25:
            raw_features = np.delete(raw_features, 12)

        # One-hot encode the 24 surrounding cells into 264 binary features
        X = one_hot(raw_features, 11, flatten=True)
        z = np.clip(np.dot(X, weights) + bias, -15.0, 15.0)
        p_mine = 1.0 / (1.0 + np.exp(-z))
        # Negative mine probability: higher score = lower mine risk (safest move)
        return -p_mine

    return policy


def load_ga_policy(weights_path: str | None = None):
    """Loads evolved weights from an .npz file and returns a ready-to-play policy."""
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "ga_best_weights.npz")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file '{weights_path}' not found. Run 'python3 -m naaz.runner' first."
        )
    data = np.load(weights_path)
    return make_ga_policy(data["weights"], float(data["bias"]))
