"""Scoring function for Minesweeper cell evaluation.

This model computes a direct weighted combination of the input features,
mapped through a sigmoid function to output predicted mine probability (0 to 1).
"""

import numpy as np


class ScoringFunction:
    """Linear / Logistic Scoring Function.

    Attributes:
        feature_count: Number of input features (264 for one-hot 24 surrounding cells in 5x5 window).
        total_weights: Total parameters to evolve = feature_count + 1 (bias).
    """

    feature_count: int
    total_weights: int

    def __init__(self, feature_count: int = 264) -> None:
        self.feature_count = feature_count
        # Each feature gets a weight, plus one bias term at the end
        self.total_weights = feature_count + 1

    def evaluate_batch(
        self, X: np.ndarray, chromosomes: np.ndarray
    ) -> np.ndarray:
        """Vectorized evaluation of multiple candidate chromosomes on dataset X.

        Args:
            X: Input features of shape (n_samples, feature_count).
            chromosomes: Population weight matrix of shape (n_individuals, total_weights).
                         Each row is [w_0, w_1, ..., w_{feature_count-1}, bias].

        Returns:
            Predictions matrix of shape (n_individuals, n_samples) with values in (0, 1).
        """
        chromosomes = np.ascontiguousarray(chromosomes, dtype=np.float64)
        if chromosomes.ndim == 1:
            chromosomes = chromosomes[None, :]

        weights = chromosomes[:, :-1]   # (n_individuals, feature_count)
        biases = chromosomes[:, -1:]    # (n_individuals, 1)

        # Vectorized linear combination: (n_individuals, n_samples)
        z = np.matmul(weights, X.T) + biases

        # Numerical stability clipping for sigmoid
        z = np.clip(z, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-z))

    def evaluate(self, x: np.ndarray, chromosome: np.ndarray) -> float:
        """Evaluates a single cell feature vector using one chromosome.

        Args:
            x: Feature vector of shape (feature_count,).
            chromosome: Weight vector of shape (total_weights,).

        Returns:
            Predicted probability (float).
        """
        w = chromosome[:-1]
        b = chromosome[-1]
        z = np.dot(w, x) + b
        z = np.clip(z, -15.0, 15.0)
        return float(1.0 / (1.0 + np.exp(-z)))
