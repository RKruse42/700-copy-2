"""
Dataset fitness for GP: the Brier score between the tree's predicted mine
probability and the CSP oracle's probability - the same primary fitness as
the GA and the neurocontroller.

fitness_fn(tree) -> float, LOWER IS BETTER. The data are loaded and encoded
once, outside the evolution loop, and captured in a closure.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from richard.features import RAW_FEATURE_COUNT, one_hot_features
from richard.node import Node
from stats.brier import brier_np

FitnessFn = Callable[[Node], float]

# Seed of the random draw of training rows. Every run uses the SAME rows
# (the run seed only controls the evolution). Set this - and check that
# sample_rows() draws rows the same way - to match the neurocontroller's
# data sample if you want both agents to fit literally the same rows.
DATA_SEED = 0


def sample_rows(n_total: int, n_rows: int | None, seed: int = DATA_SEED) -> np.ndarray:
    """Row indices: all rows if n_rows is None/too large, else a random draw
    WITHOUT replacement (sorted). Rows are grouped by game in the files, so a
    contiguous slice would cover only a few boards.
    """
    if n_rows is None or n_rows >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_rows, replace=False))


def load_split(path: str, n_rows: int | None = None, seed: int = DATA_SEED):
    """Returns (X, y): X is (n, 264) one-hot indicators of the 5x5 window with
    the target removed, y the oracle probabilities."""
    with np.load(path) as data:
        windows = data["windows"]
        probs = data["csp_probs"]
        idx = sample_rows(len(windows), n_rows, seed)
        crop = windows[idx][:, 2:7, 2:7].reshape(len(idx), RAW_FEATURE_COUNT)
        y = np.ascontiguousarray(probs[idx], dtype=np.float64)
    return one_hot_features(crop), y


def evaluate_on_dataset(tree: Node, X: np.ndarray, y: np.ndarray) -> float:
    raw = tree.evaluate(X)
    # clipped before the logistic function so huge values from early random
    # trees cannot overflow exp()
    p = 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
    return float(brier_np(y, p))


def make_dataset_fitness_fn(X: np.ndarray, y: np.ndarray) -> FitnessFn:
    def fitness_fn(tree: Node) -> float:
        return evaluate_on_dataset(tree, X, y)
    return fitness_fn