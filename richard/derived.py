"""
EXPERIMENT ONLY - NOT FOR REPORTING.

The 8 derived neighbourhood features the experiment adds on top of the shared
264 indicators, plus helpers shared by experiment_derived.py and
tune_derived.py. Nothing in the reported pipeline imports this module.

Derived features (computed about the target cell from its 5x5 window):
    hidden_neighbour_count, revealed_neighbour_count, out_of_bounds_neighbour_count,
    forced_mine, risk_estimate, avg_neighbour_value,
    min_revealed_neighbour_value, max_revealed_neighbour_value
forced_mine and risk_estimate are the handcrafted agent's two rules handed to
the tree as ready-made inputs.
"""
import json
import os

import numpy as np

import richard.terminals as terminals
from richard.features import RAW_FEATURE_COUNT, one_hot_features
from richard.fitness import DATA_SEED, sample_rows

DERIVED_NAMES = [
    "hidden_neighbour_count", "revealed_neighbour_count", "out_of_bounds_neighbour_count",
    "forced_mine", "risk_estimate", "avg_neighbour_value",
    "min_revealed_neighbour_value", "max_revealed_neighbour_value",
]

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "best_config_derived.json")


def load_derived_config():
    """The derived variant's OWN tuned config (written by tune_derived.py), or None."""
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return None


def use_terminal_set(names: list[str]) -> None:
    """Switch the GP's terminal set for this process only."""
    terminals.FEATURE_NAMES = names
    terminals.FEATURE_COUNT = len(names)


def derived_features(raw: np.ndarray) -> np.ndarray:
    """raw: (n, 25) flattened 5x5 windows -> (n, 8) derived scalars about the centre cell."""
    n = len(raw)
    w = np.asarray(raw, dtype=np.float64).reshape(n, 5, 5)
    ring = [(a, b) for a in (1, 2, 3) for b in (1, 2, 3) if (a, b) != (2, 2)]
    N = np.stack([w[:, a, b] for a, b in ring], axis=1)               # the 8 neighbours

    hidden = (N == 9).sum(1).astype(float)
    revealed_mask = (N >= 0) & (N <= 8)
    revealed = revealed_mask.sum(1).astype(float)
    oob = (N == 10).sum(1).astype(float)

    H = (w == 9).astype(np.int64)
    forced = np.zeros(n, dtype=bool)
    risk = np.full(n, -np.inf)
    for k, (a, b) in enumerate(ring):
        v = N[:, k]
        h = H[:, a - 1:a + 2, b - 1:b + 2].sum(axis=(1, 2))           # covered cells around that neighbour
        valid = revealed_mask[:, k] & (h > 0)
        forced |= valid & (h == v)
        risk = np.maximum(risk, np.where(valid & (h != v), v / np.maximum(h, 1), -np.inf))
    risk = np.where(np.isfinite(risk), risk, 0.5)                      # 0.5 = no local evidence

    lo = np.where(revealed_mask, N, np.inf).min(1)
    hi = np.where(revealed_mask, N, -np.inf).max(1)
    lo, hi = np.where(np.isfinite(lo), lo, -1.0), np.where(np.isfinite(hi), hi, -1.0)

    in_board = N != 10
    count = in_board.sum(1)
    avg = np.where(count > 0, (N * in_board).sum(1) / np.maximum(count, 1), 0.0)

    return np.stack([hidden, revealed, oob, forced.astype(float), risk, avg, lo, hi], axis=1)


def derived_inputs(raw: np.ndarray) -> np.ndarray:
    return np.asfortranarray(np.concatenate([one_hot_features(raw), derived_features(raw)], axis=1))


def load_split_derived(path: str, n_rows=None):
    with np.load(path) as data:
        windows, probs = data["windows"], data["csp_probs"]
        idx = sample_rows(len(windows), n_rows, DATA_SEED)
        raw = windows[idx][:, 2:7, 2:7].reshape(len(idx), RAW_FEATURE_COUNT)
        y = np.ascontiguousarray(probs[idx], dtype=np.float64)
    return derived_inputs(raw), y


def make_derived_policy(tree):
    """-T so that the harness' argmax uncovers the lowest estimated mine probability."""
    return lambda raw: -tree.evaluate(derived_inputs(raw))