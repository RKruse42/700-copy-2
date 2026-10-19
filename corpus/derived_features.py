"""
Derived (engineered) features computed from a raw 5x5 window around a
candidate cell, on top of the 25 raw cell values.

These specifically hand GP the same LOCAL reasoning primitives the
handcrafted baseline (baselines/handcrafted_agent.py) computes
procedurally - neighbour counts, forced-mine detection, a risk estimate -
so a GP tree has to learn how to COMBINE and WEIGH this evidence, rather
than being forced to first reconstruct "how many of my neighbours are
hidden" from scratch out of raw scalars via tree search alone (which is a
much harder structure to stumble into via random mutation/crossover than
it sounds).

This is GP-specific, not shared harness infrastructure: environment/play.py
and the baseline agents keep operating on the raw 25-value window
unmodified (that's the common "identical feature space" every policy is
fairly judged on), and it's up to each individual policy to do its own
feature engineering on top of that raw input if it wants to - exactly how
handcrafted_agent.py already computes its own Rule 1/2 logic internally.
gp/policy.py is what applies this expansion for GP specifically, both at
training time (gp/fitness.py) and at live-play time, so features never
drift between the two.
"""
from __future__ import annotations

import numpy as np

HIDDEN_VALUE = 9
OUT_OF_BOUNDS_VALUE = 10
WINDOW_SIDE = 5
CENTRE = 2

_NEIGHBOUR_OFFSETS = [
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)
]

RAW_FEATURE_COUNT = WINDOW_SIDE * WINDOW_SIDE  # 25

DERIVED_FEATURE_NAMES = [
    "hidden_neighbour_count",
    "revealed_neighbour_count",
    "out_of_bounds_neighbour_count",
    "forced_mine",
    "risk_estimate",
    "avg_neighbour_value",
    "min_revealed_neighbour_value",
    "max_revealed_neighbour_value",
]
FEATURE_COUNT = RAW_FEATURE_COUNT + len(DERIVED_FEATURE_NAMES)  # 33


def _count_hidden_neighbours(window: np.ndarray, r: int, c: int) -> int:
    count = 0
    for dr, dc in _NEIGHBOUR_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < WINDOW_SIDE and 0 <= cc < WINDOW_SIDE and window[rr, cc] == HIDDEN_VALUE:
            count += 1
    return count


def _derive_one(window: np.ndarray) -> np.ndarray:
    """window: (5, 5) raw values. Returns the 8 derived scalars listed in
    DERIVED_FEATURE_NAMES, computed about the window's centre cell.
    """
    neighbour_values = []
    for dr, dc in _NEIGHBOUR_OFFSETS:
        r, c = CENTRE + dr, CENTRE + dc
        if 0 <= r < WINDOW_SIDE and 0 <= c < WINDOW_SIDE:
            neighbour_values.append(window[r, c])
    neighbour_values = np.array(neighbour_values, dtype=np.float64)

    hidden_count = float(np.sum(neighbour_values == HIDDEN_VALUE))
    revealed_mask = (neighbour_values >= 0) & (neighbour_values <= 8)
    revealed_count = float(np.sum(revealed_mask))
    oob_count = float(np.sum(neighbour_values == OUT_OF_BOUNDS_VALUE))

    # forced-mine + risk-estimate: the SAME single-constraint logic
    # baselines/handcrafted_agent.py uses, evaluated for the candidate
    # (window centre) - see that file's docstring for the reasoning
    forced_mine = 0.0
    risks = []
    for dr, dc in _NEIGHBOUR_OFFSETS:
        r, c = CENTRE + dr, CENTRE + dc
        if not (0 <= r < WINDOW_SIDE and 0 <= c < WINDOW_SIDE):
            continue
        value = window[r, c]
        if 0 <= value <= 8:
            h = _count_hidden_neighbours(window, r, c)
            if h > 0:
                if h == value:
                    forced_mine = 1.0
                else:
                    risks.append(value / h)
    # 0.5 is a neutral "no local evidence either way" prior, not a claim
    risk_estimate = max(risks) if risks else 0.5

    if revealed_count > 0:
        revealed_values = neighbour_values[revealed_mask]
        min_revealed = float(np.min(revealed_values))
        max_revealed = float(np.max(revealed_values))
    else:
        min_revealed = -1.0  # sentinel: nothing revealed to compare
        max_revealed = -1.0

    valid_values = neighbour_values[neighbour_values != OUT_OF_BOUNDS_VALUE]
    avg_value = float(np.mean(valid_values)) if len(valid_values) else 0.0

    return np.array([
        hidden_count, revealed_count, oob_count,
        forced_mine, risk_estimate,
        avg_value, min_revealed, max_revealed,
    ], dtype=np.float64)


def build_features(raw_windows: np.ndarray) -> np.ndarray:
    """raw_windows: (N, 25) flattened 5x5 windows - the standard format
    every policy_fn receives from environment/play.py. Returns
    (N, FEATURE_COUNT): the 25 raw values followed by the 8 derived
    scalars, per row.
    """
    n = raw_windows.shape[0]
    windows_5x5 = raw_windows.reshape(n, WINDOW_SIDE, WINDOW_SIDE)
    derived = np.stack([_derive_one(windows_5x5[i]) for i in range(n)])
    return np.concatenate([raw_windows, derived], axis=1)