"""
Baseline: an agent that selects cells entirely at random.

Section 4.3: "An agent that selects valid cells entirely at random. This
establishes an absolute performance floor to verify that autonomous
learning is actively occurring."
"""
from __future__ import annotations

import numpy as np


def make_random_policy(rng: np.random.Generator | None = None):
    """Returns a policy_fn(features) -> scores, same shape every other
    policy uses in environment.play.play_game. features is ignored
    entirely - the whole point of this baseline is that it looks at
    nothing and just picks uniformly among the legal cells.
    """
    rng = rng if rng is not None else np.random.default_rng()

    def policy(features: np.ndarray) -> np.ndarray:
        return rng.random(len(features))

    return policy