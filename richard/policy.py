"""
Turns a trained GP tree into a policy_fn for environment/play.py.

play_game uncovers the cell with the HIGHEST score. A tree trained on the
oracle's labels outputs a MINE probability (after the logistic function), so
the policy returns the NEGATED tree value: the highest score is then the
LOWEST estimated mine probability, exactly as for the GA's policy
(-P(mine)). The logistic function is monotone, so ranking by -T is the same
as ranking by -p_hat.
"""
import numpy as np

from richard.features import one_hot_features
from richard.node import Node


def make_gp_policy(tree: Node):
    def policy(raw_features: np.ndarray) -> np.ndarray:
        return -tree.evaluate(one_hot_features(raw_features))
    return policy