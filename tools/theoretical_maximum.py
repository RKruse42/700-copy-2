import os
import sys
from collections import defaultdict
import numpy as np
from scipy.optimize import root_scalar


def binary_entropy(pe):
    if pe <= 0 or pe >= 1:
        return 0.0
    return -pe * np.log2(pe) - (1 - pe) * np.log2(1 - pe)


def solve_fano(conditional_entropy):
    if conditional_entropy <= 0:
        return 0.0
    if conditional_entropy >= 1.0:
        return 0.5

    def objective_function(pe):
        return binary_entropy(pe) - conditional_entropy

    result = root_scalar(objective_function, bracket=[0.0, 0.5], method='brentq')

    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge!")


def get_conditional_entropy(window_list, mine_list, window_features):
    total_samples = len(mine_list)
    if total_samples == 0:
        return 0.0

    if not window_features:
        prob_mine = np.mean(mine_list)
        if prob_mine in (0.0, 1.0):
            return 0.0
        return binary_entropy(prob_mine)

    pattern_counts = defaultdict(lambda: [0, 0])

    for i in range(total_samples):
        window = window_list[i]
        key = tuple(int(window[r, c]) for (r, c) in window_features)

        pattern_counts[key][0] += 1
        if mine_list[i]:
            pattern_counts[key][1] += 1

    conditional_entropy = 0.0

    for total, mines in pattern_counts.values():
        p_x = total / total_samples
        p_mine_given_x = mines / total

        if 0.0 < p_mine_given_x < 1.0:
            h_given_x = binary_entropy(p_mine_given_x)
            conditional_entropy += p_x * h_given_x

    return conditional_entropy


def get_theoretical_maximum(window_list, mine_list, window_features):
    prob_mine = np.mean(mine_list)
    if prob_mine in (0.0, 1.0):
        prior_entropy = 0.0
    else:
        prior_entropy = binary_entropy(prob_mine)

    cond_entropy = get_conditional_entropy(window_list, mine_list, window_features)
    mi = max(0.0, prior_entropy - cond_entropy)
    min_error_rate = solve_fano(cond_entropy)

    return mi, cond_entropy, min_error_rate


if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "cos700_dataset.npz"
    with np.load(dataset_file) as data:
        windows = data['windows']
        is_mine = data['is_mine']

    features_subset = [(3, 4), (5, 4), (4, 3), (4, 5)]

    mi, h_cond, min_error_rate = get_theoretical_maximum(windows, is_mine, features_subset)

    print(f"Conditional Entropy H(Mine | Features): {h_cond:.5f}")
    print(f"Joint Mutual Information I(Features; Mine): {mi:.5f}")
    print(f"Theoretical Minimum Error Rate (P_e): {min_error_rate * 100:.2f}%")
