"""
EXPERIMENT ONLY - NOT FOR REPORTING.

Tunes the GP that also receives the 8 derived features, so that variant can be
run at ITS OWN best settings. It is the same search as tune.py - same space,
same 20 candidate configurations (same search seed), same 3 repeats each, same
10,000-row samples, same tuning run seeds, scored on validation Brier - only the
terminal set differs. The winner is written to best_config_derived.json and
the results to gp_tuning_results_derived.csv. It never reads or writes
best_config.json or gp_tuning_results.csv, so the reported configuration is
untouched.

    python3 -m richard.tune_derived
"""
import csv
import json
import os

import numpy as np

from richard.derived import DERIVED_NAMES, load_split_derived, use_terminal_set
from richard.features import FEATURE_NAMES
from richard.fitness import make_dataset_fitness_fn
from richard.tune import (N_TRIALS, SAMPLE_ROWS, SEARCH_SPACE, TUNING_SEED_BASE,
                          evaluate_config, sample_config)

if __name__ == "__main__":
    print("EXPERIMENT ONLY: tuning the shared + derived variant (reported config untouched)\n")
    use_terminal_set(FEATURE_NAMES + DERIVED_NAMES)

    X_train, y_train = load_split_derived("cos700_dataset_train.npz", SAMPLE_ROWS)
    X_val, y_val = load_split_derived("cos700_dataset_val.npz", SAMPLE_ROWS)
    train_fn, val_fn = make_dataset_fitness_fn(X_train, y_train), make_dataset_fitness_fn(X_val, y_val)

    search_rng = np.random.default_rng(0)       # same seed as tune.py -> the same 20 candidate configurations
    results = []
    for trial in range(N_TRIALS):
        config = sample_config(search_rng)
        res = evaluate_config(config, train_fn, val_fn, TUNING_SEED_BASE + trial * 100)
        results.append(res)
        print(f"[{trial + 1}/{N_TRIALS}] N={config['population_size']:>3} G={config['generations']:>2} "
              f"k={config['tournament_size']} pc={config['crossover_rate']:.2f} pm={config['mutation_rate']:.2f} "
              f"dmax={config['max_depth']:>2} d0={config['init_max_depth']} -> "
              f"val {res['val_brier_mean']:.4f} +/- {res['val_brier_std']:.4f}")

    results.sort(key=lambda r: r["val_brier_mean"])
    with open("gp_tuning_results_derived.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = {k: results[0][k] for k in SEARCH_SPACE}
    path = os.path.join(os.path.dirname(__file__), "best_config_derived.json")
    with open(path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest configuration (val Brier {results[0]['val_brier_mean']:.4f}) written to {path}:\n{best}")