"""
Random-search tuning of the GP hyperparameters, scored on VALIDATION Brier.

20 random configurations x 3 seeds each, on 10,000-row random samples of the
training and validation splits (to keep the search affordable). The winner is
written to best_config.json, which best_config.py then loads automatically.
Re-run this whenever the inputs, the data or the sample sizes change.

    python3 -m richard.tune
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from richard.fitness import load_split, make_dataset_fitness_fn
from richard.runner import run

N_TRIALS = 20
N_REPEATS = 3
SAMPLE_ROWS = 10_000
TUNING_SEED_BASE = 10_000      # run seeds 1-10 are reserved for the final runs

SEARCH_SPACE = {
    "population_size": [30, 50, 80, 120],
    "generations": [15, 25, 40],
    "tournament_size": [2, 3, 5, 7],
    "crossover_rate": (0.60, 0.95),
    "mutation_rate": (0.05, 0.35),
    "max_depth": [8, 12, 16],
    "init_max_depth": [4, 6, 8],
}


def sample_config(rng: np.random.Generator) -> dict:
    config = {}
    for name, space in SEARCH_SPACE.items():
        if isinstance(space, tuple):
            config[name] = float(rng.uniform(*space))
        else:
            config[name] = space[int(rng.integers(0, len(space)))]
    return config


def evaluate_config(config: dict, train_fn, val_fn, base_seed: int) -> dict:
    val = [run(train_fn, val_fn, rng=np.random.default_rng(base_seed + r), verbose=False,
               **config).val_brier for r in range(N_REPEATS)]
    return {**config, "val_brier_mean": float(np.mean(val)), "val_brier_std": float(np.std(val, ddof=1))}


if __name__ == "__main__":
    X_train, y_train = load_split("cos700_dataset_train.npz", SAMPLE_ROWS)
    X_val, y_val = load_split("cos700_dataset_val.npz", SAMPLE_ROWS)
    train_fn, val_fn = make_dataset_fitness_fn(X_train, y_train), make_dataset_fitness_fn(X_val, y_val)

    search_rng = np.random.default_rng(0)
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
    with open("gp_tuning_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = {k: results[0][k] for k in SEARCH_SPACE}
    path = os.path.join(os.path.dirname(__file__), "best_config.json")
    with open(path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest configuration (val Brier {results[0]['val_brier_mean']:.4f}) written to {path}:\n{best}")