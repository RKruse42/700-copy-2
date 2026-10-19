"""
10 independent GP training runs (seeds 1-10), as for the neurocontroller.

Every run uses the same 50,000-row training sample (one data seed), the whole
validation split for model selection, and is scored ONCE on the whole test
split. All ten runs are returned and reported - the run is the unit of
analysis - not just the best one.

    python3 -m richard.multi_run
"""
from __future__ import annotations

import numpy as np

from richard.best_config import BEST_CONFIG, IS_TUNED
from richard.fitness import evaluate_on_dataset, load_split, make_dataset_fitness_fn
from richard.runner import run

RUN_SEEDS = list(range(1, 11))
TRAIN_ROWS = 50_000
TRAIN_PATH = "cos700_dataset_train.npz"
VAL_PATH = "cos700_dataset_val.npz"
TEST_PATH = "cos700_dataset_test.npz"


def train_gp_runs(run_seeds=RUN_SEEDS, train_rows=TRAIN_ROWS, val_rows=None, test_rows=None,
                  config=None, verbose=True) -> list[dict]:
    config = config or BEST_CONFIG
    if verbose and not IS_TUNED:
        print("WARNING: best_config.json not found - using UNTUNED placeholder hyperparameters. "
              "Run `python3 -m richard.tune` first.\n")

    X_train, y_train = load_split(TRAIN_PATH, train_rows)
    X_val, y_val = load_split(VAL_PATH, val_rows)
    X_test, y_test = load_split(TEST_PATH, test_rows)
    train_fn = make_dataset_fitness_fn(X_train, y_train)
    val_fn = make_dataset_fitness_fn(X_val, y_val)

    runs = []
    for seed in run_seeds:
        result = run(train_fn, val_fn, rng=np.random.default_rng(seed), verbose=False, **config)
        test_brier = evaluate_on_dataset(result.tree, X_test, y_test)   # scored once, never used to select
        runs.append({
            "seed": seed, "tree": result.tree,
            "train_brier": result.train_brier, "val_brier": result.val_brier,
            "test_brier": test_brier, "generation": result.generation,
            "nodes": result.tree.size(), "depth": result.tree.depth(),
        })
        if verbose:
            print(f"  run {seed:>2}: train {result.train_brier:.4f} | val {result.val_brier:.4f} | "
                  f"test {test_brier:.4f} | kept from generation {result.generation} | "
                  f"{result.tree.size()} nodes, depth {result.tree.depth()}")

    if verbose:
        for key in ("val_brier", "test_brier"):
            vals = [r[key] for r in runs]
            print(f"  {key}: mean {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}"
                  if len(vals) > 1 else f"  {key}: {vals[0]:.4f}")
    return runs


if __name__ == "__main__":
    print(f"Config: {BEST_CONFIG}\n")
    train_gp_runs()