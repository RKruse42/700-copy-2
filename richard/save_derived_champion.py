"""
Re-train the GP + derived-features champion and SAVE the tree.

experiment_derived.py keeps every run's tree only in memory, so the champion
(run 8 = lowest validation Brier in your 10-run experiment) was lost when the
script ended. Training is fully seeded (run seed -> np.random.default_rng(seed),
fixed data sample), so re-training seed 8 with the same config reproduces it.
By default it uses the EXACT config printed in the experiment log (population
1802, 50 generations, ...), NOT best_config_derived.json, because that file can
differ between machines/copies of the repo. Pass --json-config to use the file.

    python -m richard.save_derived_champion            # seed 8 (the champion)
    python -m richard.save_derived_champion 3 8        # any seeds -> one file each
    python -m richard.save_derived_champion --json-config
    python -m richard.save_derived_champion --pop 1801     # override population size

Output: champion_derived_seed<N>.json. Check the printed numbers against the
experiment log: for seed 8 expect train 0.04566, val 0.04577, test 0.04666,
115 nodes, kept from generation 49.
"""
import sys
import time

import numpy as np

from richard.best_config import BEST_CONFIG
from richard.derived import (DERIVED_NAMES, load_derived_config, load_split_derived,
                             use_terminal_set)
from richard.features import FEATURE_NAMES
from richard.fitness import make_dataset_fitness_fn
from richard.multi_run import TEST_PATH, TRAIN_PATH, TRAIN_ROWS, VAL_PATH
from richard.runner import run
from richard.tree_io import save_tree

# copied verbatim from the "=== GP: 10 runs ===" line of the experiment log
LOGGED_CONFIG = {'population_size': 1802, 'generations': 50, 'tournament_size': 7,
                 'crossover_rate': 0.6837793050475333, 'mutation_rate': 0.31294526924321114,
                 'max_depth': 8, 'init_max_depth': 4}
# the champion's row in that log (run 8)
EXPECTED_SEED8 = {"train": 0.04566, "val": 0.04577, "test": 0.04666, "nodes": 115, "gen": 49}

if __name__ == "__main__":
    argv = sys.argv[1:]
    pop = None
    if "--pop" in argv:
        i = argv.index("--pop")
        pop = int(argv[i + 1])
        del argv[i:i + 2]
    seeds = [int(a) for a in argv if a.isdigit()] or [8]
    config = (load_derived_config() or BEST_CONFIG) if "--json-config" in sys.argv else LOGGED_CONFIG
    config = dict(config, **({"population_size": pop} if pop else {}))
    names = FEATURE_NAMES + DERIVED_NAMES
    use_terminal_set(names)                      # must happen before any tree is built
    print(f"config: {config}")

    X_tr, y_tr = load_split_derived(TRAIN_PATH, TRAIN_ROWS)
    X_va, y_va = load_split_derived(VAL_PATH, None)
    X_te, y_te = load_split_derived(TEST_PATH, None)
    train_fn, val_fn = make_dataset_fitness_fn(X_tr, y_tr), make_dataset_fitness_fn(X_va, y_va)
    test_fn = make_dataset_fitness_fn(X_te, y_te)

    for seed in seeds:
        t0 = time.perf_counter()
        res = run(train_fn, val_fn, rng=np.random.default_rng(seed), verbose=False, **config)
        test = test_fn(res.tree)
        out = f"champion_derived_seed{seed}.json"
        save_tree(out, res.tree, names, meta={
            "variant": "GP + Derived", "run_seed": seed, "config": config,
            "train_brier": res.train_brier, "val_brier": res.val_brier, "test_brier": test,
            "kept_generation": res.generation})
        print(f"seed {seed}: train {res.train_brier:.5f} | val {res.val_brier:.5f} | test {test:.5f} | "
              f"{res.tree.size()} nodes | depth {res.tree.depth()} | kept from gen {res.generation} | "
              f"{time.perf_counter() - t0:.0f}s -> saved {out}")
        if seed == 8:
            e = EXPECTED_SEED8
            ok = (round(res.train_brier, 5) == e["train"] and round(res.val_brier, 5) == e["val"]
                  and round(test, 5) == e["test"] and res.tree.size() == e["nodes"]
                  and res.generation == e["gen"])
            print("MATCHES the logged champion." if ok else
                  f"DOES NOT MATCH the logged champion {e} - do not use this file for the report.")