"""
Load a saved GP + derived champion and play it on the 8x8 (10 mines) and
16x16 (40 mines) boards, on the same seeds the experiment used (5000-5499),
writing per-game CSVs in the same long format as experiment_derived_games_*.csv
(agent, seed, won, score, moves) so they drop straight into the statistical tests.

    python -m richard.play_derived_champion [tree.json] [--seeds START END]

    default tree: champion_derived_seed8.json, default seeds: 5000 5499 (inclusive)
Output: champion_derived_games_beginner.csv / champion_derived_games_intermediate.csv
"""
import sys

import numpy as np

from environment.evaluate import evaluate_policy, save_results_csv
from richard.derived import make_derived_policy, use_terminal_set
from richard.tree_io import load_tree

BOARDS = [("beginner", 8, 8, 10), ("intermediate", 16, 16, 40)]
AGENT = "GP Derived (champion)"


def load_champion_policy(path: str):
    """Returns (tree, policy_fn). The terminal set is switched first so the tree
    prints with the right feature names; evaluation itself only needs indices."""
    tree, payload = load_tree(path)
    use_terminal_set(payload["feature_names"])
    return tree, make_derived_policy(tree)


if __name__ == "__main__":
    args = sys.argv[1:]
    seeds = list(range(5000, 5500))
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = list(range(int(args[i + 1]), int(args[i + 2]) + 1))
        del args[i:i + 3]
    path = args[0] if args else "champion_derived_seed8.json"

    tree, policy = load_champion_policy(path)
    print(f"loaded {path}: {tree.size()} nodes, depth {tree.depth()}")
    for name, w, h, m in BOARDS:
        rows = evaluate_policy(policy, seeds, w, h, m)
        safe = w * h - m
        won = np.array([r["won"] for r in rows], dtype=float)
        cells = np.array([r["score"] for r in rows], dtype=float)
        moves = np.array([r["moves"] for r in rows], dtype=float)
        print(f"{name} {w}x{h}/{m}: win {won.mean():.1%} | cleared {(cells / safe).mean():.1%} | "
              f"cells {cells.mean():.2f} ± {cells.std(ddof=1):.2f} | moves {moves.mean():.2f} ± {moves.std(ddof=1):.2f}")
        save_results_csv(f"champion_derived_games_{name}.csv", {AGENT: rows})
    print("saved champion_derived_games_<board>.csv")