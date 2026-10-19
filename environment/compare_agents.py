"""
GP, random and handcrafted agents under the paper's evaluation protocol, so the
numbers sit next to the neurocontroller's.

- Board: intermediate, 16x16 with 40 mines (216 safe cells).
- The SAME 500 held-out boards for every agent: seeds 5000-5499.
- Metrics, as defined for the neurocontroller:
    safe moves : safe cells uncovered before the game ends; the fatal click is NOT counted
    revealed   : share of the 216 safe cells uncovered
    win rate   : share of games won
  ("moves incl. fatal click" is also printed, because that is the quantity in
  the GA's table.)
- GP: 10 independent training runs (seeds 1-10). Each run contributes ONE
  kept tree (lowest validation Brier), scored once on the test split and played
  on the 500 boards; the run is the unit of analysis, so GP is reported as
  mean +/- sd over the 10 runs.
- Baselines are deterministic or fixed-seed agents and are reported over the
  500 boards.

    python3 -m environment.compare_agents
"""
import csv

import numpy as np

from baselines.handcrafted_agent import handcrafted_policy
from baselines.random_agent import make_random_policy
from environment.evaluate import evaluate_policy
from richard.multi_run import train_gp_runs
from richard.policy import make_gp_policy

WIDTH, HEIGHT, MINES = 16, 16, 40
SAFE_CELLS = WIDTH * HEIGHT - MINES
EVAL_SEEDS = list(range(5000, 5500))


def game_metrics(rows):
    """Per-game arrays. evaluate_policy's `score` is the RAW count of safe cells
    uncovered and its `moves` includes the fatal click."""
    won = np.array([r["won"] for r in rows], dtype=bool)
    moves = np.array([r["moves"] for r in rows], dtype=float)
    return {
        "safe_moves": moves - (~won),            # fatal click not counted
        "moves_incl_fatal": moves,
        "revealed": np.array([r["score"] for r in rows], dtype=float) / SAFE_CELLS,
        "won": won.astype(float),
    }


def mean_sd(a):
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def fmt(a, pct=False, nd=2):
    m, s = mean_sd(a)
    return f"{m * 100:.1f}% ± {s * 100:.1f}%" if pct else f"{m:.{nd}f} ± {s:.{nd}f}"


if __name__ == "__main__":
    print("=== Training GP: independent runs on the oracle's labels ===")
    runs = train_gp_runs()

    print(f"\n=== Playing {len(EVAL_SEEDS)} held-out boards "
          f"({WIDTH}x{HEIGHT}, {MINES} mines, seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}) ===")
    for r in runs:
        r["metrics"] = game_metrics(
            evaluate_policy(make_gp_policy(r["tree"]), EVAL_SEEDS, WIDTH, HEIGHT, MINES))

    baselines = {
        "Random": game_metrics(evaluate_policy(
            make_random_policy(np.random.default_rng(0)), EVAL_SEEDS, WIDTH, HEIGHT, MINES)),
        "Handcrafted": game_metrics(evaluate_policy(
            handcrafted_policy, EVAL_SEEDS, WIDTH, HEIGHT, MINES)),
    }

    # ---- per-run GP table -------------------------------------------------
    print(f"\n### GP, per run (each run's kept tree, {len(EVAL_SEEDS)} boards)\n")
    print("| Run | Test Brier | Safe moves | Revealed | Win rate | Nodes | Depth | Kept from gen. |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in runs:
        m = r["metrics"]
        print(f"| {r['seed']} | {r['test_brier']:.4f} | {m['safe_moves'].mean():.2f} "
              f"| {m['revealed'].mean():.1%} | {m['won'].mean():.1%} "
              f"| {r['nodes']} | {r['depth']} | {r['generation']} |")

    # ---- summary table (paper format) ------------------------------------
    per_run = lambda key: np.array([r["metrics"][key].mean() for r in runs])
    print("\n### Summary\n")
    print(f"GP: mean ± sd over {len(runs)} runs (each run = mean over {len(EVAL_SEEDS)} boards). "
          f"Baselines: mean ± sd over the {len(EVAL_SEEDS)} boards.\n")
    print("| Agent | Test Brier | Safe moves | Revealed | Win rate | Moves incl. fatal click |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: |")
    print(f"| **GP** | {fmt([r['test_brier'] for r in runs], nd=4)} | {fmt(per_run('safe_moves'))} "
          f"| {fmt(per_run('revealed'), pct=True)} | {fmt(per_run('won'), pct=True)} "
          f"| {fmt(per_run('moves_incl_fatal'))} |")
    for name, m in baselines.items():
        print(f"| **{name}** | --- | {fmt(m['safe_moves'])} | {fmt(m['revealed'], pct=True)} "
              f"| {fmt(m['won'], pct=True)} | {fmt(m['moves_incl_fatal'])} |")

    # ---- per-run CSV for the cross-algorithm statistics -------------------
    with open("gp_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "run_seed", "test_brier", "val_brier", "safe_moves", "revealed",
                    "win_rate", "nodes", "depth", "generation"])
        for r in runs:
            m = r["metrics"]
            w.writerow(["GP", r["seed"], r["test_brier"], r["val_brier"], m["safe_moves"].mean(),
                        m["revealed"].mean(), m["won"].mean(), r["nodes"], r["depth"], r["generation"]])
    with open("baseline_games.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "board_seed", "safe_moves", "revealed", "won"])
        for name, m in baselines.items():
            for seed, sm, rv, wn in zip(EVAL_SEEDS, m["safe_moves"], m["revealed"], m["won"]):
                w.writerow([name, seed, sm, rv, int(wn)])
    print("\nSaved gp_runs.csv (one row per GP run) and baseline_games.csv (one row per baseline game).")