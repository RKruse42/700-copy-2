"""
EXPERIMENT ONLY - NOT FOR REPORTING.

Does giving the GP 8 derived neighbourhood features, on top of the shared 264
one-hot indicators, change what it learns? Output is laid out like the GA's
results so the two can be compared directly:

  * training: Brier (train / validation / test) per run, and the champion's
    convergence table (the GA's "Brier score progression" table)
  * for BOTH boards - beginner 8x8/10 mines and intermediate 16x16/40 mines -
    on the SAME 500 held-out boards (seeds 5000-5499) for every agent:
      - the GA-style table: Win Rate, Board Cleared (%), Cells Cleared, Moves
        Survived (moves INCLUDING the fatal click, as in the GA table), plus
        Safe Moves (fatal click not counted, as in the paper's neurocontroller
        table)
      - the paper-style table: GP as mean +/- sd over runs
      - two-tailed Mann-Whitney U tests, every agent vs every other, on Cells
        Cleared and on Moves Survived, with a Bonferroni correction, plus the
        targeted comparisons the GA reports (vs Random, vs Handcrafted)
  * CSVs: experiment_derived_runs.csv (one row per run and board) and
    experiment_derived_games_<board>.csv (one row per game)

The GP is trained ONCE on the dataset (16x16 windows) and the same trees play
both boards, exactly as the GA champion is deployed on both.

The GA-style tables use the CHAMPION of each variant: the run with the lowest
VALIDATION Brier (the same rule the GA uses to pick its champion). The paper
-style table and the CSV use all runs.

Variants:  shared   = the 264 shared indicators only        (the reported GP)
           derived  = the 264 indicators + 8 derived features (the experiment)
HYPERPARAMETERS: each variant uses its own tuned config by default
(best_config.json / best_config_derived.json); --same-config forces both onto
the shared one. --derived-only skips the shared variant.

Nothing here touches the reported pipeline; results never go to gp_runs.csv.

    python3 -m richard.experiment_derived [n_runs] [--same-config] [--derived-only]
    (n_runs defaults to 5; 10 matches the reported protocol)
"""
import csv
import sys
import time

import numpy as np

from baselines.handcrafted_agent import handcrafted_policy
from baselines.random_agent import make_random_policy
from environment.compare_agents import EVAL_SEEDS, fmt
from environment.evaluate import (evaluate_policy, mann_whitney_compare, pairwise_mann_whitney,
                                  print_pairwise_table, save_results_csv)
from richard.best_config import BEST_CONFIG, IS_TUNED
from richard.derived import (DERIVED_NAMES, load_derived_config, load_split_derived,
                             make_derived_policy, use_terminal_set)
from richard.features import FEATURE_NAMES
from richard.fitness import load_split, make_dataset_fitness_fn
from richard.multi_run import RUN_SEEDS, TEST_PATH, TRAIN_PATH, TRAIN_ROWS, VAL_PATH
from richard.policy import make_gp_policy
from richard.runner import run

VAL_ROWS = None     # None = the whole split (the protocol); set smaller only for quick tests
TEST_ROWS = None

BOARDS = [("Beginner", 8, 8, 10), ("Intermediate", 16, 16, 40)]     # same order as the GA tables


# ---------------------------------------------------------------- helpers
def arrays(rows, safe_cells):
    """Per-game arrays. evaluate_policy's `score` is the RAW count of safe cells
    uncovered and `moves` includes the fatal click."""
    won = np.array([r["won"] for r in rows], dtype=bool)
    moves = np.array([r["moves"] for r in rows], dtype=float)
    cells = np.array([r["score"] for r in rows], dtype=float)
    return {"won": won.astype(float), "moves": moves, "safe_moves": moves - (~won),
            "cells": cells, "cleared": cells / safe_cells}


def fmt_p(p):
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"


def brier_split(tree, X, y):
    p = 1.0 / (1.0 + np.exp(-np.clip(tree.evaluate(X), -60, 60)))
    amb = (y > 1e-9) & (y < 1 - 1e-9)
    return {"all": float(np.mean((p - y) ** 2)),
            "determined": float(np.mean((p[~amb] - y[~amb]) ** 2)),
            "ambiguous": float(np.mean((p[amb] - y[amb]) ** 2)),
            "constant_on_ambiguous": float(np.var(y[amb])),
            "ambiguous_share": float(amb.mean())}


# ---------------------------------------------------------------- training + play
def run_variant(label, spec, config, n_runs):
    print(f"\n=== {label}: {n_runs} runs ===")
    print(f"  config: {config}")
    use_terminal_set(spec["names"])
    X_tr, y_tr = spec["load"](TRAIN_PATH, TRAIN_ROWS)
    X_va, y_va = spec["load"](VAL_PATH, VAL_ROWS)
    X_te, y_te = spec["load"](TEST_PATH, TEST_ROWS)
    train_fn, val_fn = make_dataset_fitness_fn(X_tr, y_tr), make_dataset_fitness_fn(X_va, y_va)

    runs = []
    for seed in RUN_SEEDS[:n_runs]:
        t0 = time.perf_counter()
        res = run(train_fn, val_fn, rng=np.random.default_rng(seed), verbose=False, **config)
        seconds = time.perf_counter() - t0
        b = brier_split(res.tree, X_te, y_te)
        policy = spec["policy"](res.tree)
        rows = {name: evaluate_policy(policy, EVAL_SEEDS, w, h, m) for name, w, h, m in BOARDS}
        r = {"variant": label, "seed": seed, "tree": res.tree, "train": res.train_brier,
             "val": res.val_brier, "test": b, "generation": res.generation, "history": res.history,
             "seconds": seconds, "rows": rows, "nodes": res.tree.size(), "depth": res.tree.depth(),
             "arr": {name: arrays(rows[name], w * h - m) for name, w, h, m in BOARDS}}
        runs.append(r)
        s = r["arr"]["Intermediate"]
        print(f"  run {seed:>2}: val {res.val_brier:.4f} | test {b['all']:.4f} (determined {b['determined']:.4f}, "
              f"ambiguous {b['ambiguous']:.4f}) | intermediate safe moves {s['safe_moves'].mean():.2f} | "
              f"{res.tree.size()} nodes | {seconds:.0f}s")
    return runs, b["constant_on_ambiguous"], b["ambiguous_share"]


# ---------------------------------------------------------------- printing
def print_convergence(label, champion):
    hist, kept = champion["history"], champion["generation"]
    last = hist[-1][0]
    gens = sorted({0, last // 4, last // 2, (3 * last) // 4, last, kept})
    print(f"\n#### Brier progression, {label} champion (run {champion['seed']}, lowest validation Brier)\n")
    print("| Evaluation stage | Training Brier | Validation Brier | Test Brier |")
    print("| :--- | ---: | ---: | ---: |")
    for g in gens:
        _, tr, va = hist[g]
        tag = " (champion)" if g == kept else (" (initial)" if g == 0 else (" (final)" if g == last else ""))
        te = f"{champion['test']['all']:.5f}" if g == kept else "---"
        print(f"| Generation {g}{tag} | {tr:.5f} | {va:.5f} | {te} |")


def print_ga_table(title, agents):
    print(f"\n{title}\n")
    print("| Agent | Win Rate | Board Cleared (%) | Cells Cleared | Moves Survived | Safe Moves |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for name, a in agents.items():
        print(f"| **{name}** | {a['won'].mean():.1%} | {fmt(a['cleared'], pct=True)} | {fmt(a['cells'])} "
              f"| {fmt(a['moves'])} | {fmt(a['safe_moves'])} |")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    args = sys.argv[1:]
    same_config, derived_only = "--same-config" in args, "--derived-only" in args
    numeric = [a for a in args if a.isdigit()]
    n_runs = int(numeric[0]) if numeric else 5

    print("EXPERIMENT ONLY - not for reporting.")
    if not IS_TUNED and not derived_only:
        print("WARNING: the shared variant is using the untuned placeholder config - run `python3 -m richard.tune` first.")
    derived_config = None if same_config else load_derived_config()
    if same_config:
        print("--same-config: BOTH variants use the shared config.")
    elif derived_config is None:
        print("NOTE: best_config_derived.json not found - the derived variant falls back to the SHARED config "
              "(run `python3 -m richard.tune_derived` to tune it separately).")
    derived_config = derived_config or BEST_CONFIG

    variants = {}
    if not derived_only:
        variants["GP Shared"] = (dict(names=FEATURE_NAMES, load=load_split, policy=make_gp_policy), BEST_CONFIG)
    variants["GP + Derived"] = (dict(names=FEATURE_NAMES + DERIVED_NAMES, load=load_split_derived,
                                     policy=make_derived_policy), derived_config)

    results, const, share = {}, None, None
    for label, (spec, config) in variants.items():
        results[label], const, share = run_variant(label, spec, config, n_runs)

    # ---- baselines, on both boards, same seeds ----------------------------------
    baseline_rows = {"Random Baseline": {}, "Handcrafted Baseline": {}}
    for name, w, h, m in BOARDS:
        baseline_rows["Random Baseline"][name] = evaluate_policy(
            make_random_policy(np.random.default_rng(0)), EVAL_SEEDS, w, h, m)
        baseline_rows["Handcrafted Baseline"][name] = evaluate_policy(handcrafted_policy, EVAL_SEEDS, w, h, m)

    champions = {label: min(runs, key=lambda r: r["val"]) for label, runs in results.items()}

    # ================= training summary =================
    print("\n" + "=" * 78 + f"\n## Training (mean ± sd over {n_runs} runs; test split is {share:.1%} ambiguous rows)\n")
    print("| Variant | Train Brier | Validation Brier | Test Brier | Test: determined rows | Test: ambiguous rows | Kept from gen. | Nodes | Depth | Seconds/run |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, runs in results.items():
        col = lambda k: [r["test"][k] for r in runs]
        print(f"| **{label}** | {fmt([r['train'] for r in runs], nd=4)} | {fmt([r['val'] for r in runs], nd=4)} "
              f"| {fmt(col('all'), nd=4)} | {fmt(col('determined'), nd=4)} | {fmt(col('ambiguous'), nd=4)} "
              f"| {fmt([r['generation'] for r in runs], nd=1)} | {fmt([r['nodes'] for r in runs], nd=0)} "
              f"| {fmt([r['depth'] for r in runs], nd=1)} | {fmt([r['seconds'] for r in runs], nd=0)} |")
    print(f"\nBest constant guess on the ambiguous test rows: Brier {const:.4f}  (a tree worse than this is "
          f"worse than ignoring the board on the cells where a guess is needed).")
    for label, champ in champions.items():
        print_convergence(label, champ)

    # ================= per board =================
    csv_runs = []
    for name, w, h, m in BOARDS:
        safe = w * h - m
        print("\n" + "=" * 78 + f"\n## {name} board ({w} × {h}, {m} mines, {safe} safe cells) - "
                                f"{len(EVAL_SEEDS)} evaluation seeds ({EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]})")

        # per-game rows by agent (champions for the GP variants)
        per_game = {label: champ["rows"][name] for label, champ in champions.items()}
        per_game = {"Random Baseline": baseline_rows["Random Baseline"][name],
                    "Handcrafted Baseline": baseline_rows["Handcrafted Baseline"][name], **per_game}
        arr = {k: arrays(v, safe) for k, v in per_game.items()}

        print_ga_table(f"### GA-style table (GP rows = champion run, mean ± sd over the {len(EVAL_SEEDS)} games)",
                       {**{k: arr[k] for k in ("Random Baseline", "Handcrafted Baseline")},
                        **{f"{k} (champion)": arr[k] for k in champions}})

        print_ga_table(f"### Paper-style table (GP rows = mean ± sd over {n_runs} runs, each run = mean over the games)",
                       {**{k: arr[k] for k in ("Random Baseline", "Handcrafted Baseline")},
                        **{k: {key: np.array([r["arr"][name][key].mean() for r in runs])
                               for key in ("won", "cleared", "cells", "moves", "safe_moves")}
                           for k, runs in results.items()}})

        # ---- statistics --------------------------------------------------------
        print("\n### Pairwise Mann–Whitney U tests, two-tailed, all agents (per-game results; GP = champion)")
        for metric, label in (("score", "Cells Cleared"), ("moves", "Moves Survived")):
            print(f"\n{label}:")
            print_pairwise_table(pairwise_mann_whitney(per_game, metric=metric))

        print("\n### Targeted comparisons (as reported for the GA)")
        for gp in champions:
            for ref in ("Random Baseline", "Handcrafted Baseline"):
                U, p = (lambda r: (float(r.statistic), float(r.pvalue)))(
                    mann_whitney_compare(per_game[gp], per_game[ref], "moves"))
                Us, ps = (lambda r: (float(r.statistic), float(r.pvalue)))(
                    mann_whitney_compare(per_game[gp], per_game[ref], "score"))
                ratio = arr[gp]["moves"].mean() / arr[ref]["moves"].mean()
                d_moves = 100 * (ratio - 1)
                d_cells = arr[gp]["cells"].mean() - arr[ref]["cells"].mean()
                print(f"- {gp} vs {ref}: moves U={U:.1f}, {fmt_p(p)} | cells U={Us:.1f}, {fmt_p(ps)} | "
                      f"survives {ratio:.2f}× as long ({d_moves:+.1f}%), {d_cells:+.2f} cells per game")
        if len(champions) == 2:
            U, p = (lambda r: (float(r.statistic), float(r.pvalue)))(
                mann_whitney_compare(per_game["GP + Derived"], per_game["GP Shared"], "moves"))
            Us, ps = (lambda r: (float(r.statistic), float(r.pvalue)))(
                mann_whitney_compare(per_game["GP + Derived"], per_game["GP Shared"], "score"))
            print(f"- GP + Derived vs GP Shared: moves U={U:.1f}, {fmt_p(p)} | cells U={Us:.1f}, {fmt_p(ps)}")

        save_results_csv(f"experiment_derived_games_{name.lower()}.csv", per_game)
        for label, runs in results.items():
            for r in runs:
                a = r["arr"][name]
                csv_runs.append([label, r["seed"], name, r["train"], r["val"], r["test"]["all"],
                                 r["test"]["determined"], r["test"]["ambiguous"], a["won"].mean(),
                                 a["cleared"].mean(), a["cells"].mean(), a["moves"].mean(),
                                 a["safe_moves"].mean(), r["nodes"], r["depth"], r["generation"]])

    with open("experiment_derived_runs.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["variant", "run_seed", "board", "train_brier", "val_brier", "test_brier",
                     "test_brier_determined", "test_brier_ambiguous", "win_rate", "board_cleared",
                     "cells_cleared", "moves_survived", "safe_moves", "nodes", "depth", "kept_generation"])
        w_.writerows(csv_runs)
    print("\nSaved experiment_derived_runs.csv and experiment_derived_games_<board>.csv "
          "(never overwrites gp_runs.csv).")