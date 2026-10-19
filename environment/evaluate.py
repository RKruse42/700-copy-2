"""
The ONE evaluation procedure every agent - GP (either training mode), GA,
the Neurocontroller, and both baselines - is meant to be run through before
its numbers go anywhere near the report or a Mann-Whitney test.

This is deliberately separate from any algorithm's fitness function.
Fitness (Brier score, game_reward, whatever a GA/PSO uses internally) is
allowed to be shaped however helps that algorithm learn. Evaluation is not:
it applies zero shaping, computes exactly the two numbers Section 4.4 of
the proposal asks for (score, length of play), the same way, for everyone,
on the same fixed seed set - which is what makes results across completely
different algorithms actually comparable.

Metrics, matching the proposal exactly:
- "score": cells_revealed - the raw count of safe cells the agent cleared
  before the game ended (0 if it dies immediately, the full safe_cell_count
  on a win). No death penalty, no normalisation, no sigmoid - a plain
  outcome count. Since board width/height/mines are held fixed across one
  comparison run, safe_cell_count is constant across seeds too, so raw
  counts are already directly comparable within that run.
- "moves": moves_made - "length of play", exactly as returned by
  environment.play.play_game, unmodified.
"""
from __future__ import annotations

import csv
from typing import Sequence

import numpy as np
from scipy import stats

from environment.play import PolicyFn, play_game


def evaluate_policy(policy_fn: PolicyFn, seeds: Sequence[int], width: int, height: int,
                     mines: int) -> list[dict]:
    """Plays policy_fn through every seed and returns one raw, unshaped
    result row per game: {seed, won, score, moves}.
    """
    rows = []
    for seed in seeds:
        result = play_game(policy_fn, width, height, mines, seed)
        rows.append({
            "seed": seed,
            "won": result["won"],
            "score": result["cells_revealed"],
            "moves": result["moves_made"],
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    """Mean/std for score and moves, plus win rate - Section 4.4's "mean
    performance across all runs, and the standard deviation to track
    performance variance."
    """
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    moves = np.array([r["moves"] for r in rows], dtype=np.float64)
    wins = np.array([r["won"] for r in rows], dtype=np.float64)
    return {
        "n_runs": len(rows),
        "win_rate": float(wins.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)) if len(rows) > 1 else 0.0,
        "moves_mean": float(moves.mean()),
        "moves_std": float(moves.std(ddof=1)) if len(rows) > 1 else 0.0,
    }


def run_comparison(policies: dict[str, PolicyFn], seeds: Sequence[int], width: int,
                    height: int, mines: int) -> dict[str, list[dict]]:
    """Evaluates every named policy (any mix of GP trees, GA individuals,
    the Neurocontroller, baseline agents...) on the exact same seed list,
    so every entry in the returned dict is directly comparable to every
    other. This is the function everything else - GA, PSO, baselines -
    should call when it's time to produce report-ready numbers.
    """
    return {name: evaluate_policy(policy_fn, seeds, width, height, mines)
            for name, policy_fn in policies.items()}


def mann_whitney_compare(rows_a: list[dict], rows_b: list[dict], metric: str = "score"):
    """Section 4.4: "a Mann-Whitney U-test is utilised to establish
    statistical significance" between two independent samples. Returns
    scipy's (statistic, p-value) for the given metric ("score" or "moves").
    """
    a = [r[metric] for r in rows_a]
    b = [r[metric] for r in rows_b]
    return stats.mannwhitneyu(a, b, alternative="two-sided")


def pairwise_mann_whitney(results_by_name: dict[str, list[dict]], metric: str = "score",
                           alpha: float = 0.05) -> list[dict]:
    """Runs mann_whitney_compare on EVERY pair of agents, not just one
    agent against the rest - with n agents that's n*(n-1)/2 comparisons.

    Running many pairwise tests inflates the chance of a false positive
    (with 6 tests at alpha=0.05, there's a meaningfully higher-than-5%
    chance at least one comes back "significant" by pure luck), so this
    also applies a Bonferroni correction: the significance threshold used
    for "significant" is alpha divided by the number of comparisons,
    rather than alpha itself. The raw p-value is still returned unmodified
    too, so you can report either the corrected or uncorrected view.
    """
    names = list(results_by_name.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    bonferroni_alpha = alpha / len(pairs) if pairs else alpha

    rows = []
    for name_a, name_b in pairs:
        stat, p = mann_whitney_compare(results_by_name[name_a], results_by_name[name_b], metric)
        rows.append({
            "agent_a": name_a,
            "agent_b": name_b,
            "metric": metric,
            "U": float(stat),
            "p_value": float(p),
            "significant_raw": bool(p < alpha),
            "significant_bonferroni": bool(p < bonferroni_alpha),
        })
    return rows


def print_pairwise_table(rows: list[dict]) -> None:
    """Pretty-prints pairwise_mann_whitney's output as a readable table."""
    if not rows:
        return
    alpha_note = rows[0]["metric"]
    print(f"Pairwise Mann-Whitney U-tests ({alpha_note}):")
    for r in rows:
        flag = " *" if r["significant_bonferroni"] else (" ." if r["significant_raw"] else "")
        print(f"  {r['agent_a']:22s} vs {r['agent_b']:22s}: "
              f"U={r['U']:.1f}, p={r['p_value']:.4f}{flag}")
    print("  (* = significant after Bonferroni correction, "
          ". = significant at raw alpha=0.05 only)")


def save_results_csv(path: str, results_by_name: dict[str, list[dict]]) -> None:
    """Flattens every agent's per-seed rows into one long-format CSV
    (agent, seed, won, score, moves) - easy to hand to any stats tool,
    or re-load later without re-running every agent again.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "seed", "won", "score", "moves"])
        for name, rows in results_by_name.items():
            for row in rows:
                writer.writerow([name, row["seed"], row["won"], row["score"], row["moves"]])