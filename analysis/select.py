"""
Feature-set selection over sliding-window cells.

Everything is scored through analysis.score.HeldOutScorer, so forward selection,
backward elimination and the random baseline all sit on one axis and are directly
comparable. That comparability is the point: the gap between the forward and backward
curves at matched N is the evidence for (or against) cells carrying information
jointly rather than individually.

A "unit" is whatever the selector adds or removes in one step -- a single cell, or a
whole D4 orbit. Orbit-level selection is what you hand to the agents: it is D4-closed,
so masking and canonicalisation commute, and it does not select on within-orbit noise
(the game is isotropic, so the true MI is constant within an orbit).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# units and shapes -- all return FLAT window indices
# --------------------------------------------------------------------------- #

def _offsets(size, radius=None):
    centre = size // 2
    lim = centre if radius is None else radius
    for r in range(size):
        for c in range(size):
            dr, dc = r - centre, c - centre
            if (dr or dc) and max(abs(dr), abs(dc)) <= lim:
                yield r, c, dr, dc


def cell_units(size, radius=None):
    """One unit per cell. Fine resolution -- use for the elbow curve."""
    return [(r * size + c,) for r, c, _, _ in _offsets(size, radius)]


def orbit_units(size, radius=None):
    """One unit per D4 orbit, ordered by Euclidean distance. For a 7x7 there are 9
    orbits, so N steps through 4, 8, 12, ... -- coarser, but D4-closed."""
    orbits = {}
    for r, c, dr, dc in _offsets(size, radius):
        orbits.setdefault(tuple(sorted((abs(dr), abs(dc)))), []).append(r * size + c)
    return [tuple(sorted(v)) for _, v in
            sorted(orbits.items(), key=lambda kv: (kv[0][0] ** 2 + kv[0][1] ** 2))]


def disk(size, radius):
    """Euclidean disk. MI tracks Euclidean distance (Spearman -0.99) far better than
    Chebyshev (-0.89), so disks are the defensible 'regular' comparison shape."""
    return tuple(sorted(r * size + c for r, c, dr, dc in _offsets(size)
                        if dr * dr + dc * dc <= radius * radius))


def square(size, radius):
    return tuple(sorted(r * size + c for r, c, _, _ in _offsets(size, radius)))


def diamond(size, radius):
    return tuple(sorted(r * size + c for r, c, dr, dc in _offsets(size)
                        if abs(dr) + abs(dc) <= radius))


def _cells(units):
    return tuple(sorted(i for u in units for i in u))


# --------------------------------------------------------------------------- #
# selectors
# --------------------------------------------------------------------------- #

def forward_greedy(scorer, units, max_units=None, verbose=True):
    """Add the unit that most improves the held-out score, until nothing helps.

    Myopic by construction: it cannot see two cells that are individually useless but
    jointly determining. Minesweeper has exactly that structure -- the 1-2-1 and
    1-2-2-1 patterns are textbook joint-information cases -- so expect this to
    under-select, and use backward_greedy to find out by how much.

    Note it does NOT stop at the peak: it keeps going so you get the full curve, and
    the elbow is whichever step maximises `score`. A run whose score falls after a few
    steps has not failed; that fall is the finding.
    """
    remaining = list(units)
    chosen, steps = [], []
    limit = len(units) if max_units is None else min(max_units, len(units))

    while remaining and len(chosen) < limit:
        best, best_score = None, -np.inf
        for u in remaining:
            s = scorer.score(_cells(chosen + [u]))
            if s > best_score:
                best, best_score = u, s
        chosen.append(best)
        remaining.remove(best)
        cells = _cells(chosen)
        steps.append({"n_units": len(chosen), "n_cells": len(cells),
                      "cells": cells, "score": best_score, "changed": best,
                      "unseen": scorer.unseen_fraction(cells)})
        if verbose:
            print(f"  +{str(best):<14} n={len(cells):>3}  score={best_score:.4f}")
    return steps


def backward_greedy(scorer, units, min_units=1, verbose=True):
    """Start from every unit, repeatedly drop whichever costs least.

    Evaluates each removal in the context of everything still present, so it keeps
    synergistic groups that forward selection never assembles.

    IMPORTANT -- where to start it. With model="table", a set of ~48 cells scores
    about 0 (nearly every validation pattern is unseen), so every early removal is
    within noise of every other and the first two thirds of the path is arbitrary.
    Held-out scoring fixes the "every removal looks free" problem of the plug-in
    criterion, but it does not by itself make a saturated starting set meaningful.

    Two ways to make this work, either is defensible:
      * start reduced -- pass the top ~16 units by marginal MI, or a shape like
        disk(size, 2), instead of everything; or
      * pass a scorer built with model="logistic", which degrades gracefully at
        large N instead of collapsing.
    Say in the write-up which you used and why; do not report a backward curve that
    starts from a set whose score is indistinguishable from zero.
    """
    current = list(units)
    cells = _cells(current)
    steps = [{"n_units": len(current), "n_cells": len(cells), "cells": cells,
              "score": scorer.score(cells), "changed": None,
              "unseen": scorer.unseen_fraction(cells)}]

    while len(current) > min_units:
        best, best_score = None, -np.inf
        for u in current:
            kept = [x for x in current if x != u]
            s = scorer.score(_cells(kept))
            if s > best_score:                    # least damaging removal
                best, best_score = u, s
        current.remove(best)
        cells = _cells(current)
        steps.append({"n_units": len(current), "n_cells": len(cells), "cells": cells,
                      "score": best_score, "changed": best,
                      "unseen": scorer.unseen_fraction(cells)})
        if verbose:
            print(f"  -{str(best):<14} n={len(cells):>3}  score={best_score:.4f}")
    return list(reversed(steps))                  # ascending N, to plot against forward


def random_baseline(scorer, units, sizes=None, reps=50, seed=0):
    """Percentile ribbon for randomly chosen unit sets.

    Without this, "shape A beats shape B" has no scale. With it, you can say whether a
    selected shape is better than an arbitrary one of the same size -- which is the
    claim the report actually needs to make.
    """
    rng = np.random.default_rng(seed)
    sizes = sizes or range(1, len(units) + 1)
    out = []
    for k in sizes:
        scores = [scorer.score(_cells([units[i] for i in rng.choice(len(units), k, replace=False)]))
                  for _ in range(reps)]
        cells = _cells(units[:k])
        out.append({"n_units": k, "n_cells": len(cells),
                    "p5": float(np.percentile(scores, 5)),
                    "p50": float(np.percentile(scores, 50)),
                    "p95": float(np.percentile(scores, 95))})
    return out


def stability(scorer_factory, units, n_boot=50, max_units=5, seed=0):
    """Bootstrap forward selection over GAMES and record WHEN each unit is picked.

    Resampling games (not rows) is what respects the dependence between samples from
    one board. Expect ring 1 at 100 % and the outer picks to be much less stable --
    reporting that honestly is worth more than a clean ordering you cannot defend.

    `max_units` must be smaller than len(units). At max_units == len(units) forward
    selection exhausts the pool, every unit is picked in every replicate, and every
    frequency is 1.0 before any data is touched -- a deterministic identity dressed
    up as a measurement. Refused rather than returned.

    Per-step counts, not a single total. Frequency-within-j is then a prefix of this
    run for every j <= max_units: the selector is deterministic given a scorer and
    never touches `rng`, so a budget-j run with the same seed would produce exactly
    these first j steps. Report several budgets off one set of fits.

    Membership frequency alone saturates: with a budget of k, up to k units can sit
    at 1.00 whenever the selected SET is stable but its ORDER is not. `mean_rank`
    is what separates "always first" from "always fifth", which is the claim the
    selected-feature-set section actually needs to make.

    Returns dict with:
      per_step   unit -> [count at step 1, ..., step max_units]
      prefix     unit -> [freq within 1 step, ..., within max_units steps]
      freq       unit -> prefix[-1], i.e. picked at all within the budget
      mean_rank  unit -> mean 1-based step when picked (nan if never picked)
      n_boot     replicate count, for binomial error bars on freq
    """
    if max_units >= len(units):
        raise ValueError(
            f"max_units={max_units} >= len(units)={len(units)}: forward selection "
            f"would exhaust the candidate pool, so every unit is picked in every "
            f"replicate and all frequencies are 1.0 by construction. Selection "
            f"frequency only measures anything when the budget is smaller than the "
            f"pool -- use max_units <= {len(units) - 1}.")

    rng = np.random.default_rng(seed)
    per_step = {u: [0] * max_units for u in units}
    for _ in range(n_boot):
        sc = scorer_factory(rng)
        for k, step in enumerate(forward_greedy(sc, units, max_units=max_units,
                                                verbose=False)):
            per_step[step["changed"]][k] += 1

    prefix = {u: [float(v) for v in np.cumsum(c) / n_boot] for u, c in per_step.items()}
    mean_rank = {}
    for u, c in per_step.items():
        picked = sum(c)
        mean_rank[u] = (sum((k + 1) * v for k, v in enumerate(c)) / picked
                        if picked else float("nan"))
    return {"per_step": per_step, "prefix": prefix,
            "freq": {u: p[-1] for u, p in prefix.items()},
            "mean_rank": mean_rank, "n_boot": n_boot}


# --------------------------------------------------------------------------- #

def format_steps(steps, label):
    out = [f"--- {label} ---", f"{'N cells':>8} {'score (bits)':>13} {'unseen':>8}  change"]
    for s in steps:
        out.append(f"{s['n_cells']:>8} {s['score']:>13.4f} {100*s['unseen']:>7.1f}%  {s['changed']}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    from analysis.score import HeldOutScorer, load_split

    prefix = sys.argv[1] if len(sys.argv) > 1 else "cos700_dataset"
    Xtr, ytr, Xva, yva = load_split(prefix, target="is_mine")
    scorer = HeldOutScorer(Xtr, ytr, Xva, yva)

    size = Xtr.shape[1]
    RADIUS = 3                                    # inner 7x7 of the 9x9 window

    cells = cell_units(size, RADIUS)
    orbits = orbit_units(size, RADIUS)
    ceiling = scorer.score(_cells(cells))

    print(f"train {len(ytr):,}  val {len(yva):,}")
    print(f"full 7x7 ceiling: {ceiling:.4f} bits "
          f"(unseen {100*scorer.unseen_fraction(_cells(cells)):.1f}%)\n")

    print("forward, orbit-level:")
    fwd = forward_greedy(scorer, orbits)
    print("\nbackward, orbit-level:")
    bwd = backward_greedy(scorer, orbits)
    rnd = random_baseline(scorer, orbits, reps=30)

    # Orbits have different sizes (4 or 8 cells), so forward and backward reach
    # different CELL counts at the same step. Compare them at equal n_units and carry
    # n_cells along as the x-value each curve is actually plotted at.
    print(f"\n{'units':>6} {'fwd cells':>10} {'forward':>9} {'bwd cells':>10} "
          f"{'backward':>9} {'rand p50':>9} {'rand p95':>9}")
    for f, b, r in zip(fwd, bwd, rnd):
        assert f["n_units"] == b["n_units"] == r["n_units"]
        print(f"{f['n_units']:>6} {f['n_cells']:>10} {f['score']:>9.4f} "
              f"{b['n_cells']:>10} {b['score']:>9.4f} {r['p50']:>9.4f} {r['p95']:>9.4f}")

    np.savez_compressed("analysis/selections.npz",
                        forward=np.array([s["score"] for s in fwd]),
                        forward_n=np.array([s["n_cells"] for s in fwd]),
                        backward=np.array([s["score"] for s in bwd]),
                        backward_n=np.array([s["n_cells"] for s in bwd]),
                        ceiling=ceiling)
