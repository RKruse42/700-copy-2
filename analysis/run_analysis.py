import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats.information import (bin_target, mi_map, permutation_null,
                                d4_orbit_average, orbit_profile, asymmetry)
from analysis.score import HeldOutScorer, load_split, saturation_table, format_saturation_table
from analysis.select import (cell_units, orbit_units, disk, square, diamond,
                             forward_greedy, backward_greedy, random_baseline, _cells)

RADIUS = 3               # inner 7x7 of the 9x9 window
SYNERGY_MAX_N = 16       # table scorer stays meaningful to roughly here
LOGISTIC_MAX_TRAIN = 20_000   # rows subsampled for each logistic fit
RANDOM_BASELINE_REPS = 10     # ribbon reps; 10 is enough to show the shape
STABILITY_BUDGET = 5          # forward steps counted per bootstrap; MUST be < 9 orbits

# categorical slots 1-4, validated light-mode (adjacent CVD dE 9.1, normal 22.9)
C = {"forward": "#2a78d6", "backward": "#eb6834", "disk": "#1baf7a", "square": "#eda100"}
INK, MUTED, GRID = "#1a1a19", "#52514e", "#c3c2b7"
SURFACE = "#fcfcfb"


def marginal_ranking(windows, probs, radius=RADIUS):
    """Cells ordered by I(cell; binned csp_prob), best first. Used to give backward
    elimination a start set that is small enough for the table scorer to score."""
    m = mi_map(windows, bin_target(probs))
    size = m.shape[0]
    allowed = {c[0] for c in cell_units(size, radius)}
    flat = [(m[i // size, i % size], i) for i in range(size * size) if i in allowed]
    return [i for _, i in sorted(flat, reverse=True)]


def part_a_synergy(Xtr, ytr, Xva, yva, ptr):
    print("\n" + "=" * 74)
    print("A. SYNERGY DIAGNOSTIC  (table scorer, cell-level, N <= %d)" % SYNERGY_MAX_N)
    print("=" * 74)
    scorer = HeldOutScorer(Xtr, ytr, Xva, yva, model="table")

    # Both directions must search the SAME candidate pool, or the gap measures the
    # pool rather than the search direction. Backward has to start from a set the
    # table can still score, so the pool is the top-N cells by marginal MI and
    # forward is restricted to it too.
    pool = [(i,) for i in marginal_ranking(Xtr, ptr)[:SYNERGY_MAX_N]]
    print(f"pool: top {SYNERGY_MAX_N} cells by marginal MI (shared by both directions)")

    print("forward ...")
    fwd = forward_greedy(scorer, pool, verbose=False)
    print("backward ...")
    bwd = backward_greedy(scorer, pool, verbose=False)

    size = Xtr.shape[1]
    rc = lambda i: f"({i // size - size // 2:+d},{i % size - size // 2:+d})"
    print(f"\n{'N':>3} {'forward':>9} {'backward':>9} {'gap':>8}   forward adds")
    rows = []
    for f, b in zip(fwd, bwd):
        gap = b["score"] - f["score"]
        rows.append((f["n_cells"], f["score"], b["score"], gap))
        print(f"{f['n_cells']:>3} {f['score']:>9.4f} {b['score']:>9.4f} {gap:>+8.4f}   "
              f"{rc(f['changed'][0])}")

    best_gap = max(rows, key=lambda r: r[3])
    print(f"\nlargest backward-over-forward gap: {best_gap[3]:+.4f} bits at N={best_gap[0]}")
    print("  gap ~ 0 across the healthy region  -> information is effectively additive;")
    print("  gap clearly positive somewhere     -> cells carry information jointly there.")
    return rows


def part_b_elbow(Xtr, ytr, Xva, yva, label):
    print("\n" + "=" * 74)
    print(f"B. ELBOW  (logistic scorer, orbit-level)   [{label}]")
    print("=" * 74)
    scorer = HeldOutScorer(Xtr, ytr, Xva, yva, model="logistic",
                           max_train=LOGISTIC_MAX_TRAIN)
    size = Xtr.shape[1]
    orbits = orbit_units(size, RADIUS)

    ceiling = scorer.score(_cells(cell_units(size, RADIUS)))
    print(f"full 7x7 ceiling: {ceiling:.4f} bits   (H(Y) = {scorer.h_val:.4f})")
    if ceiling <= 0.01:
        print("  !! ceiling is at or below zero -- this split is too small for the "
              "logistic scorer to fit 528 one-hot features. Normalised percentages "
              "below are meaningless; report raw bits, or use more games.")
    pct = (lambda v: f"{100*v/ceiling:5.1f}%") if ceiling > 0.01 else (lambda v: "   n/a")

    print("forward ...")
    fwd = forward_greedy(scorer, orbits, verbose=False)
    print("backward ...")
    bwd = backward_greedy(scorer, orbits, verbose=False)
    print("random baseline ...")
    rnd = random_baseline(scorer, orbits, reps=RANDOM_BASELINE_REPS)

    shapes = {}
    for r in (1, 2, 3):
        for name, fn in (("disk", disk), ("square", square), ("diamond", diamond)):
            cells = fn(size, r)
            if cells:
                shapes[f"{name}{r}"] = (len(cells), scorer.score(cells))

    print(f"\n{'units':>6} {'fwd N':>6} {'forward':>9} {'bwd N':>6} {'backward':>9} "
          f"{'rand p50':>9} {'rand p95':>9}")
    for f, b, r in zip(fwd, bwd, rnd):
        print(f"{f['n_units']:>6} {f['n_cells']:>6} {f['score']:>9.4f} "
              f"{b['n_cells']:>6} {b['score']:>9.4f} {r['p50']:>9.4f} {r['p95']:>9.4f}")

    print("\nreference shapes:")
    for k, (n, s) in sorted(shapes.items(), key=lambda kv: kv[1][0]):
        print(f"  {k:>9}  N={n:>2}  {s:.4f} bits  ({pct(s)} of ceiling)")

    peak = max(fwd, key=lambda s: s["score"])
    print(f"\npeak: {peak['score']:.4f} bits at N={peak['n_cells']} "
          f"({pct(peak['score'])} of the full 7x7)")
    return fwd, bwd, rnd, shapes, ceiling


def part_c_stability(prefix, n_boot=50, max_orbits=STABILITY_BUDGET, seed=42):
    """Bootstrap stability: resample games, re-run forward selection, record when each
    unit is picked.

    Uses the stability() hook from select.py with a scorer_factory that rebuilds
    HeldOutScorer from a bootstrap resample of games.  Out-of-bag games form the
    validation set, which is the only statistically sound choice (using in-bag rows
    for val would systematically inflate scores and make the inner elbow look better
    than it is).

    max_orbits: the selection BUDGET, and it must stay below the 9 orbits in the 7x7
    -- at 9 the search exhausts the pool and every frequency is 1.0 by construction.
    5 covers the region where the forward curve is still climbing; the marginal gain
    past step 4 (+0.006 bits at step 5) is small enough that the order should visibly
    churn there, which is the instability this figure exists to show.

    An orbit budget is not a cell budget: orbits hold 4 or 8 cells and are picked in
    a different order in every replicate, so 5 orbits is somewhere in 20-32 cells
    depending on the resample. Quote the orbit count in the caption and give N as a
    range, not a single number.
    """
    from analysis.select import stability

    print("\n" + "=" * 74)
    print(f"C. STABILITY  (bootstrap n={n_boot}, orbit-level forward, max {max_orbits} orbits)")
    print("=" * 74)

    # Load the full dataset (train + val) so we can re-split over games freely.
    arrays = {}
    for split in ("train", "val"):
        with np.load(f"{prefix}_{split}.npz") as d:
            for k in d.files:
                arrays.setdefault(k, []).append(d[k])
    full = {k: np.concatenate(v) for k, v in arrays.items()}

    all_games = np.unique(full["seeds"])
    n_games   = len(all_games)
    windows   = full["windows"]
    y         = full["is_mine"].astype(float)
    seeds_arr = full["seeds"]

    size   = windows.shape[1]   # 9
    orbits = orbit_units(size, RADIUS)

    def scorer_factory(rng):
        """Build a HeldOutScorer from one bootstrap resample of games."""
        boot_idx   = rng.choice(n_games, size=n_games, replace=True)
        boot_games = all_games[boot_idx]
        oob_games  = np.setdiff1d(all_games, boot_games)

        # Edge case: if every game was drawn (rare for n_games >= 50), fall back.
        if len(oob_games) == 0:
            oob_games = all_games[rng.choice(n_games, size=max(1, n_games // 5),
                                             replace=False)]

        in_mask  = np.isin(seeds_arr, boot_games)
        oob_mask = np.isin(seeds_arr, oob_games)

        return HeldOutScorer(windows[in_mask],  y[in_mask],
                             windows[oob_mask], y[oob_mask],
                             model="logistic", max_train=LOGISTIC_MAX_TRAIN,
                             seed=int(rng.integers(1 << 31)))

    # Wrap scorer_factory to print progress every 10 replicates.
    completed = [0]
    def factory_with_progress(rng):
        sc = scorer_factory(rng)
        completed[0] += 1
        if completed[0] % 10 == 0:
            print(f"  {completed[0]}/{n_boot} resamples done")
        return sc

    stab = stability(factory_with_progress, orbits,
                     n_boot=n_boot, max_units=max_orbits, seed=seed)

    # Every budget j <= max_orbits is a prefix of this one run, so report the whole
    # ladder: a smaller budget needs no second set of fits.
    rc = lambda i: f"({i // size - size // 2:+d},{i % size - size // 2:+d})"
    hdr = "  ".join(f"j={j+1:<4}" for j in range(max_orbits))
    print(f"\nselection frequency within the first j steps, and mean pick rank:")
    print(f"  {'orbit':<12} {'cells':>5}  {hdr}  {'rank':>5}")
    order = sorted(orbits, key=lambda u: (-stab["freq"][u], stab["mean_rank"][u]))
    for orb in order:
        pre = "  ".join(f"{v:<6.2f}" for v in stab["prefix"][orb])
        r = stab["mean_rank"][orb]
        rs = "  -  " if r != r else f"{r:5.2f}"
        print(f"  {rc(orb[0]):<12} {len(orb):>5}  {pre}  {rs}")
    print(f"  (frequencies are out of n_boot={stab['n_boot']}; "
          f"binomial se at p=0.5 is {0.5 / n_boot ** 0.5:.3f})")

    return stab, orbits, size


def figure_stability(stab, orbits, size, depth=None,
                     out="figures/fig4_stability.png",
                     csv_out="figures/fig4_stability.csv"):
    """7x7 heatmap of per-cell bootstrap selection frequency, annotated with mean rank.

    depth: report frequency within the first `depth` forward steps. Defaults to the
    full budget the bootstrap was run at. Because stability() stores per-step counts
    and the selector is deterministic given a scorer, any depth <= that budget is the
    run you would have got by setting the budget there -- no refitting needed.

    Frequency alone saturates once the selected SET stabilises even if its ORDER does
    not, so the mean rank is printed under each frequency. Two cells both at 1.00 but
    at rank 1.0 and rank 4.6 are not the same finding.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)

    budget = len(next(iter(stab["prefix"].values())))
    depth  = budget if depth is None else min(depth, budget)
    freq   = {u: p[depth - 1] for u, p in stab["prefix"].items()}

    # Mean rank must be conditioned on the SAME depth as the frequency, or a cell
    # reading 0.00 can still show a rank earned at a step this figure excludes.
    rank_at = {}
    for u, c in stab["per_step"].items():
        picked = sum(c[:depth])
        rank_at[u] = (sum((k + 1) * v for k, v in enumerate(c[:depth])) / picked
                      if picked else float("nan"))

    # Build (size x size) maps: each cell carries its orbit's frequency and mean rank.
    fmap = np.full((size, size), np.nan)
    rmap = np.full((size, size), np.nan)
    for orb_tuple, f in freq.items():
        for flat_idx in orb_tuple:
            r, c = divmod(flat_idx, size)
            fmap[r, c] = f
            rmap[r, c] = rank_at[orb_tuple]

    centre = size // 2
    inner  = fmap[centre-3:centre+4, centre-3:centre+4].copy()
    rank   = rmap[centre-3:centre+4, centre-3:centre+4].copy()
    inner[3, 3] = np.nan   # centre cell is the target, not a feature
    rank[3, 3]  = np.nan

    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)

    im = ax.imshow(inner, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("selection frequency", color=MUTED)
    cbar.ax.yaxis.set_tick_params(color=MUTED)

    # Annotate each cell with its frequency and, under it, the mean pick rank.
    for r in range(inner.shape[0]):
        for c in range(inner.shape[1]):
            v = inner[r, c]
            if np.isnan(v):
                continue
            col = INK if v < 0.6 else SURFACE
            ax.text(c, r, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=col)
            rk = rank[r, c]
            ax.text(c, r, "r -" if rk != rk else f"r{rk:.1f}", ha="center", va="top",
                    fontsize=6, color=col, alpha=0.75)

    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels([f"{i-3:+d}" for i in range(7)], color=MUTED)
    ax.set_yticklabels([f"{i-3:+d}" for i in range(7)], color=MUTED)
    ax.set_xlabel("col offset from centre", color=MUTED, fontsize=9)
    ax.set_ylabel("row offset from centre", color=MUTED, fontsize=9)
    ax.set_title(f"Bootstrap selection frequency, first {depth} forward steps\n"
                 f"orbit-level, {stab['n_boot']} resamples over games "
                 f"(r = mean pick rank)",
                 color=INK, fontsize=9.5, loc="left")

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")

    # CSV: one row per non-centre cell, with the frequency at EVERY budget j <= the
    # one the bootstrap ran at, so the write-up can quote a smaller budget without
    # re-running anything.
    rows = []
    for r in range(7):
        for c in range(7):
            if np.isnan(inner[r, c]):
                continue
            flat = (r + centre - 3) * size + (c + centre - 3)
            orb = next(u for u in stab["prefix"] if flat in u)
            rows.append([r - 3, c - 3, rank_at[orb]] + stab["prefix"][orb])
    hdr = "delta_row,delta_col,mean_rank," + ",".join(
        f"freq_within_{j+1}" for j in range(budget))
    np.savetxt(csv_out, np.array(rows, dtype=float), delimiter=",",
               header=hdr, comments="")
    print(f"wrote {csv_out}")


def figure3(fwd, bwd, rnd, shapes, ceiling, out="figures/fig3_elbow.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    fn = [s["n_cells"] for s in fwd]
    fs = [s["score"] / ceiling for s in fwd]
    bn = [s["n_cells"] for s in bwd]
    bs = [s["score"] / ceiling for s in bwd]
    rn = [s["n_cells"] for s in rnd]

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    # random ribbon first, so the selected curves sit on top of it
    ax1.fill_between(rn, [s["p5"] / ceiling for s in rnd], [s["p95"] / ceiling for s in rnd],
                     color=GRID, alpha=0.45, linewidth=0, label="random shapes (5-95%)")
    # forward solid and slightly heavier, backward dashed on top: where the two
    # selectors agree exactly (which happens whenever the orbit landscape is nested)
    # the reader sees dashed orange over solid blue rather than one vanished series
    ax1.plot(fn, fs, color=C["forward"], linewidth=3, marker="o", markersize=6,
             label="forward", zorder=3)
    ax1.plot(bn, bs, color=C["backward"], linewidth=1.8, linestyle=(0, (4, 3)),
             marker="s", markersize=4.5, markerfacecolor=SURFACE, markeredgewidth=1.4,
             label="backward", zorder=4)

    for name, mk, col in (("disk", "^", C["disk"]), ("square", "D", C["square"])):
        pts = sorted((v for k, v in shapes.items() if k.startswith(name)))
        if pts:
            ax1.plot([p[0] for p in pts], [p[1] / ceiling for p in pts], mk,
                     color=col, markersize=7, linestyle="--", linewidth=1.5, label=name + "s")

    # relief for the low-contrast slots: label the lines directly, not by colour alone
    peak = max(fwd, key=lambda s: s["score"])
    ax1.annotate(f"peak {peak['score']:.3f} bits\nat N={peak['n_cells']}",
                 xy=(peak["n_cells"], peak["score"] / ceiling),
                 xytext=(peak["n_cells"] + 6, peak["score"] / ceiling - 0.16),
                 color=INK, fontsize=9,
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))

    ax1.set_xlabel("cells in the feature set (N)", color=MUTED, fontsize=10)
    ax1.set_ylabel("fraction of full 7x7 information", color=MUTED, fontsize=10)
    ax1.set_title("Information recovered", color=INK, fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=9, labelcolor=MUTED)

    per = [s["score"] / (11 * max(s["n_cells"], 1)) for s in fwd]
    ax2.plot(fn, per, color=C["forward"], linewidth=2, marker="o", markersize=5)
    ax2.set_xlabel("cells in the feature set (N)", color=MUTED, fontsize=10)
    ax2.set_ylabel("bits per input weight", color=MUTED, fontsize=10)
    ax2.set_title("Information per parameter (11 one-hot inputs per cell)",
                  color=INK, fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\nwrote {out}")


def figure1(probs_train, is_mine_train, probs_val, is_mine_val,
            n_seen_train, n_seen_val,
            out="figures/fig1_dataset.png", csv_out="figures/fig1_dataset.csv"):
    """Dataset composition (six-bar histogram) + calibration table.

    Reservoir sampling picks each cell's recorded turn with probability 1/n_seen,
    and n_seen correlates with the outcome (a mine is never played, so it lingers on
    the frontier and is read many times before the one turn that gets kept). That
    makes the raw sample under-represent mines in the intermediate probability bins.
    Weighting each row by its n_seen is the Horvitz-Thompson correction for that
    unequal inclusion probability, and it recovers the all-turns distribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)

    # --- composition bars --------------------------------------------------- #
    labels = ["{0}", "(0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1)", "{1}"]
    colors = ["#2a78d6", "#7ab4ef", "#c3c2b7", "#c3c2b7", "#eb9d7a", "#eb6834"]

    p_all = np.concatenate([probs_train, probs_val])
    w_all = np.concatenate([n_seen_train, n_seen_val]).astype(float)
    cal_bins = [
        p_all == 0,
        (p_all > 0) & (p_all < 0.25),
        (p_all >= 0.25) & (p_all < 0.5),
        (p_all >= 0.5) & (p_all < 0.75),
        (p_all >= 0.75) & (p_all < 1),
        p_all == 1,
    ]
    counts = np.array([m.sum() for m in cal_bins], dtype=float)
    fracs = np.array([w_all[m].sum() for m in cal_bins]) / w_all.sum()

    # --- calibration table -------------------------------------------------- #
    # 6 bins matching the same edges, evaluated on the combined set, weighted by
    # n_seen (Horvitz-Thompson correction for the reservoir's inclusion probability).
    y_all = np.concatenate([is_mine_train, is_mine_val]).astype(float)
    cal_rate = [np.average(y_all[m], weights=w_all[m]) if m.any() else float("nan")
                for m in cal_bins]
    # Plot against the weighted mean predicted probability in the bin, not the bin
    # midpoint: the mass sits at 1/4, 1/3, 1/2 etc, so the midpoint overstates bins
    # like [0.25, 0.5) before any bias enters.
    cal_pred = [np.average(p_all[m], weights=w_all[m]) if m.any() else float("nan")
                for m in cal_bins]

    # --- plot --------------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.5, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    xs = np.arange(len(labels))
    bars = ax1.bar(xs, fracs * 100, color=colors, edgecolor=SURFACE, linewidth=0.8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax1.set_ylabel("% of samples", color=MUTED, fontsize=10)
    ax1.set_title("Dataset composition", color=INK, fontsize=11, loc="left")
    n_total = int(counts.sum())
    ax1.text(0.98, 0.97, f"n = {n_total:,}", transform=ax1.transAxes,
             ha="right", va="top", color=MUTED, fontsize=9)

    # calibration: n_seen-weighted mean predicted probability vs actual mine rate
    ax2.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--", label="perfect")
    ax2.scatter(cal_pred, cal_rate, color=colors, s=60, zorder=3,
                edgecolors=INK, linewidths=0.6)
    for mid, rate, lbl in zip(cal_pred, cal_rate, labels):
        if not np.isnan(rate):
            ax2.annotate(lbl, (mid, rate), textcoords="offset points",
                         xytext=(5, 3), fontsize=7.5, color=MUTED)
    ax2.set_xlabel("CSP probability (n_seen-weighted bin mean)", color=MUTED, fontsize=10)
    ax2.set_ylabel("actual mine rate (n_seen-weighted)", color=MUTED, fontsize=10)
    ax2.set_title("Calibration", color=INK, fontsize=11, loc="left")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")

    # --- CSV backing data --------------------------------------------------- #
    rows = np.column_stack([cal_pred, fracs, cal_rate])
    np.savetxt(csv_out, rows, delimiter=",",
               header="bin_mean_pred,fraction,actual_mine_rate", comments="")
    print(f"wrote {csv_out}")
    return counts, cal_rate


def figure2(windows_train, probs_train, null, out="figures/fig2_mi.png",
            csv_out="figures/fig2_orbits.csv"):
    """(a) D4-averaged MI heatmap; (b) MI vs Euclidean distance, one point per orbit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)

    target = bin_target(probs_train)
    raw = mi_map(windows_train, target)
    avg = d4_orbit_average(raw)
    null_mean = null["mean"]
    null_p95 = null["p95"]

    profile = orbit_profile(raw, radius=3, null_mean=null_mean)

    # --- plot --------------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    # (a) heatmap — inner 7x7 of the 9x9 window
    centre = avg.shape[0] // 2
    inner = avg[centre-3:centre+4, centre-3:centre+4].copy()
    inner[3, 3] = np.nan   # centre cell is the target — never included
    im = ax1.imshow(inner, cmap="Blues", interpolation="nearest",
                    vmin=0, vmax=inner[~np.isnan(inner)].max())
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).set_label("bits", color=MUTED)
    ax1.set_title("(a) MI heatmap (D4-averaged, bias-corrected)", color=INK,
                  fontsize=10, loc="left")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(0.02, 0.02, f"null p95 = {null_p95:.5f} bits",
             transform=ax1.transAxes, fontsize=8, color=MUTED, va="bottom")

    # (b) MI vs Euclidean distance
    euclid = [r["euclidean"] for r in profile]
    mi_vals = [r["mi"] for r in profile]
    mi_sd = [r["mi_std"] for r in profile]
    orbit_labels = [str(r["orbit"]) for r in profile]

    ax2.axhspan(0, null_p95, color=GRID, alpha=0.3, label=f"null band (p95={null_p95:.5f})")
    ax2.errorbar(euclid, mi_vals, yerr=mi_sd, fmt="o", color=C["forward"],
                 markersize=7, linewidth=1.5, capsize=3, markeredgecolor=INK,
                 markeredgewidth=0.6, label="D4 orbit (bias-corrected)")
    for x, y, lbl in zip(euclid, mi_vals, orbit_labels):
        ax2.annotate(lbl, (x, y), textcoords="offset points",
                     xytext=(5, 2), fontsize=7.5, color=MUTED)

    ax2.set_xlabel("Euclidean distance from centre", color=MUTED, fontsize=10)
    ax2.set_ylabel("I(cell; target) − null mean  (bits)", color=MUTED, fontsize=10)
    ax2.set_title("(b) MI vs distance, per D4 orbit", color=INK, fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    ax2.grid(True, color=GRID, linewidth=0.6, alpha=0.5)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")

    # --- CSV backing data --------------------------------------------------- #
    rows = np.column_stack([euclid, mi_vals, mi_sd, [r["n_cells"] for r in profile]])
    np.savetxt(csv_out, rows, delimiter=",",
               header="euclidean,mi_bits,mi_sd,n_cells", comments="")
    print(f"wrote {csv_out}")
    return profile


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "cos700_dataset"
    Xtr, ytr, Xva, yva = load_split(prefix, target="is_mine")
    with np.load(f"{prefix}_train.npz") as d:
        ptr = d["csp_probs"]
        Xtr_raw = d["windows"]          # raw windows for MI maps
        itr = d["is_mine"]
        ntr = d["n_seen"]
    with np.load(f"{prefix}_val.npz") as d:
        pva = d["csp_probs"]
        iva = d["is_mine"]
        nva = d["n_seen"]

    print(f"train {len(ytr):,}   val {len(yva):,}")

    # --- permutation null (cached) ------------------------------------------ #
    null_cache = "figures/null.npz"
    os.makedirs("figures", exist_ok=True)
    if os.path.exists(null_cache):
        print(f"loading cached null from {null_cache}")
        null = dict(np.load(null_cache, allow_pickle=True))
        # scalar items are 0-d arrays after np.load; unwrap them
        null = {k: v.item() if v.ndim == 0 else v for k, v in null.items()}
    else:
        print("computing permutation null (200 perms) ...")
        null = permutation_null(Xtr_raw, bin_target(ptr), n_perm=200)
        np.savez(null_cache, **{k: np.array(v) for k, v in null.items()})
        print(f"wrote {null_cache}")
    print(f"null mean={null['mean']:.5f} bits  p95={null['p95']:.5f} bits  "
          f"asymmetry_p95={null['asymmetry_p95']:.3f}")

    # --- Figures 1 and 2 ---------------------------------------------------- #
    print("\n--- Figure 1: dataset composition + calibration ---")
    figure1(ptr, itr, pva, iva, ntr, nva)

    print("\n--- Figure 2: MI heatmap + orbit profile ---")
    figure2(Xtr_raw, ptr, null)

    # --- saturation / synergy / elbow (as before) --------------------------- #
    scorer = HeldOutScorer(Xtr, ytr, Xva, yva, model="table")
    size = Xtr.shape[1]
    order = marginal_ranking(Xtr, ptr)
    print("\nsaturation table (plug-in vs held-out), cells in marginal-MI order:")
    print(format_saturation_table(saturation_table(scorer, order, max_n=12)))

    syn = part_a_synergy(Xtr, ytr, Xva, yva, ptr)
    fwd, bwd, rnd, shapes, ceiling = part_b_elbow(Xtr, ytr, Xva, yva, "all frontier cells")

    np.savetxt("figures/fig3_curves.csv",
               np.array([[f["n_cells"], f["score"], b["n_cells"], b["score"],
                          r["p5"], r["p50"], r["p95"]] for f, b, r in zip(fwd, bwd, rnd)]),
               delimiter=",", header="fwd_n,fwd_bits,bwd_n,bwd_bits,rand_p5,rand_p50,rand_p95",
               comments="")
    np.savetxt("figures/synergy.csv", np.array(syn), delimiter=",",
               header="n_cells,forward_bits,backward_bits,gap_bits", comments="")
    figure3(fwd, bwd, rnd, shapes, ceiling)

    # --- ambiguous-only robustness run ------------------------------------- #
    amb_tr, amb_va = (ptr > 0) & (ptr < 1), (pva > 0) & (pva < 1)
    print(f"\nambiguous subset: train {amb_tr.sum():,}  val {amb_va.sum():,}")
    if amb_tr.sum() < 5000 or amb_va.sum() < 1000:
        print("  skipped: too few ambiguous rows to fit 528 one-hot features. "
              "This run needs the full corpus, not a smoke-test split.")
    else:
        part_b_elbow(Xtr[amb_tr], ytr[amb_tr], Xva[amb_va], yva[amb_va], "0 < p < 1 only")

    # --- Block 11: bootstrap stability -------------------------------------- #
    # The budget MUST stay below the 9 orbits in the 7x7. Running all 9 makes forward
    # selection exhaust the pool, so every orbit is picked in every replicate and the
    # figure is uniformly 1.00 -- an identity, not a measurement. 5 covers the region
    # where the forward curve is still climbing (steps 4 and 5 add only +0.019 and
    # +0.006 bits, so that is where the order should start to churn).
    stab, orbits_used, win_size = part_c_stability(prefix, max_orbits=STABILITY_BUDGET)

    # Two figures off ONE bootstrap. depth=5 is the stability evidence (where the
    # order starts to churn); depth=4 is the selected feature set itself, which is a
    # different claim and wants a figure that is not muddied by the churning step.
    figure_stability(stab, orbits_used, win_size)
    figure_stability(stab, orbits_used, win_size, depth=4,
                     out="figures/fig4_selected.png",
                     csv_out="figures/fig4_selected.csv")

    # Cache the raw counts: re-plotting at another depth should never cost a re-run.
    np.savez("figures/fig4_stability.npz",
             orbit_first=np.array([u[0] for u in orbits_used]),
             per_step=np.array([stab["per_step"][u] for u in orbits_used]),
             n_boot=stab["n_boot"])
    print("wrote figures/fig4_stability.npz")


if __name__ == "__main__":
    main()
