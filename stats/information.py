"""
Information-theoretic primitives for the sliding-window analysis.

One estimator, one unit. Everything here is a contingency-table ("plug-in") estimate
reported in BITS. This module replaces stats/mutual_information.py (which used a k-NN
continuous estimator on categorical codes), stats/mrmr.py's private redundancy
estimator, and tools/radial_average.py.

Window alphabet: 0-8 = adjacent mine count, 9 = hidden, 10 = out of bounds.
"""

import numpy as np

N_SYMBOLS = 11          # cell alphabet: 0..10
BIN_EDGES = (0.25, 0.5, 0.75)


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #

def bin_target(p):
    """Discretise csp_prob into 6 classes, keeping the point masses separate.

        0 -> {0}      1 -> (0, .25)   2 -> [.25, .5)
        3 -> [.5,.75) 4 -> [.75, 1)   5 -> {1}

    The point masses at 0 and 1 are their own classes deliberately: they are
    qualitatively different events (proven safe / proven mine), not just extreme
    probabilities, and lumping them into neighbouring bins hides the structure of
    the corpus.
    """
    p = np.asarray(p, dtype=float)
    t = (np.digitize(p, BIN_EDGES, right=False) + 1).astype(np.int64)
    t[p == 0.0] = 0
    t[p == 1.0] = 5
    return t


def binary_entropy(p):
    """H(p) in bits, with the 0 log 0 = 0 convention."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


# --------------------------------------------------------------------------- #
# mutual information
# --------------------------------------------------------------------------- #

def mi_plugin(x, y, nx=None, ny=None):
    """I(X;Y) in bits from the empirical joint distribution.

    x, y must be non-negative integer arrays. Pass nx/ny (alphabet sizes) when known
    to skip the relabelling pass -- this is the hot path inside the selectors.
    """
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if nx is None:
        x = np.unique(x, return_inverse=True)[1]
        nx = x.max() + 1 if len(x) else 1
    if ny is None:
        y = np.unique(y, return_inverse=True)[1]
        ny = y.max() + 1 if len(y) else 1

    n = len(x)
    if n == 0:
        return 0.0

    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (px @ py)[nz])))


def mi_map(windows, target):
    """Per-cell I(cell; target) in bits. Returns an array shaped like one window."""
    windows = np.asarray(windows)
    n, h, w = windows.shape
    flat = windows.reshape(n, h * w).astype(np.int64)
    target = np.asarray(target, dtype=np.int64)
    ny = int(target.max()) + 1

    scores = np.empty(h * w)
    for i in range(h * w):
        scores[i] = mi_plugin(flat[:, i], target, nx=N_SYMBOLS, ny=ny)
    return scores.reshape(h, w)


def permutation_null(windows, target, n_perm=200, seed=0):
    """Noise floor for mi_map.

    Shuffling the target destroys every real dependence while preserving both
    marginals, so whatever MI survives is pure finite-sample bias. Returns the
    per-cell mean (use it as a bias correction) and the pooled percentiles (use the
    95th as the threshold below which per-cell claims are unsupported).
    """
    rng = np.random.default_rng(seed)
    target = np.asarray(target)
    maps = np.stack([mi_map(windows, rng.permutation(target)) for _ in range(n_perm)])
    return {
        "per_cell_mean": maps.mean(axis=0),
        "mean": float(maps.mean()),
        "p95": float(np.percentile(maps, 95)),
        "p99": float(np.percentile(maps, 99)),
        # asymmetry of a pure-noise map: anything at or below this is not evidence
        # of a sampling artefact, it is just finite n.
        "asymmetry_p95": float(np.percentile([asymmetry(m) for m in maps], 95)),
        "n_perm": n_perm,
    }


# --------------------------------------------------------------------------- #
# D4 symmetry
# --------------------------------------------------------------------------- #

def d4_orbit_average(matrix):
    """Average a square map over the eight symmetries of the square.

    Mine placement is i.i.d. uniform and the rules are isotropic, so the true map is
    D4-symmetric. Averaging eight estimates of the same quantity is a legitimate
    variance reduction -- unlike 8x row augmentation, which achieves the same
    symmetry by duplicating data.

    Estimate the map WITHOUT this first and inspect the residual asymmetry: that is
    your honest noise diagnostic, and it disappears once you average.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"d4_orbit_average needs a square matrix, got {m.shape}")
    images = []
    for k in range(4):
        r = np.rot90(m, k)
        images.append(r)
        images.append(np.fliplr(r))
    return np.mean(images, axis=0)


def asymmetry(matrix):
    """Mean |M - d4_average(M)| / mean(M). A pure-noise map still scores > 0;
    compare against the same statistic on a permutation-null map."""
    m = np.asarray(matrix, dtype=float)
    return float(np.abs(m - d4_orbit_average(m)).mean() / max(m.mean(), 1e-12))


def d4_orbits(size, radius=None):
    """Map each D4 orbit key (|dr|,|dc|) sorted -> list of (r, c) in a size x size
    window. The centre cell is excluded. `radius` restricts to a Chebyshev radius
    (radius=3 gives the inner 7x7 of a 9x9 window)."""
    centre = size // 2
    lim = centre if radius is None else radius
    orbits = {}
    for r in range(size):
        for c in range(size):
            dr, dc = r - centre, c - centre
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) > lim:
                continue
            orbits.setdefault(tuple(sorted((abs(dr), abs(dc)))), []).append((r, c))
    return dict(sorted(orbits.items(), key=lambda kv: (kv[0][0] ** 2 + kv[0][1] ** 2)))


def orbit_profile(matrix, radius=None, null_mean=0.0):
    """Per-orbit summary of an MI map, ordered by Euclidean distance.

    Pass the RAW map, not the D4 average: the orbit mean already *is* the D4 average
    for that orbit, and passing the averaged map makes `mi_std` identically zero and
    throws away the diagnostic. `mi_std` is the spread between cells that symmetry
    says must be equal, so it is a direct read on estimator noise AT THAT MI
    MAGNITUDE. Do not compare it against permutation_null()['p95']: the null is the
    magnitude of MI when there is NO signal, a different and much smaller quantity,
    so that comparison always looks like a failure even when nothing is wrong. The
    calibrated test for a sampling artefact is asymmetry() vs asymmetry_p95.

    Prefer this to a Chebyshev "ring" average: rings lump together cells at genuinely
    different distances. Empirically MI tracks Euclidean distance (Spearman -0.99)
    far better than Chebyshev (-0.89) -- the ring-3 corner (3,3) is less informative
    than the ring-4 axis cell (0,4), which a ring average cannot express.
    """
    m = np.asarray(matrix, dtype=float)
    rows = []
    for key, cells in d4_orbits(m.shape[0], radius).items():
        vals = [m[r, c] for r, c in cells]
        rows.append({
            "orbit": key,
            "n_cells": len(cells),
            "euclidean": float(np.hypot(*key)),
            "chebyshev": int(max(key)),
            "mi": float(np.mean(vals)) - null_mean,
            "mi_std": float(np.std(vals)),
        })
    return rows


def format_orbit_profile(rows):
    out = [f"{'orbit':>8} {'cells':>6} {'euclid':>7} {'cheb':>5} {'MI (bits)':>10} {'within-orbit sd':>16}"]
    for r in rows:
        out.append(f"{str(r['orbit']):>8} {r['n_cells']:>6} {r['euclidean']:>7.2f} "
                   f"{r['chebyshev']:>5} {r['mi']:>10.4f} {r['mi_std']:>16.4f}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# redundancy (for mRMR, if you keep it)
# --------------------------------------------------------------------------- #

def redundancy(flat_windows, cell_a, cell_b):
    """I(cell_a; cell_b) in bits -- same estimator as relevance, which is the whole
    point: relevance - redundancy is only meaningful if both come from the same
    estimator with the same bias characteristics."""
    return mi_plugin(flat_windows[:, cell_a], flat_windows[:, cell_b],
                     nx=N_SYMBOLS, ny=N_SYMBOLS)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "cos700_dataset_train.npz"
    with np.load(path) as d:
        windows, probs = d["windows"], d["csp_probs"]

    target = bin_target(probs)
    raw = mi_map(windows, target)
    null = permutation_null(windows, target, n_perm=50)

    print(f"n = {len(windows):,}   null mean = {null['mean']:.4f} bits   "
          f"null p95 = {null['p95']:.4f} bits")
    print(f"asymmetry: {asymmetry(raw):.3f}   (pure-noise 95th pct: {null['asymmetry_p95']:.3f})")
    print("  -> above the null means a residual sampling artefact; at or below it, "
          "just finite n.")
    print()
    # raw map, not the D4 average: keeps the within-orbit spread as a noise read
    print(format_orbit_profile(orbit_profile(raw, radius=3, null_mean=null["mean"])))
    print("\n'within-orbit sd' is the spread between cells that symmetry says must be "
          "equal, i.e. the sampling error of the MI estimate at that magnitude.\n"
          "Do NOT compare it to the permutation null -- the null is the size of MI "
          "when there is no signal, a different quantity, and the comparison will "
          "always look alarming.\nThe calibrated test for a sampling artefact is the "
          "asymmetry statistic above.")
    # d4_orbit_average(raw) is what you plot as the heatmap figure.
