"""
Held-out scoring for sliding-window feature sets.

Why this exists
---------------
tools/theoretical_maximum.py keys a table on the literal tuple of cell values and
grades itself on the same data it fitted. With an 11-symbol alphabet, N cells give up
to 11^N patterns; once that exceeds the sample count every row is its own pattern,
P(mine | pattern) collapses to 0 or 1, and the tool reports perfect prediction. It is
measuring memorisation, not information, and it has no maximum -- so it cannot produce
an elbow.

This module fits the same table on TRAIN and grades it on VAL. Unseen patterns fall
back to the prior. The resulting score rises, peaks and falls, and the peak is the
feature-set size worth using.

`information_gain` is a valid lower bound on I(features; label): it is exactly 0 for
the empty feature set and can never be inflated by adding junk features.

Run BOTH scorers and report both curves
---------------------------------------
They are lower bounds with opposite failure modes, and the truth sits above both:

  table     sees arbitrary interactions between cells (it is a full joint lookup), so
            it is the one that can detect synergy -- but it has no capacity. Past
            roughly log_11(n) cells most validation patterns are unseen and the score
            collapses to 0. Where it collapses scales with n, so quote the sample size.

  logistic  is additive in the one-hot cells, so it CANNOT represent a 1-2-1 pattern
            and will understate joint information -- but it keeps working at large N.

Measured on a 3.3k-row training split, the table peaked at 4 cells (0.234 bits) while
logistic was still climbing at 32 cells (0.455 bits). Reporting only the table would
have put the elbow at 4 and called it a property of Minesweeper, when it was a
property of the lookup table's sample efficiency. Use logistic for the headline elbow,
the table for the small-N region and for the forward-vs-backward synergy diagnostic.
"""

import numpy as np

N_SYMBOLS = 11
_EPS = 1e-12


def binary_entropy(p):
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def _dense_ids(X, cells):
    """Map each row's pattern over `cells` to a dense integer id.

    Base-11 positional encoding, compressed back to dense ids before it can overflow
    int64. O(n*k) and roughly 50x faster than np.unique(axis=0), which matters because
    the selectors call this ~1200 times.
    """
    key = np.zeros(len(X), dtype=np.int64)
    for c in cells:
        key = key * N_SYMBOLS + X[:, c]
        if key.max(initial=0) > (1 << 55):
            key = np.unique(key, return_inverse=True)[1]
    return np.unique(key, return_inverse=True)[1]


class HeldOutScorer:
    """Scores a set of window cells by how well a model fitted on train predicts val.

    cells are FLAT indices into the window (row * width + col).

    model="table" (default)
        Laplace-smoothed lookup table. It is the empirical Bayes-optimal predictor
        for a discrete feature set, has no hyperparameters, and is fast enough to sit
        inside a selection loop. Use it for forward selection and for the elbow curve.
        It cannot represent large feature sets: past ~10 cells most validation
        patterns are unseen and the score collapses toward 0.

    model="logistic"
        Multinomial logistic regression on one-hot cells. Degrades gracefully instead
        of collapsing, so it is the one to use when scoring large sets -- in
        particular for backward elimination, which starts from everything. Costs a
        model fit per call; use `max_train` to subsample.
    """

    def __init__(self, train_windows, train_y, val_windows, val_y,
                 alpha=1.0, model="table", max_train=None, seed=0):
        self.Xtr = np.asarray(train_windows).reshape(len(train_windows), -1).astype(np.int64)
        self.Xva = np.asarray(val_windows).reshape(len(val_windows), -1).astype(np.int64)
        self.ytr = np.asarray(train_y).astype(np.float64)
        self.yva = np.asarray(val_y).astype(np.float64)

        if max_train is not None and len(self.ytr) > max_train:
            idx = np.random.default_rng(seed).choice(len(self.ytr), max_train, replace=False)
            self.Xtr, self.ytr = self.Xtr[idx], self.ytr[idx]

        self.alpha = alpha
        self.model = model
        self.n_features = self.Xtr.shape[1]

        self.prior = float(self.ytr.mean())
        self.ce_prior = self._cross_entropy(np.full(len(self.yva), self.prior))
        self.h_val = binary_entropy(self.yva.mean())
        self._cache = {}

    # -- internals ---------------------------------------------------------- #

    def _cross_entropy(self, p):
        p = np.clip(p, _EPS, 1 - _EPS)
        return float(np.mean(-(self.yva * np.log2(p) + (1 - self.yva) * np.log2(1 - p))))

    def _onehot(self, X, cells):
        from scipy.sparse import csr_matrix
        n = len(X)
        # rows and cols must use the SAME ordering.
        # np.concatenate groups by cell (all n values for cell 0, then cell 1, ...),
        # so rows must also group by cell: tile gives [0,1,...,n-1] repeated len(cells)
        # times, NOT repeat which groups by sample.
        rows = np.tile(np.arange(n), len(cells))
        cols = np.concatenate([j * N_SYMBOLS + X[:, c] for j, c in enumerate(cells)])
        return csr_matrix((np.ones(len(rows), dtype=np.float32),
                           (rows, cols)),
                          shape=(n, len(cells) * N_SYMBOLS))

    def _predict_logistic(self, cells):
        from sklearn.linear_model import LogisticRegression

        y = (self.ytr > 0.5).astype(int)
        if y.min() == y.max():
            return np.full(len(self.yva), self.prior), np.zeros(len(self.yva), bool)
        # lbfgs with sparse one-hot is ~10x faster than dense lbfgs and converges
        # reliably at this dataset size; saga is designed for millions of rows and
        # undershoots here within 150 iterations.
        clf = LogisticRegression(solver="lbfgs", max_iter=200, C=1.0, tol=1e-3)
        clf.fit(self._onehot(self.Xtr, cells), y)
        p = clf.predict_proba(self._onehot(self.Xva, cells))[:, 1]
        return p, np.zeros(len(self.yva), bool)

    def _predict(self, cells):
        """P(mine | features) for every val row, fitted on train only."""
        if self.model == "logistic":
            return self._predict_logistic(cells)

        both = np.concatenate([self.Xtr, self.Xva], axis=0)
        ids = _dense_ids(both, cells)
        itr, iva = ids[:len(self.Xtr)], ids[len(self.Xtr):]
        k = int(ids.max()) + 1

        counts = np.bincount(itr, minlength=k).astype(np.float64)
        mines = np.bincount(itr, weights=self.ytr, minlength=k)

        table = (mines + self.alpha) / (counts + 2 * self.alpha)
        table[counts == 0] = self.prior          # pattern never seen in training
        return table[iva], counts[iva] == 0

    # -- public ------------------------------------------------------------- #

    def score(self, cells):
        """Information gain in bits: ce_prior - held-out cross-entropy.

        0 for the empty set. Negative means the feature set actively hurts, which is
        what over-fitting looks like from the outside.
        """
        cells = tuple(sorted(cells))
        if cells in self._cache:
            return self._cache[cells]
        if not cells:
            value = 0.0
        else:
            p, _ = self._predict(cells)
            value = self.ce_prior - self._cross_entropy(p)
        self._cache[cells] = value
        return value

    def brier(self, cells):
        if not cells:
            return float(np.mean((self.yva - self.prior) ** 2))
        p, _ = self._predict(cells)
        return float(np.mean((self.yva - p) ** 2))

    def unseen_fraction(self, cells):
        """Share of val rows whose pattern never occurred in training. Above ~30 %
        the score is mostly measuring the prior fallback, not the features."""
        if not cells:
            return 0.0
        _, unseen = self._predict(cells)
        return float(unseen.mean())

    def plugin_score(self, cells):
        """The self-graded estimate, for comparison only. Fits and evaluates on the
        same rows, so it climbs monotonically to H(Y) and never peaks. Reproduce the
        saturation table with this and cite it as the reason for the change."""
        if not cells:
            return 0.0
        X = np.concatenate([self.Xtr, self.Xva], axis=0)
        y = np.concatenate([self.ytr, self.yva])
        ids = _dense_ids(X, cells)
        k = int(ids.max()) + 1
        counts = np.bincount(ids, minlength=k).astype(np.float64)
        mines = np.bincount(ids, weights=y, minlength=k)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(counts > 0, mines / np.maximum(counts, 1), 0.0)
        h_cond = float(np.sum(counts / len(ids) * np.array([binary_entropy(v) for v in p])))
        return binary_entropy(y.mean()) - h_cond


def saturation_table(scorer, ordered_cells, max_n=12):
    """Reproduce the plug-in vs held-out comparison. Goes straight into the report as
    the justification for abandoning the plug-in criterion."""
    rows = []
    for n in range(1, max_n + 1):
        cells = ordered_cells[:n]
        rows.append({
            "n": n,
            "plugin": scorer.plugin_score(cells),
            "heldout": scorer.score(cells),
            "unseen": scorer.unseen_fraction(cells),
        })
    return rows


def format_saturation_table(rows):
    out = [f"{'N':>3} {'plug-in (self-graded)':>22} {'held-out':>10} {'unseen val patterns':>21}"]
    for r in rows:
        out.append(f"{r['n']:>3} {r['plugin']:>22.4f} {r['heldout']:>10.4f} {100*r['unseen']:>20.1f}%")
    return "\n".join(out)


def load_split(prefix="cos700_dataset", target="is_mine"):
    """Returns (train_windows, train_y, val_windows, val_y). `target` is 'is_mine'
    (ground truth -- use for the final curves) or 'csp_prob' (the denoised oracle --
    lower variance, use when selecting)."""
    out = []
    for split in ("train", "val"):
        with np.load(f"{prefix}_{split}.npz") as d:
            out.append(d["windows"])
            out.append(d[target if target != "csp_prob" else "csp_probs"].astype(float))
    return tuple(out)


if __name__ == "__main__":
    import sys

    prefix = sys.argv[1] if len(sys.argv) > 1 else "cos700_dataset"
    Xtr, ytr, Xva, yva = load_split(prefix, target="is_mine")
    s = HeldOutScorer(Xtr, ytr, Xva, yva)

    size = Xtr.shape[1]
    centre = size // 2
    # nearest-first ordering, purely for the demonstration
    order = [r * size + c for r, c in
             sorted(((r, c) for r in range(size) for c in range(size) if (r, c) != (centre, centre)),
                    key=lambda rc: (rc[0] - centre) ** 2 + (rc[1] - centre) ** 2)]

    print(f"train {len(ytr):,}  val {len(yva):,}  H(Y) = {s.h_val:.4f} bits\n")
    print(format_saturation_table(saturation_table(s, order, max_n=12)))
