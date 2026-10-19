"""
Baseline: a deterministic, rule-based agent implementing single-constraint
Minesweeper heuristics - Section 4.3's "standard heuristics utilised by
human players." Two rules, applied in order:

1. Forced mine (logically certain): if a revealed cell's number of hidden
   neighbours exactly equals its own value, every one of those hidden
   neighbours is certainly a mine. E.g. a revealed "2" with exactly two
   hidden neighbours means both of them are mines - full stop, no ambiguity.

2. Estimated risk (a heuristic, not a certainty): for any cell not caught
   by rule 1, estimate its mine probability from each touching revealed
   number by assuming that number's remaining mines are spread evenly
   across its hidden neighbours (e.g. a "2" with 4 hidden neighbours gives
   each ~25% estimated risk), taking the worst such estimate across every
   touching number, and preferring the cell with the lowest worst-case
   estimate.

This is deliberately NOT the CSP solver's exhaustive multi-constraint
reasoning (solver/csp_solver.py), which combines several revealed numbers'
constraints together to prove things these two rules alone can't (e.g. the
classic "1-2-1" pattern, or exact rather than estimated safe-cell
deduction). A handcrafted baseline is meant to represent a competent human
player applying obvious rules of thumb, not compute the exact answer - if
it used the CSP solver it wouldn't be a baseline, it would just be the
solver.

This operates on the exact same 5x5-window feature representation every
other policy_fn uses (rather than reading the live board directly), so it
stays a fair, like-for-like baseline: it "sees" precisely as much local
context as a GP tree or the Neurocontroller does. That means some
deductions a full-board human could make near a window's edge (where a
relevant revealed number falls just outside the window) aren't visible to
it - a deliberate, documented simplification, not an oversight.
"""
from __future__ import annotations

import numpy as np

HIDDEN_VALUE = 9
OUT_OF_BOUNDS_VALUE = 10
WINDOW_SIDE = 5
CENTRE = 2  # the candidate cell's own position within the 5x5 window

MINE_SCORE = -1_000.0  # forced far out of contention once ruled a mine

_NEIGHBOUR_OFFSETS = [
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)
]


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < WINDOW_SIDE and 0 <= c < WINDOW_SIDE


def _count_hidden_neighbours(window: np.ndarray, r: int, c: int) -> int:
    count = 0
    for dr, dc in _NEIGHBOUR_OFFSETS:
        rr, cc = r + dr, c + dc
        if _in_bounds(rr, cc) and window[rr, cc] == HIDDEN_VALUE:
            count += 1
    return count


def _is_forced_mine(window: np.ndarray) -> bool:
    """The candidate (window centre) is a forced mine if ANY revealed
    neighbouring number's hidden-neighbour count exactly equals that
    number - Rule 1, applied to every revealed cell touching the candidate.
    """
    for dr, dc in _NEIGHBOUR_OFFSETS:
        r, c = CENTRE + dr, CENTRE + dc
        if not _in_bounds(r, c):
            continue
        value = window[r, c]
        if 0 <= value <= 8:  # a revealed number
            if _count_hidden_neighbours(window, r, c) == value:
                return True
    return False


def _estimate_risk(window: np.ndarray) -> float | None:
    """A single-constraint mine-probability ESTIMATE (not exact, unlike the
    CSP solver): for each revealed number touching the candidate, assume
    its remaining mines are spread evenly across its hidden neighbours -
    e.g. a revealed "2" with 4 hidden neighbours gives each of them a rough
    25% estimated risk. Takes the WORST (highest) estimate across every
    touching number, since being next to even one highly-constrained
    number should dominate the decision.

    This is still purely local, single-number-at-a-time reasoning - it
    never combines evidence from multiple revealed numbers together the
    way the CSP solver does, so it stays a heuristic estimate rather than
    an exact deduction. Returns None if the candidate has no revealed
    neighbour to reason from at all (a genuinely uninformed guess).
    """
    risks = []
    for dr, dc in _NEIGHBOUR_OFFSETS:
        r, c = CENTRE + dr, CENTRE + dc
        if not _in_bounds(r, c):
            continue
        value = window[r, c]
        if 0 <= value <= 8:
            hidden_count = _count_hidden_neighbours(window, r, c)
            # hidden_count == value is the forced-mine case, already
            # handled by _is_forced_mine before this is ever called, so
            # hidden_count > value always holds here, keeping risk in (0, 1)
            if hidden_count > 0:
                risks.append(value / hidden_count)
    return max(risks) if risks else None


def _fallback_score(window: np.ndarray) -> float:
    """No revealed neighbour touches the candidate at all - there's
    nothing local to reason from, so fall back to preferring cells whose
    wider neighbourhood has a LOWER average value, a simple, explainable
    stand-in for "this area looks less mine-dense" - roughly what a human
    eyeballing the board without doing exact deduction would go on.
    Out-of-bounds padding is excluded so board edges don't skew the average.
    """
    valid = window[window != OUT_OF_BOUNDS_VALUE]
    return -float(np.mean(valid))


def _score_one(flat_window: np.ndarray) -> float:
    window = flat_window.reshape(WINDOW_SIDE, WINDOW_SIDE)
    if _is_forced_mine(window):
        return MINE_SCORE
    risk = _estimate_risk(window)
    if risk is not None:
        # risk in (0, 1) here (forced-mine case already excluded above),
        # so -risk always outranks the much more negative _fallback_score
        # -> any cell with SOME local evidence beats a totally blind guess
        return -risk
    return _fallback_score(window)


def handcrafted_policy(features: np.ndarray) -> np.ndarray:
    """features: (n_cells, 25) flattened 5x5 windows, one row per legal
    cell - the same batched signature every other policy_fn uses.
    """
    return np.array([_score_one(row) for row in features])