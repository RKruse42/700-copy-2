import numpy as np
from corpus.revealer import reveal_section
from corpus.generate_board import generate_board
# takes a board, and hides all of it except the one section that is revealed, returning
# the newly "hidden" board
def hide_board(board, rng=None):
    # rng: an np.random.Generator (e.g. np.random.default_rng(seed)) for a
    # reproducible opening click. Defaults to the global np.random state
    # (old behaviour) for backward compatibility with existing callers.
    if rng is None:
        rng = np.random

    (rows, cols) = board.shape

    # 9 means hidden
    board_all_hidden = np.full((rows, cols), 9)

    safe_spots = np.argwhere(board == 0)
    if len(safe_spots) == 0:
        raise ValueError("board has no zero-cell to open on")

    r_safe, c_safe = safe_spots[rng.integers(len(safe_spots))]
    reveal_section(board=board, board_hidden=board_all_hidden, row=r_safe, col=c_safe)
    return board_all_hidden