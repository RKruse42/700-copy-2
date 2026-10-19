"""
The GP's inputs: exactly the shared feature set of the paper.

The 5x5 window around the target cell, the target itself removed (24 cells),
each cell one-hot encoded over the eleven symbols (0-8 = adjacent mines,
9 = covered, 10 = off the board) -> 264 binary indicators. The layout is the
same as simon.encoding.one_hot(..., flatten=True) applied to the 24 cells
(cell-major, symbol-minor), so a GP terminal means the same thing as the
corresponding GA/neurocontroller input.

The same function is used at training time (dataset rows) and at play time
(live boards), so the two can never drift apart.
"""
import numpy as np

WINDOW_SIDE = 5
RAW_FEATURE_COUNT = WINDOW_SIDE * WINDOW_SIDE   # 25 values per window
CENTRE_INDEX = 12                               # the target cell (always covered)
N_SYMBOLS = 11
N_CELLS = RAW_FEATURE_COUNT - 1                 # 24
FEATURE_COUNT = N_CELLS * N_SYMBOLS             # 264


def _symbol_name(symbol: int) -> str:
    return "H" if symbol == 9 else ("B" if symbol == 10 else str(symbol))


def _cell_offsets() -> list[tuple[int, int]]:
    """(row offset, column offset) of the 24 cells, row-major, centre removed."""
    return [(i // WINDOW_SIDE - 2, i % WINDOW_SIDE - 2)
            for i in range(RAW_FEATURE_COUNT) if i != CENTRE_INDEX]


# e.g. "(-1,+0)=H" : the cell one row above the target is covered
#      "(+2,+1)=B" : the cell two rows below / one right is off the board
FEATURE_NAMES = [
    f"({dr:+d},{dc:+d})={_symbol_name(s)}"
    for (dr, dc) in _cell_offsets()
    for s in range(N_SYMBOLS)
]


def one_hot_features(raw_windows: np.ndarray) -> np.ndarray:
    """raw_windows: (n, 25) flattened 5x5 windows (the format every policy
    receives from environment/play.py and the dataset crop produces).
    Returns (n, 264) float64 indicators, column-contiguous so that a tree
    reading one terminal touches contiguous memory.
    """
    raw = np.asarray(raw_windows)
    cells = np.delete(raw, CENTRE_INDEX, axis=1).astype(np.intp)         # (n, 24)
    encoded = cells[:, :, None] == np.arange(N_SYMBOLS)                  # (n, 24, 11)
    return np.asfortranarray(encoded.reshape(len(cells), FEATURE_COUNT).astype(np.float64))