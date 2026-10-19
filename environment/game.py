"""
A stateful, agent-driven Minesweeper game, built on top of the existing
corpus/ functions (generate_board, hide_board, reveal_section) rather than
reimplementing board logic from scratch.

This is deliberately NOT corpus/generate_dataset.py's process_seed loop:
that loop always reveals the CSP solver's exact lowest-risk cell and
silently skips over mines to keep harvesting labels for the dataset. Here,
an external policy (a GP tree, a neural net, a fixed rule, random choice)
picks the cell, and hitting a mine actually ends the game - this is what
"playing" the game means for evaluating an agent, as opposed to labelling it.
"""
from __future__ import annotations

from enum import Enum, auto

import numpy as np

from corpus.board_hider import hide_board
from corpus.generate_board import generate_board
from corpus.revealer import reveal_section
from tools.fronter_cells import get_frontier_cells

MINE_VALUE = 12
HIDDEN_VALUE = 9


class MoveResult(Enum):
    SAFE = auto()               # revealed a safe cell, game continues
    ALREADY_REVEALED = auto()   # clicked a visible cell - invalid, no-op
    MINE = auto()                # hit a mine, game over (loss)
    WIN = auto()                  # revealed the last safe cell, game over (win)


class MinesweeperGame:
    """One playthrough on a fixed, seeded board."""

    def __init__(self, width: int, height: int, mines: int, seed: int):
        self.width = width
        self.height = height
        self.mines = mines
        self.seed = seed

        self.board = generate_board(width, height, mines, seed)
        # a seeded rng, separate from generate_board's own internal one,
        # so the opening click is reproducible for this seed too - not
        # just the mine layout
        opening_rng = np.random.default_rng(seed)
        self.board_hidden = hide_board(self.board, rng=opening_rng)

        self.safe_cell_count = int(np.sum(self.board != MINE_VALUE))
        self.moves_made = 0
        self.game_over = False
        self.won = False

    def cells_revealed(self) -> int:
        return int(np.sum(self.board_hidden != HIDDEN_VALUE))

    def is_hidden(self, row: int, col: int) -> bool:
        return self.board_hidden[row][col] == HIDDEN_VALUE

    def hidden_cells(self) -> list[tuple[int, int]]:
        rows, cols = np.where(self.board_hidden == HIDDEN_VALUE)
        return list(zip(rows.tolist(), cols.tolist()))

    def frontier_cells(self) -> list[tuple[int, int]]:
        """Hidden cells bordering at least one revealed number - reuses
        tools/fronter_cells.py unchanged."""
        return list(get_frontier_cells(self.board_hidden))

    def legal_moves(self) -> list[tuple[int, int]]:
        """What the agent should choose from this turn: the frontier if
        one exists, otherwise every remaining hidden cell - an unguided
        guess is unavoidable when nothing borders a revealed number yet
        (e.g. immediately after the opening cascade closes off)."""
        frontier = self.frontier_cells()
        return frontier if frontier else self.hidden_cells()

    def step(self, row: int, col: int) -> MoveResult:
        if self.game_over:
            raise RuntimeError("step() called after the game already ended")

        if not self.is_hidden(row, col):
            return MoveResult.ALREADY_REVEALED

        self.moves_made += 1

        if self.board[row][col] == MINE_VALUE:
            self.board_hidden[row][col] = MINE_VALUE
            self.game_over = True
            self.won = False
            return MoveResult.MINE

        reveal_section(self.board, self.board_hidden, row, col)

        if self.cells_revealed() >= self.safe_cell_count:
            self.game_over = True
            self.won = True
            return MoveResult.WIN

        return MoveResult.SAFE