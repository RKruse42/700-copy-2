"""
The shared harness: plays one full game with a given policy in control.
Every algorithm (GA, GP, Neurocontroller) and every baseline (random,
handcrafted) should play through this exact function so results are
directly comparable, per the proposal's cross-validation requirement
(same seeded boards for everyone).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from corpus.extract_sliding_window import get_sliding_window
from environment.game import MinesweeperGame

WINDOW_SIZE = 9
CROP_SIZE = 5
# matches the 5x5 crop richard/runner.py and simon/runner.py both train on
CROP_OFFSET = (WINDOW_SIZE - CROP_SIZE) // 2

PolicyFn = Callable[[np.ndarray], np.ndarray]  # (n_cells, 25) -> (n_cells,) scores


def extract_features(board_hidden: np.ndarray, row: int, col: int) -> np.ndarray:
    """Same feature representation used at training time (richard/runner.py,
    simon/runner.py), pulled from the live board instead of a pre-recorded
    dataset row - this is what keeps play-time and train-time comparable.
    """
    window = get_sliding_window(board_hidden, row, col, WINDOW_SIZE)
    cropped = window[CROP_OFFSET:CROP_OFFSET + CROP_SIZE, CROP_OFFSET:CROP_OFFSET + CROP_SIZE]
    return cropped.reshape(-1)


def choose_move(game: MinesweeperGame, policy_fn: PolicyFn) -> tuple[int, int]:
    """Scores every legal cell with policy_fn and returns the highest-scoring
    one. policy_fn takes a batch of feature vectors and returns a batch of
    scores - the same batched shape richard/simon fitness evaluation already uses,
    so a GP tree's evaluate() or a Neurocontroller's forward pass can be
    passed in directly.
    """
    candidates = game.legal_moves()
    features = np.stack([extract_features(game.board_hidden, r, c) for (r, c) in candidates])
    scores = policy_fn(features)
    best_index = int(np.argmax(scores))
    return candidates[best_index]


def play_game(policy_fn: PolicyFn, width: int, height: int, mines: int, seed: int,
              max_moves: int | None = None) -> dict:
    """Plays one full game start to finish, letting policy_fn pick every
    move. Returns outcome stats for a fitness function to score - this is
    the "actually play Minesweeper" counterpart to gp/runner.py's static
    Brier-score evaluation.
    """
    game = MinesweeperGame(width, height, mines, seed)
    max_moves = max_moves or (width * height)

    while not game.game_over and game.moves_made < max_moves:
        row, col = choose_move(game, policy_fn)
        game.step(row, col)

    return {
        "won": game.won,
        "cells_revealed": game.cells_revealed(),
        "safe_cell_count": game.safe_cell_count,
        "moves_made": game.moves_made,
    }