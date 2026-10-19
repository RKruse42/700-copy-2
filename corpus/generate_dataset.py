import os
import sys
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from corpus.extract_sliding_window import get_sliding_window
from corpus.generate_board import generate_playable_board as generate_board
from corpus.board_hider import hide_board
from corpus.revealer import reveal_section
from tools.constraint_builder import build_constraints
from solver.csp_solver import calculate_csp_probabilities

WIDTH, HEIGHT, MINES = 16, 16, 40
WINDOW = 9                      # keep 9x9 even though the analysis is 7x7 --
                                # you can mask a window down, you cannot mask it up
SAMPLING = "reservoir"


def process_seed(seed, sampling=SAMPLING, window=WINDOW):
    rng = np.random.default_rng(seed)
    board = generate_board(WIDTH, HEIGHT, MINES, seed)
    board_hidden = hide_board(board, rng)

    records = {}            # cell -> dict, the sample currently held for that cell
    n_seen = {}             # cell -> how many turns it has been on the frontier
    played_on = {}          # cell -> the turn it was cleared
    rows = []               # only used by sampling="all"
    turn = 0

    while True:
        frontier, constraints = build_constraints(board_hidden)
        if not frontier or not constraints:
            break

        probabilities, total_states = calculate_csp_probabilities(frontier, constraints)
        if not probabilities or total_states == 0:
            break

        # -- choose the cell to clear: lowest risk, ties broken at random (D4) ----
        keys = list(probabilities)
        rng.shuffle(keys)
        ordered = sorted(keys, key=lambda k: probabilities[k])
        played = next((c for c in ordered if board[c[0]][c[1]] != 12), None)

        # -- record the frontier (D3) --------------------------------------------
        for cell in probabilities:
            r, c = cell
            n_seen[cell] = n_seen.get(cell, 0) + 1

            if sampling == "first" and n_seen[cell] > 1:
                continue
            # reservoir sampling of size 1: keep turn t with probability 1/n_seen,
            # which leaves every turn the cell was on the frontier equally likely
            if sampling == "reservoir" and rng.random() >= 1.0 / n_seen[cell]:
                continue

            sample = {
                "seed": seed,
                "turn": turn,
                "coord": cell,
                "window": get_sliding_window(board_hidden, r, c, window),
                "csp_prob": probabilities[cell],
                "is_mine": bool(board[r][c] == 12),
            }
            if sampling == "all":
                rows.append(sample)
            else:
                records[cell] = sample

        if played is None:          # every frontier cell is a proven mine
            break
        played_on[played] = turn
        reveal_section(board, board_hidden, *played)
        turn += 1

    out = rows if sampling == "all" else list(records.values())
    for s in out:
        cell = s["coord"]
        s["n_seen"] = n_seen[cell]
        s["was_played"] = cell in played_on
        s["played_turn"] = played_on.get(cell, -1)
    return out


def generate(total_seeds, output, sampling=SAMPLING, processes=None):
    processes = processes or os.cpu_count() or 4
    chunk = max(1, total_seeds // (processes * 4))

    collected = []
    work = [(s, sampling, WINDOW) for s in range(total_seeds)]
    with Pool(processes=processes) as pool:
        with tqdm(total=total_seeds, desc=f"games ({sampling})", unit="game") as bar:
            for result in pool.imap_unordered(_worker, work, chunksize=chunk):
                collected.extend(result)
                bar.update(1)
                bar.set_postfix({"samples": len(collected)})

    # imap_unordered returns games in completion order, so sort to make the saved
    # file a pure function of (total_seeds, sampling) -- this is what lets you assert
    # byte-identical output across runs.
    collected.sort(key=lambda s: (s["seed"], s["turn"], s["coord"]))

    print(f"\n{len(collected):,} samples from {total_seeds:,} games "
          f"({len(collected)/max(total_seeds,1):.1f} per game)")

    np.savez_compressed(
        output,
        seeds=np.array([s["seed"] for s in collected], dtype=np.int32),
        turns=np.array([s["turn"] for s in collected], dtype=np.int16),
        coords=np.array([s["coord"] for s in collected], dtype=np.int8),
        windows=np.array([s["window"] for s in collected], dtype=np.int8),
        csp_probs=np.array([s["csp_prob"] for s in collected], dtype=np.float64),
        is_mine=np.array([s["is_mine"] for s in collected], dtype=bool),
        was_played=np.array([s["was_played"] for s in collected], dtype=bool),
        played_turn=np.array([s["played_turn"] for s in collected], dtype=np.int16),
        n_seen=np.array([s["n_seen"] for s in collected], dtype=np.int16),
    )
    print(f"saved {output}")
    return collected


def _worker(args):
    return process_seed(*args)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    mode = sys.argv[2] if len(sys.argv) > 2 else SAMPLING
    out = sys.argv[3] if len(sys.argv) > 3 else f"cos700_dataset.npz"
    generate(n, out, sampling=mode)
