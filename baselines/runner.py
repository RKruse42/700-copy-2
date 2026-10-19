"""
Runs both baseline agents through the same play_game harness every other
algorithm uses, across a fixed seed set, and prints summary stats - the
numbers GP (and later GA/Neurocontroller) results need to beat.
Run with:
    python3 -m baselines.runner
"""
import numpy as np

from baselines.handcrafted_agent import handcrafted_policy
from baselines.random_agent import make_random_policy
from environment.play import play_game

WIDTH, HEIGHT, MINES = 8, 8, 10
SEEDS = list(range(30))


def summarise(name: str, policy_fn) -> None:
    results = [play_game(policy_fn, WIDTH, HEIGHT, MINES, seed) for seed in SEEDS]
    win_rate = np.mean([r["won"] for r in results])
    avg_progress = np.mean([r["cells_revealed"] / r["safe_cell_count"] for r in results])
    avg_moves = np.mean([r["moves_made"] for r in results])
    print(f"{name}: win rate {win_rate:.1%}, avg board cleared {avg_progress:.1%}, "
          f"avg moves {avg_moves:.1f}  (over {len(SEEDS)} seeds)")


if __name__ == "__main__":
    print(f"Board: {WIDTH}x{HEIGHT}, {MINES} mines, {len(SEEDS)} seeds\n")
    summarise("Random agent     ", make_random_policy(np.random.default_rng(0)))
    summarise("Handcrafted agent", handcrafted_policy)