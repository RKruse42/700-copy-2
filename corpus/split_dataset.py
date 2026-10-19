import sys
import numpy as np

# splits 70/15/15
def split_dataset(input_file, seed=0):
    data = np.load(input_file)
    seeds = data['seeds']
    games = np.random.default_rng(seed).permutation(np.unique(seeds))
    n = len(games)
    train_g, val_g = games[:round(.7*n)], games[round(.7*n):round(.85*n)]
    test_g = games[round(.85*n):]
    for name, g in [("train", train_g), ("val", val_g), ("test", test_g)]:
        m = np.isin(seeds, g)
        np.savez_compressed(f"cos700_dataset_{name}.npz", **{k: data[k][m] for k in data})


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "cos700_dataset.npz"
    split_dataset(src)
