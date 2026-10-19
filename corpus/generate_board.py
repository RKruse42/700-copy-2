import numpy as np


# generates a minesweeper game board, duh
def generate_board(width: int, height: int, mines: int, seed: int, verbose:bool=False):
    # set seed
    rng = np.random.default_rng(seed)

    # generate list of width*height (our 2d mine field in a single list)
    zeros_array = np.zeros((height,width))

    if verbose:
        print("Empty Field:")
        print(zeros_array)

    # select mine indices
    flat_indices = rng.choice(width*height, size=mines, replace=False)

    mines_array = zeros_array.copy()

    for index in flat_indices:
        col = index % width
        row = round((index-col)/width)
        mines_array[row][col] = 12


    if verbose:
        print("With Mines:")
        print(mines_array)

    # calculate cell values
    for row in range(height):
        for col in range(width):
            if mines_array[row][col] == 12:
                continue

            mine_count = 0

            # top row
            if row>0:
                # top left
                if col > 0 and mines_array[row-1][col-1] == 12:
                    mine_count += 1
                # top center
                if mines_array[row-1][col] == 12:
                    mine_count += 1
                # top right
                if col < width-1 and mines_array[row-1][col+1] == 12:
                    mine_count += 1

            # left
            if col > 0 and mines_array[row][col-1] == 12:
                mine_count += 1

            # right
            if col < width-1 and mines_array[row][col+1] == 12:
                mine_count += 1

            # bottom row
            if row<height-1:
                # top left
                if col > 0 and mines_array[row+1][col-1] == 12:
                    mine_count += 1
                # top center
                if mines_array[row+1][col] == 12:
                    mine_count += 1
                # top right
                if col < width-1 and mines_array[row+1][col+1] == 12:
                    mine_count += 1

            mines_array[row][col] = mine_count

    if verbose:
        print("After calculations:")
        print(mines_array)
    return mines_array

def generate_playable_board(width, height, mines, seed, max_retry=1000):
    if mines >= width * height:
        raise ValueError(f"{mines} mines do not fit in a {width}x{height} board")
    for attempt in range(max_retry):
        board = generate_board(width, height, mines, seed + attempt * 1_000_003)
        if (board == 0).any():
            return board
    raise ValueError(f"no playable board for seed {seed} after {max_retry} retries")


def generate_boards(width: int, height: int, mines: int, seed_start: int, count: int, filename: str):
    boards = []
    for i in range(count):
        seed = seed_start + i
        boards.append(generate_board(width=width, height=height, mines=mines, seed=seed))

    np.savez_compressed(filename, boards=boards, seeds=np.arange(seed_start, seed_start+count))


# generate_boards(8, 8, 20, 0, 1000, 'boards.npz')
