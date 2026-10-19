import numpy as np

# this reveals a part of the minesweeper board, similar to how it would work
# when an actual player "clicks" a cell and it reveals somewhat
def reveal_section(board, board_hidden, row: int, col: int):
    # list of cells to check for expansion
    to_reveal = [(row,col)]
    checked = set()

    while len(to_reveal)>0:
        (r, c) = to_reveal.pop(0)

        # skip items outside the bounds of the board
        if r < 0 or c < 0 or r >= board.shape[0] or c >= board.shape[1] or (r, c) in checked:
            continue

        val = board[r][c]

        # if val == 0 it's an empty cell and we need to "fill" outwards
        if val == 0:
            # top row
            to_reveal.append((r-1,c-1))
            to_reveal.append((r-1,c))
            to_reveal.append((r-1,c+1))

            # left and right
            to_reveal.append((r,c-1))
            to_reveal.append((r,c+1))

            # bottom row
            to_reveal.append((r+1,c-1))
            to_reveal.append((r+1,c))
            to_reveal.append((r+1,c+1))


        # reveal on board_hidden
        board_hidden[r][c] = board[r][c]
        checked.add((r, c))
