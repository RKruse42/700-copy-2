from os import error

import numpy as np

from corpus.generate_board import generate_board


# gets a cell value or inserts an out of bounds number
def get_value(board, row, col):
    if row < 0 or col < 0 or row>=board.shape[0] or col>=board.shape[1]:
        return 10
    return board[row][col]


# gets the sliding window of size*size centered around [row:col]
def get_sliding_window(board, row:int, col:int, size:int):
    if (size%2)!=1:
        raise ValueError("size should be odd")

    # the amount to go in any direction here
    dist = round((size - 1)/2)

    result = np.zeros((size, size))

    for r in range(row-dist, row+dist+1):
        for c in range(col-dist, col+dist+1):
            result[r-(row-dist)][c-(col-dist)] = get_value(board, r, c)

    return result
