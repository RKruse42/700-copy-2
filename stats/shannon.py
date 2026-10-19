from math import log2

import numpy as np

def calculate_shannon(prob_true):
    return -(prob_true*log2(prob_true) + (1-prob_true)*log2(1-prob_true))

def get_shannon_entropy(filename):
    with np.load(filename) as data:
        is_mine_column = data['is_mine']

    count = 0

    for val in is_mine_column:
        if val:
            count += 1

    print("Total length: " + str(len(is_mine_column)))
    print("Mine count: " + str(count))

    prob_mine = count/len(is_mine_column)

    shannon_value = calculate_shannon(prob_mine)
    print("Shannon: " + str(round(shannon_value, 4)))
    return shannon_value
