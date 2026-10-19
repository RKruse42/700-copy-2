"""
The GP function set: every possible "internal node" a tree can use.

Kept deliberately small and closed (every function takes floats/arrays and
returns a float/array) so any combination of functions and terminals produces
a valid, evaluatable tree - no type errors, no crashes on generated trees.
"""
import numpy as np


def protected_div(a, b):
    """Standard GP fix for a/0: division that never blows up.

    Works elementwise whether a, b are scalars or numpy arrays.
    """
    b = np.asarray(b, dtype=np.float64)
    safe_b = np.where(np.abs(b) < 1e-6, 1.0, b)
    return np.where(np.abs(b) < 1e-6, 1.0, a / safe_b)


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def min2(a, b):
    return np.minimum(a, b)


def max2(a, b):
    return np.maximum(a, b)


def if_lte(a, b, c, d):
    """if a <= b then c else d -- gives trees a way to branch on comparisons,
    e.g. "if this neighbour's value <= 0 then ... else ...".
    """
    return np.where(a <= b, c, d)


# name -> (arity, callable). Arity is how many children the node needs.
FUNCTION_SET = {
    "+": (2, add),
    "-": (2, sub),
    "*": (2, mul),
    "/": (2, protected_div),
    "min": (2, min2),
    "max": (2, max2),
    "if_lte": (4, if_lte),
}