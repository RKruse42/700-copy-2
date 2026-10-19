"""
The tree representation itself: a Node is either a function node (an
operator with children) or wraps a Terminal (a leaf). A whole tree is just
its root Node - there's no separate Tree class, since a node and everything
below it already IS a valid, evaluatable (sub)tree.
"""
from __future__ import annotations

import numpy as np

from richard.functions import FUNCTION_SET
from richard.terminals import Terminal, random_terminal


class Node:
    __slots__ = ("func_name", "arity", "children", "terminal")

    def __init__(self, func_name: str | None = None, children: list["Node"] | None = None,
                 terminal: Terminal | None = None):
        is_function = func_name is not None
        is_terminal = terminal is not None
        if is_function == is_terminal:
            raise ValueError("Node must be exactly one of function or terminal")

        self.func_name = func_name
        self.terminal = terminal
        self.children = children or []
        self.arity = FUNCTION_SET[func_name][0] if is_function else 0

        if is_function and len(self.children) != self.arity:
            raise ValueError(f"{func_name} needs {self.arity} children, got {len(self.children)}")

    @property
    def is_terminal(self) -> bool:
        return self.terminal is not None

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """X: (n_samples, FEATURE_COUNT). Returns (n_samples,) desirability scores."""
        if self.is_terminal:
            return self.terminal.evaluate(X)
        _, fn = FUNCTION_SET[self.func_name]
        child_results = [child.evaluate(X) for child in self.children]
        return fn(*child_results)

    def depth(self) -> int:
        if self.is_terminal:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        """Total node count - used later for bloat tracking."""
        if self.is_terminal:
            return 1
        return 1 + sum(child.size() for child in self.children)

    def __repr__(self) -> str:
        if self.is_terminal:
            return repr(self.terminal)
        args = ", ".join(repr(c) for c in self.children)
        return f"{self.func_name}({args})"


def generate_tree(rng: np.random.Generator, method: str, max_depth: int, depth: int = 0) -> Node:
    """Recursively builds one random tree.

    method='full': every branch goes all the way to max_depth before adding
        a terminal -> bushy, uniform-depth trees.
    method='grow': a terminal can be chosen at any depth (with growing
        probability as depth increases) -> irregular, varied-shape trees.
    """
    function_names = list(FUNCTION_SET.keys())

    at_max_depth = depth >= max_depth
    # 'grow' can stop early; 'full' only stops once forced to by max_depth
    stop_here = at_max_depth or (method == "grow" and depth > 0 and rng.random() < 0.3)

    if stop_here:
        return Node(terminal=random_terminal(rng))

    func_name = function_names[rng.integers(0, len(function_names))]
    arity, _ = FUNCTION_SET[func_name]
    children = [generate_tree(rng, method, max_depth, depth + 1) for _ in range(arity)]
    return Node(func_name=func_name, children=children)


def ramped_half_and_half(rng: np.random.Generator, population_size: int,
                          min_depth: int = 2, max_depth: int = 6) -> list[Node]:
    """Standard GP initialisation: spreads the population evenly across
    depths min_depth..max_depth, and evenly across 'full'/'grow' at each
    depth, so the starting population has diverse tree shapes and sizes
    rather than population_size near-identical trees.
    """
    population: list[Node] = []
    depth_range = list(range(min_depth, max_depth + 1))

    for i in range(population_size):
        depth = depth_range[i % len(depth_range)]
        method = "full" if (i // len(depth_range)) % 2 == 0 else "grow"
        population.append(generate_tree(rng, method, depth))

    return population