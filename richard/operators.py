"""

Node is effectively immutable (nothing is set after __init__), so these
operators never mutate an existing tree in place - they always build and
return a brand new Node. That means subtrees can be safely shared between
parents and children without risk of one individual's later mutation
corrupting another's structure.
"""
from __future__ import annotations

import numpy as np

from richard.node import Node, generate_tree

MAX_TREE_DEPTH = 12  # bloat control: no individual may exceed this


def tournament_selection(rng: np.random.Generator, population: list[Node],
                          fitnesses: list[float], k: int = 3) -> Node:
    """Randomly samples k individuals, returns the fittest of that subset.
    Lower fitness (Brier score) is better, so we minimise.
    """
    candidate_idx = rng.integers(0, len(population), size=k)
    best = min(candidate_idx, key=lambda i: fitnesses[i])
    return population[best]


def _collect_nodes(node: Node, out: list[Node]) -> None:
    """Pre-order traversal: root first, then children left-to-right.
    Used to build a flat, indexable list of every node in the tree so we
    can pick a uniformly random "cut point" for crossover/mutation.
    """
    out.append(node)
    for child in node.children:
        _collect_nodes(child, out)


def get_all_nodes(root: Node) -> list[Node]:
    nodes: list[Node] = []
    _collect_nodes(root, nodes)
    return nodes


def replace_subtree(root: Node, index: int, replacement: Node) -> Node:
    """Rebuilds the tree with the node at the given pre-order index swapped
    out for `replacement`. Every ancestor on the path to that node gets
    reconstructed (new Node objects); everything else is untouched.
    """
    counter = [0]

    def _rebuild(node: Node) -> Node:
        current_index = counter[0]
        counter[0] += 1
        if current_index == index:
            return replacement
        if node.is_terminal:
            return node
        new_children = [_rebuild(child) for child in node.children]
        return Node(func_name=node.func_name, children=new_children)

    return _rebuild(root)


def subtree_crossover(rng: np.random.Generator, parent1: Node, parent2: Node,
                       max_depth: int = MAX_TREE_DEPTH, max_attempts: int = 5) -> Node:
    """Picks a random subtree in parent1 and replaces it with a random
    subtree from parent2. Retries with a different cut point if the result
    would exceed max_depth; if it still can't find a valid combination,
    parent1 is returned unchanged (crossover simply has no effect that time).
    """
    nodes1 = get_all_nodes(parent1)
    nodes2 = get_all_nodes(parent2)

    for _ in range(max_attempts):
        idx1 = int(rng.integers(0, len(nodes1)))
        idx2 = int(rng.integers(0, len(nodes2)))
        child = replace_subtree(parent1, idx1, nodes2[idx2])
        if child.depth() <= max_depth:
            return child

    return parent1


def subtree_mutation(rng: np.random.Generator, tree: Node,
                      max_depth: int = MAX_TREE_DEPTH, new_subtree_depth: int = 3,
                      max_attempts: int = 5) -> Node:
    """Picks a random subtree in `tree` and replaces it with a freshly
    generated random subtree, introducing variation the population didn't
    already have. Same depth-cap retry behaviour as crossover.
    """
    nodes = get_all_nodes(tree)

    for _ in range(max_attempts):
        idx = int(rng.integers(0, len(nodes)))
        new_subtree = generate_tree(rng, "grow", new_subtree_depth)
        mutated = replace_subtree(tree, idx, new_subtree)
        if mutated.depth() <= max_depth:
            return mutated

    return tree