"""
The GP evolution loop (dataset fitness).

Model selection follows the paper's protocol for the neurocontroller: at every
generation the population's best individual (by TRAINING Brier) is scored on
the validation split, and the individual with the lowest VALIDATION Brier over
the whole run is the one returned. A run that keeps going past its best
generation therefore cannot return a worse model (this plays the role of the
early stopping used by the GA and the neurocontroller).

Run a single training run with:
    python3 -m richard.runner
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from richard.fitness import FitnessFn
from richard.node import Node, ramped_half_and_half
from richard.operators import MAX_TREE_DEPTH, subtree_crossover, subtree_mutation, tournament_selection


@dataclass
class RunResult:
    tree: Node               # the kept individual (lowest validation Brier)
    train_brier: float       # its training Brier
    val_brier: float         # its validation Brier
    generation: int          # generation it was found in (0 = initial population)
    history: list = field(default_factory=list)   # (generation, train, val) of each generation's best


def evaluate_population(population: list[Node], fitness_fn: FitnessFn) -> list[float]:
    return [fitness_fn(tree) for tree in population]


def next_generation(rng: np.random.Generator, population: list[Node], fitnesses: list[float],
                    tournament_size: int = 3, crossover_rate: float = 0.8,
                    mutation_rate: float = 0.15, max_depth: int = MAX_TREE_DEPTH) -> list[Node]:
    """Elitism (the single best individual is copied unchanged), then tournament
    selection, subtree crossover (probability crossover_rate, otherwise the
    parent is copied) and subtree mutation (probability mutation_rate)."""
    population_size = len(population)
    best_idx = int(np.argmin(fitnesses))
    new_population = [population[best_idx]]

    while len(new_population) < population_size:
        parent1 = tournament_selection(rng, population, fitnesses, tournament_size)

        if rng.random() < crossover_rate:
            parent2 = tournament_selection(rng, population, fitnesses, tournament_size)
            child = subtree_crossover(rng, parent1, parent2, max_depth)
        else:
            child = parent1

        if rng.random() < mutation_rate:
            child = subtree_mutation(rng, child, max_depth)

        new_population.append(child)

    return new_population


def run(train_fitness_fn: FitnessFn, val_fitness_fn: FitnessFn, population_size: int,
        generations: int, rng: np.random.Generator, tournament_size: int = 3,
        crossover_rate: float = 0.8, mutation_rate: float = 0.15,
        max_depth: int = MAX_TREE_DEPTH, init_min_depth: int = 2, init_max_depth: int = 6,
        verbose: bool = True) -> RunResult:
    population = ramped_half_and_half(rng, population_size, init_min_depth, init_max_depth)
    fitnesses = evaluate_population(population, train_fitness_fn)

    best: RunResult | None = None
    history: list = []
    last_tree, last_val = None, None      # an elite survives unchanged: do not re-validate it

    for generation in range(generations + 1):
        if generation > 0:
            population = next_generation(rng, population, fitnesses, tournament_size,
                                         crossover_rate, mutation_rate, max_depth)
            fitnesses = evaluate_population(population, train_fitness_fn)

        i = int(np.argmin(fitnesses))
        candidate = population[i]
        val = last_val if candidate is last_tree else val_fitness_fn(candidate)
        last_tree, last_val = candidate, val
        history.append((generation, fitnesses[i], val))

        if best is None or val < best.val_brier:
            best = RunResult(candidate, fitnesses[i], val, generation)

        if verbose:
            print(f"Generation {generation}: train {fitnesses[i]:.4f} val {val:.4f} "
                  f"(kept: val {best.val_brier:.4f} from generation {best.generation})")

    best.history = history
    return best


if __name__ == "__main__":
    from richard.best_config import BEST_CONFIG, IS_TUNED
    from richard.fitness import load_split, make_dataset_fitness_fn

    if not IS_TUNED:
        print("WARNING: best_config.json not found - using untuned default hyperparameters. "
              "Run `python3 -m richard.tune` first.")

    X_train, y_train = load_split("cos700_dataset_train.npz", 50_000)
    X_val, y_val = load_split("cos700_dataset_val.npz")
    result = run(make_dataset_fitness_fn(X_train, y_train), make_dataset_fitness_fn(X_val, y_val),
                 rng=np.random.default_rng(1), **BEST_CONFIG)
    print(f"Kept tree: val Brier {result.val_brier:.4f}, generation {result.generation}, "
          f"depth {result.tree.depth()}, size {result.tree.size()}")
    print(result.tree)