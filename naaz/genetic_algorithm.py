from collections.abc import Callable
import numpy as np


class GeneticAlgorithm:
    """Real-Coded Genetic Algorithm (RCGA) for continuous parameter optimization.

    Optimizes the weights and bias of a linear/logistic scoring function by minimizing the Brier score.
    """

    pop_size: int
    chrom_len: int
    rng: np.random.Generator
    fitness_fn: Callable[[np.ndarray], np.ndarray]

    population: np.ndarray
    fitness_scores: np.ndarray
    best_chromosome: np.ndarray
    best_score: float

    # Hyperparameters
    p_c: float            # Crossover probability
    p_m: float            # Mutation probability per gene
    sigma: float          # Gaussian mutation standard deviation
    k: int                # Tournament size
    n_elite: int          # Elite count preserved unchanged

    def __init__(
        self,
        population_size: int,
        feature_count: int,
        seed: int,
        fitness_fn: Callable[[np.ndarray], np.ndarray],
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.02,
        mutation_sigma: float = 0.1,
        tournament_size: int = 3,
        elitism_count: int = 2,
    ) -> None:
        self.pop_size = population_size
        self.chrom_len = feature_count
        self.rng = np.random.default_rng(seed)
        self.fitness_fn = fitness_fn

        self.p_c = crossover_rate
        self.p_m = mutation_rate
        self.sigma = mutation_sigma
        self.k = tournament_size
        self.n_elite = elitism_count

        self.population = np.empty((0, 0), dtype=np.float64)
        self.fitness_scores = np.empty(0, dtype=np.float64)
        self.best_chromosome = np.empty(0, dtype=np.float64)
        self.best_score = np.inf

    def initialize(self) -> None:
        """Initializes the population uniformly in [-1.0, 1.0] and computes initial fitness."""
        self.population = self.rng.uniform(
            low=-1.0, high=1.0, size=(self.pop_size, self.chrom_len)
        )
        self._evaluate_all()

    def _evaluate_all(self) -> None:
        """Evaluates Brier scores for the full population in a single batch."""
        self.fitness_scores = self.fitness_fn(self.population)

        min_idx = int(np.argmin(self.fitness_scores))
        current_best = float(self.fitness_scores[min_idx])
        if current_best < self.best_score:
            self.best_score = current_best
            self.best_chromosome = self.population[min_idx].copy()

    def _tournament_select(self) -> np.ndarray:
        """Selects one parent via tournament selection (k candidates; lowest Brier score wins)."""
        candidates = self.rng.integers(0, self.pop_size, size=self.k)
        winner = candidates[np.argmin(self.fitness_scores[candidates])]
        return self.population[winner]

    def iterate(self) -> None:
        """Advances the population by one generation: Elitism -> Tournament Selection -> Arithmetic Crossover -> Gaussian Mutation."""
        if len(self.population) == 0:
            raise ValueError(
                "Population is not initialized. Call initialize() first."
            )

        next_gen: list[np.ndarray] = []

        # 1. Elitism: preserve top performers unchanged
        elite_indices = np.argsort(self.fitness_scores)[: self.n_elite]
        for idx in elite_indices:
            next_gen.append(self.population[idx].copy())

        # 2. Breeding loop
        while len(next_gen) < self.pop_size:
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            # Real-Coded Arithmetic Crossover (gene-wise blending)
            if self.rng.random() < self.p_c:
                alpha = self.rng.uniform(0.0, 1.0, size=self.chrom_len)
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = (1.0 - alpha) * p1 + alpha * p2
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Gaussian Mutation
            for child in (c1, c2):
                if len(next_gen) < self.pop_size:
                    mutate_mask = (
                        self.rng.random(size=self.chrom_len) < self.p_m
                    )
                    noise = self.rng.normal(
                        0.0, self.sigma, size=self.chrom_len
                    )
                    child[mutate_mask] += noise[mutate_mask]
                    next_gen.append(child)

        self.population = np.array(next_gen, dtype=np.float64)
        self._evaluate_all()