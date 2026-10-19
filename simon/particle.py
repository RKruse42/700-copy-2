import numpy as np


class Particle:
    positions: np.ndarray
    local_best_positions: np.ndarray
    velocities: np.ndarray
    best_score: float = np.inf

    # inertia weight
    w: float = 0.7
    # cognitive coefficient
    c1: float = 1.5
    # social coefficient
    c2: float = 1.5

    def __init__(self, feature_count: int, rng: np.random.Generator) -> None:
        self.positions = rng.uniform(low=-1.0, high=1.0, size=feature_count)
        self.local_best_positions = self.positions.copy()
        self.velocities = rng.uniform(low=-1.0, high=1.0, size=feature_count)

    def update_best(self, current_score: float) -> None:
        if current_score < self.best_score:
            self.best_score = current_score
            self.local_best_positions = self.positions.copy()

    def update(self, global_best_positions: np.ndarray, rng: np.random.Generator):
        r1 = rng.random(size=len(self.positions))
        r2 = rng.random(size=len(self.positions))
        cognitive = self.c1 * r1 * (self.local_best_positions - self.positions)
        social = self.c2 * r2 * (global_best_positions - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social
        self.positions = self.positions + self.velocities

    def get_value(self) -> np.ndarray:
        return self.positions
