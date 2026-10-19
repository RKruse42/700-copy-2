from collections.abc import Callable

import numpy as np

from simon.particle import Particle


class PSO:
    """Basic PSO algorithm"""

    particle_count: int
    feature_count: int
    rng: np.random.Generator
    particles: list[Particle]
    best_position: np.ndarray
    best_score: float
    f: Callable[[np.ndarray], float]

    def __init__(self, particle_count: int, feature_count: int, seed: int,  f: Callable[[np.ndarray], float]) -> None:
        self.particle_count = particle_count
        self.feature_count = feature_count
        self.rng = np.random.default_rng(seed)
        self.f = f
        self.particles = []
        self.best_position = np.array([])
        self.best_score = np.inf

    def initialize(self):
        self.particles = [Particle(self.feature_count, self.rng)
                          for _ in range(self.particle_count)]
        self._evaluate_all()

    def _evaluate_all(self):
        positions = np.stack([p.positions for p in self.particles])   # (P, D)
        scores = self.f(positions)
        for particle, score in zip(self.particles, scores):
            particle.update_best(score)
            if score < self.best_score:
                self.best_score = score
                self.best_position = particle.positions.copy()

    def iterate(self):
        if len(self.best_position) == 0:
            raise ValueError("No global best particle! Did you run initialize?")
        for particle in self.particles:
            particle.update(self.best_position, self.rng)
        self._evaluate_all()
