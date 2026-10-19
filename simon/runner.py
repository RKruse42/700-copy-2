import numpy as np

from simon.encoding import one_hot
from simon.neural_network import NeuralNetwork
from simon.pso import PSO
from stats.brier import brier_np

print(f"Loading dataset from cos700_dataset_train.npz...")
with np.load('cos700_dataset_train.npz') as data:
    seeds = data['seeds']
    windows = data['windows']
    csp_probs = data['csp_probs']
    is_mine = data['is_mine']
    coords = data['coords']

# extract a 5x5 window as the input features
windows_5x5 = windows[:, 2:7, 2:7].reshape(len(windows), 25)

batch = list(zip(windows_5x5[:50000], csp_probs[:50000]))

with np.load('cos700_dataset_val.npz') as data:
    val_windows = data['windows'][:, 2:7, 2:7].reshape(-1, 25)
    val_csp_probs = data['csp_probs']

X_train: np.ndarray = one_hot(windows_5x5[:50000], 11, True)
y_train: np.ndarray = np.ascontiguousarray(csp_probs[:50000], dtype=np.float64)

X_val: np.ndarray = one_hot(val_windows[:50000], 11, True)
y_val: np.ndarray = np.ascontiguousarray(val_csp_probs[:50000], dtype=np.float64)

net = NeuralNetwork([8,8], 275)

def calculate_batch_brier(features: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """features: (n_particles, total_feature_count) -> (n_particles,) Brier scores"""
    weights, biases = net.unpack(np.atleast_2d(features))
    preds = net.evaluate_batch(X, weights, biases)      # (n_particles, n_samples)
    return ((y[None, :] - preds) ** 2).mean(axis=1)

def evaluate_fitness(features: np.ndarray) -> np.ndarray:
    return calculate_batch_brier(features, X_train, y_train)

pso = PSO(50, net.total_feature_count, 1,  evaluate_fitness)
pso.initialize()
print("Starting fitness: " + str(pso.best_score))
for i in range(10):
    pso.iterate()
    print("Iteration fitness: " + str(pso.best_score) + " and val: " + str(calculate_batch_brier(pso.best_position, X_val, y_val)))
