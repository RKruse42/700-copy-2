import os
import sys
import numpy as np

# Ensure root repository directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simultaneously print to terminal and save exact terminal output to ga_runner_output.txt
class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

output_log_path = os.path.join(os.path.dirname(__file__), "ga_runner_output.txt")
sys.stdout = TeeLogger(output_log_path)

from naaz.genetic_algorithm import GeneticAlgorithm
from naaz.scoring_function import ScoringFunction
from simon.encoding import one_hot

print("Loading dataset splits...")
with np.load("cos700_dataset_train.npz") as data:
    train_windows_25 = data["windows"][:, 2:7, 2:7].reshape(-1, 25)
    train_windows = np.delete(train_windows_25, 12, axis=1)  # Drop center cell (index 12) -> 24 cells
    train_csp_probs = data["csp_probs"]

with np.load("cos700_dataset_val.npz") as data:
    val_windows_25 = data["windows"][:, 2:7, 2:7].reshape(-1, 25)
    val_windows = np.delete(val_windows_25, 12, axis=1)  # Drop center cell (index 12) -> 24 cells
    val_csp_probs = data["csp_probs"]

with np.load("cos700_dataset_test.npz") as data:
    test_windows_25 = data["windows"][:, 2:7, 2:7].reshape(-1, 25)
    test_windows = np.delete(test_windows_25, 12, axis=1)  # Drop center cell (index 12) -> 24 cells
    test_csp_probs = data["csp_probs"]

# Slicing 50,000 samples 
N_SAMPLES = min(50000, len(train_windows))
print(f"One-hot encoding {N_SAMPLES:,} samples (264 binary features each, excluding center cell)...")
X_train = one_hot(train_windows[:N_SAMPLES], 11, flatten=True)
y_train = np.ascontiguousarray(train_csp_probs[:N_SAMPLES], dtype=np.float64)

X_val = one_hot(val_windows[:N_SAMPLES], 11, flatten=True)
y_val = np.ascontiguousarray(val_csp_probs[:N_SAMPLES], dtype=np.float64)

X_test = one_hot(test_windows[:N_SAMPLES], 11, flatten=True)
y_test = np.ascontiguousarray(test_csp_probs[:N_SAMPLES], dtype=np.float64)

# Linear / Logistic Scoring Function (264 input features + 1 bias = 265 weights)
scoring_fn = ScoringFunction(feature_count=264)


def calculate_batch_brier(
    chromosomes: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Vectorized evaluation of multiple candidate chromosomes across all samples."""
    preds = scoring_fn.evaluate_batch(X, np.atleast_2d(chromosomes))
    return ((y[None, :] - preds) ** 2).mean(axis=1)


def evaluate_fitness(chromosomes: np.ndarray) -> np.ndarray:
    return calculate_batch_brier(chromosomes, X_train, y_train)


# Setup Genetic Algorithm with Early Stopping
POPULATION_SIZE = 50
MAX_GENERATIONS = 10000
PATIENCE = 100       # Stop if no improvement in 100 consecutive generations
MIN_DELTA = 1e-5     # Minimum threshold to consider as an improvement
SEED = 42

ga = GeneticAlgorithm(
    population_size=POPULATION_SIZE,
    feature_count=scoring_fn.total_weights,  # 265 parameters (264 weights + 1 bias)
    seed=SEED,
    fitness_fn=evaluate_fitness,
    crossover_rate=0.85,
    mutation_rate=0.02,
    mutation_sigma=0.1,
    tournament_size=3,
    elitism_count=2,
)

print(
    f"Initializing GA population (pop_size={POPULATION_SIZE},"
    f" parameters={scoring_fn.total_weights})..."
)
ga.initialize()

val_init = calculate_batch_brier(ga.best_chromosome, X_val, y_val)[0]
print(f"Gen 0000 | Train Brier: {ga.best_score:.5f} | Val Brier: {val_init:.5f}")

history = [(0, ga.best_score, val_init)]
best_val = val_init
best_gen = 0
best_chromosome = ga.best_chromosome.copy()
no_improve_count = 0

print(f"\nStarting Evolution (Max {MAX_GENERATIONS:,} Gens | Early Stopping Patience = {PATIENCE})...\n")

# Evolutionary Loop with Early Stopping
for gen in range(1, MAX_GENERATIONS + 1):
    ga.iterate()
    val_brier = calculate_batch_brier(ga.best_chromosome, X_val, y_val)[0]
    history.append((gen, ga.best_score, val_brier))

    # Track best validation score and patience
    if val_brier < best_val - MIN_DELTA:
        best_val = val_brier
        best_gen = gen
        best_chromosome = ga.best_chromosome.copy()
        no_improve_count = 0
    else:
        no_improve_count += 1

    # Print progress every 50 generations or early on
    if gen % 50 == 0 or gen <= 5:
        print(
            f"Generation {gen:4d} | Current Train: {ga.best_score:.5f} | Current Val: {val_brier:.5f} | "
            f"Best Score: {best_val:.5f} (discovered at Gen {best_gen:4d})"
        )

    if no_improve_count >= PATIENCE:
        print(
            f"\n>>> Early Stopping triggered at Generation {gen}!\n"
            f"    Validation score did not improve for {PATIENCE} consecutive generations.\n"
            f"    Champion Weights: Selected from Generation {best_gen} (Best Score = {best_val:.5f})."
        )
        break

# Final Evaluation of the Best Checkpoint on unseen Test Set
test_brier = calculate_batch_brier(best_chromosome, X_test, y_test)[0]
train_best = calculate_batch_brier(best_chromosome, X_train, y_train)[0]

print("\n" + "=" * 80)
print(f" FINAL CHAMPION PERFORMANCE (Weights Selected from Generation {best_gen})")
print("=" * 80)
print(f"  • Total Generations Executed : {len(history) - 1:,} generations (Stopped by Early Stopping)")
print(f"  • Best Score Discovered At   : Generation {best_gen}")
print(f"  • Champion Training Brier    : {train_best:.5f}")
print(f"  • Champion Validation Brier  : {best_val:.5f}  <-- Best score discovered at Gen {best_gen}")
print(f"  • Champion Test Brier        : {test_brier:.5f}  <-- Evaluated on unseen test split")
print("=" * 80)

# 1. Save detailed convergence history with best generation flagged
import csv

csv_path = os.path.join(os.path.dirname(__file__), "ga_convergence.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["generation", "train_brier", "val_brier", "status"])
    for gen, train_b, val_b in history:
        writer.writerow([gen, f"{train_b:.5f}", f"{val_b:.5f}", f"Best score discovered at Gen {best_gen}" if gen == best_gen else ""])
print(f"\nSaved convergence history (best score marked at Gen {best_gen}) to {csv_path}")

# 2. Save final run results summary CSV
results_path = os.path.join(os.path.dirname(__file__), "ga_final_results.csv")
with open(results_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value", "description"])
    writer.writerow(["Champion Generation", str(best_gen), f"The specific generation where the best score was discovered (Gen {best_gen})"])
    writer.writerow(["Best Score", f"{best_val:.5f}", f"Lowest validation Brier score (discovered at Gen {best_gen})"])
    writer.writerow(["Validation Brier", f"{best_val:.5f}", f"Lowest validation Brier score (discovered at Gen {best_gen})"])
    writer.writerow(["Test Brier", f"{test_brier:.5f}", "Generalization score of champion on unseen test split"])
    writer.writerow(["Training Brier", f"{train_best:.5f}", "Score of champion chromosome on training set"])
    writer.writerow(["Generations Executed", str(len(history) - 1), f"Total generations run before early stopping (Max: {MAX_GENERATIONS:,})"])
    writer.writerow(["Early Stopping", f"Triggered at Gen {gen}", f"Stopped automatically after {PATIENCE} generations without improvement"])
    writer.writerow(["Population Size", str(POPULATION_SIZE), "Number of candidate solutions"])
    writer.writerow(["Crossover Rate", "0.85", "Arithmetic blend probability"])
    writer.writerow(["Mutation Rate", "0.02", "Gaussian mutation probability"])
    writer.writerow(["Mutation Sigma", "0.10", "Gaussian mutation step size"])
    writer.writerow(["Tournament Size", "3", "Tournament selection pressure"])
    writer.writerow(["Elitism Count", "2", "Number of top individuals preserved"])
    writer.writerow(["Random Seed", str(SEED), "RNG seed (guarantees 100% reproducibility)"])
    writer.writerow(["Sample Count", str(N_SAMPLES), "Training samples used"])
print(f"Saved final results summary to {results_path}")


weights_path = os.path.join(os.path.dirname(__file__), "ga_best_weights.npz")
np.savez_compressed(
    weights_path,
    weights=best_chromosome[:-1],
    bias=best_chromosome[-1],
    full_chromosome=best_chromosome,
)
print(f"Saved champion scoring function weights to {weights_path}")