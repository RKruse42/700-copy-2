"""Hyperparameter Tuning & Sensitivity Analysis for Genetic Algorithm (GA).

This script systematically tests different values for each GA hyperparameter:
  1. Mutation Rate (p_m): low (0.005), balanced (0.02), high (0.08)
  2. Mutation Step Size (sigma): fine (0.05), balanced (0.10), coarse (0.30)
  3. Population Size (pop_size): small (20), balanced (50), large (100)
  4. Crossover Rate (p_c): moderate (0.60), balanced (0.85), high (0.95)
  5. Generation Budget: 50, 500, 1000, 2000, 3000, 4000

For each configuration, it evaluates:
  - Training Brier Score
  - Validation Brier Score (lower = better probability prediction)
  - Held-out Test Brier Score (unseen dataset generalization)
  - Runtime in seconds
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naaz.genetic_algorithm import GeneticAlgorithm
from naaz.scoring_function import ScoringFunction
from simon.encoding import one_hot

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
print("Loading dataset splits...")
with np.load("cos700_dataset_train.npz") as data:
    train_windows_25 = data["windows"][:, 2:7, 2:7].reshape(-1, 25)
    train_windows = np.delete(train_windows_25, 12, axis=1)  # Drop center cell (index 12) -> 24 cells
    train_csp = data["csp_probs"]

with np.load("cos700_dataset_val.npz") as data:
    val_windows_25 = data["windows"][:, 2:7, 2:7].reshape(-1, 25)
    val_windows = np.delete(val_windows_25, 12, axis=1)  # Drop center cell (index 12) -> 24 cells
    val_csp = data["csp_probs"]

with np.load("cos700_dataset_test.npz") as data:
    test_windows_25 = data["windows"][:, 2:7, 2:7].reshape(-1, 25)
    test_windows = np.delete(test_windows_25, 12, axis=1)  # Drop center cell (index 12) -> 24 cells
    test_csp = data["csp_probs"]

N_SAMPLES = min(50000, len(train_windows))
print(f"One-hot encoding {N_SAMPLES:,} samples (264 binary features each, excluding center cell)...")
X_train = one_hot(train_windows[:N_SAMPLES], 11, flatten=True)
y_train = np.ascontiguousarray(train_csp[:N_SAMPLES], dtype=np.float64)

X_val = one_hot(val_windows[:N_SAMPLES], 11, flatten=True)
y_val = np.ascontiguousarray(val_csp[:N_SAMPLES], dtype=np.float64)

X_test = one_hot(test_windows[:N_SAMPLES], 11, flatten=True)
y_test = np.ascontiguousarray(test_csp[:N_SAMPLES], dtype=np.float64)

scoring_fn = ScoringFunction(feature_count=264)
SEED = 42


def evaluate_batch_brier(chromosomes: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    preds = scoring_fn.evaluate_batch(X, np.atleast_2d(chromosomes))
    return ((y[None, :] - preds) ** 2).mean(axis=1)


def evaluate_train_fitness(chromosomes: np.ndarray) -> np.ndarray:
    return evaluate_batch_brier(chromosomes, X_train, y_train)


def run_configuration(
    label: str,
    pop_size: int = 50,
    generations: int = 50,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.02,
    mutation_sigma: float = 0.10,
    tournament_size: int = 3,
    elitism_count: int = 2,
) -> dict:
    """Trains a GA under specific hyperparameters and measures dataset metrics."""
    t0 = time.time()
    ga = GeneticAlgorithm(
        population_size=pop_size,
        feature_count=scoring_fn.total_weights,
        seed=SEED,
        fitness_fn=evaluate_train_fitness,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_sigma=mutation_sigma,
        tournament_size=tournament_size,
        elitism_count=elitism_count,
    )
    ga.initialize()

    for _ in range(generations):
        ga.iterate()

    runtime = time.time() - t0

    train_brier = float(ga.best_score)
    val_brier = float(evaluate_batch_brier(ga.best_chromosome, X_val, y_val)[0])
    test_brier = float(evaluate_batch_brier(ga.best_chromosome, X_test, y_test)[0])

    return {
        "label": label,
        "pop_size": pop_size,
        "generations": generations,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "mutation_sigma": mutation_sigma,
        "train_brier": train_brier,
        "val_brier": val_brier,
        "test_brier": test_brier,
        "runtime_sec": runtime,
    }


def run_generation_checkpoints(checkpoints: list[int]) -> list[dict]:
    """Runs a single GA evolution up to max(checkpoints) and records metrics at each milestone.

    Because GA RNG is deterministic with seed=42, checkpointing along a 4000-generation
    trajectory produces the exact identical state as running each generation budget separately,
    saving minutes of duplicate computation.
    """
    max_gen = max(checkpoints)
    print(f"\n[Generation Budget Trajectory] Running single continuous pass up to {max_gen} generations...")
    
    t0 = time.time()
    ga = GeneticAlgorithm(
        population_size=50,
        feature_count=scoring_fn.total_weights,
        seed=SEED,
        fitness_fn=evaluate_train_fitness,
        crossover_rate=0.85,
        mutation_rate=0.02,
        mutation_sigma=0.10,
        tournament_size=3,
        elitism_count=2,
    )
    ga.initialize()

    results = []
    ckpt_set = set(checkpoints)

    for gen in range(1, max_gen + 1):
        ga.iterate()
        if gen in ckpt_set:
            elapsed = time.time() - t0
            train_brier = float(ga.best_score)
            val_brier = float(evaluate_batch_brier(ga.best_chromosome, X_val, y_val)[0])
            test_brier = float(evaluate_batch_brier(ga.best_chromosome, X_test, y_test)[0])
            
            label = f"Generations = {gen:,}"
            print(f"  -> Checkpoint Gen {gen:4d}: Train={train_brier:.5f} | Val={val_brier:.5f} | Test={test_brier:.5f} ({elapsed:.1f}s)")
            results.append({
                "label": label,
                "pop_size": 50,
                "generations": gen,
                "crossover_rate": 0.85,
                "mutation_rate": 0.02,
                "mutation_sigma": 0.10,
                "train_brier": train_brier,
                "val_brier": val_brier,
                "test_brier": test_brier,
                "runtime_sec": elapsed,
            })

    return results


def main():
    # 1. Parameter Sensitivity Experiments (Fixed at 50 Generations)
    param_experiments = [
        # Baseline / Control
        {"label": "Baseline (50 Gens)", "pop_size": 50, "generations": 50, "crossover_rate": 0.85, "mutation_rate": 0.02, "mutation_sigma": 0.10},

        # 1. Mutation Rate Sensitivity (Too low vs Selected vs Too high)
        {"label": "Low Mut Rate (p_m=0.005)", "pop_size": 50, "generations": 50, "crossover_rate": 0.85, "mutation_rate": 0.005, "mutation_sigma": 0.10},
        {"label": "High Mut Rate (p_m=0.080)", "pop_size": 50, "generations": 50, "crossover_rate": 0.85, "mutation_rate": 0.080, "mutation_sigma": 0.10},

        # 2. Mutation Sigma Sensitivity (Too small vs Selected vs Too coarse)
        {"label": "Small Sigma (sigma=0.05)", "pop_size": 50, "generations": 50, "crossover_rate": 0.85, "mutation_rate": 0.02, "mutation_sigma": 0.05},
        {"label": "Large Sigma (sigma=0.30)", "pop_size": 50, "generations": 50, "crossover_rate": 0.85, "mutation_rate": 0.02, "mutation_sigma": 0.30},

        # 3. Population Size Sensitivity (Small vs Selected vs Large)
        {"label": "Small Pop (pop=20)", "pop_size": 20, "generations": 50, "crossover_rate": 0.85, "mutation_rate": 0.02, "mutation_sigma": 0.10},
        {"label": "Large Pop (pop=100)", "pop_size": 100, "generations": 50, "crossover_rate": 0.85, "mutation_rate": 0.02, "mutation_sigma": 0.10},

        # 4. Crossover Rate Sensitivity (Low vs Selected vs Very High)
        {"label": "Low Crossover (p_c=0.60)", "pop_size": 50, "generations": 50, "crossover_rate": 0.60, "mutation_rate": 0.02, "mutation_sigma": 0.10},
        {"label": "High Crossover (p_c=0.95)", "pop_size": 50, "generations": 50, "crossover_rate": 0.95, "mutation_rate": 0.02, "mutation_sigma": 0.10},
    ]

    print("\n" + "=" * 90)
    print("        GA HYPERPARAMETER TUNING & GENERATION SENSITIVITY EXPERIMENTS")
    print("=" * 90)

    results = []
    print("\n--- Part 1: Parameter Sensitivity (Fixed at 50 Generations) ---")
    for i, exp in enumerate(param_experiments, 1):
        print(f"[{i}/{len(param_experiments)}] Running: {exp['label']}...", end="", flush=True)
        res = run_configuration(**exp)
        results.append(res)
        print(f" Done! (Test Brier: {res['test_brier']:.5f} in {res['runtime_sec']:.1f}s)")

    # 2. Generation Budget Sensitivity (50, 500, 1000, 2000, 3000, 4000)
    print("\n--- Part 2: Generation Budget Sensitivity (Comparing 50 to 4,000 Generations) ---")
    gen_checkpoints = [50, 500, 1000, 2000, 3000, 4000]
    gen_results = run_generation_checkpoints(gen_checkpoints)
    results.extend(gen_results)

    # Print Summary Comparison Table
    sep = "=" * 115
    subsep = "-" * 115
    print("\n" + sep)
    print(" GA HYPERPARAMETER & GENERATION SENSITIVITY ANALYSIS RESULTS")
    print(sep)
    print(" Baseline: Pop=50 | p_c=0.85 | p_m=0.02 | sigma=0.10 | k=3 | elitism=2")
    print(subsep)
    print(f"{'Experiment':<28} | {'N':>3} | {'Gens':>5} | {'p_c':>4} | {'p_m':>5} | {'sigma':>5} | {'Train Brier':<11} | {'Val Brier':<10} | {'Test Brier':<10} | {'Time':<6}")
    print(subsep)

    for r in results:
        star = " *" if r["label"].startswith("Baseline") else ""
        print(
            f"{r['label']:<28} | {r['pop_size']:>3} | {r['generations']:>5d} | {r['crossover_rate']:>4.2f} | {r['mutation_rate']:>5.3f} | {r['mutation_sigma']:>5.2f} | "
            f"{r['train_brier']:<11.5f} | {r['val_brier']:<10.5f} | {r['test_brier']:<10.5f} | {r['runtime_sec']:>5.1f}s{star}"
        )
    print(sep)
    print("(* = Literature-grounded baseline configuration)")

    # Save to CSV inside naaz/
    csv_path = os.path.join(os.path.dirname(__file__), "ga_tuning_results.csv")
    with open(csv_path, "w") as f:
        f.write("experiment,pop_size,generations,crossover_rate,mutation_rate,mutation_sigma,train_brier,val_brier,test_brier,runtime_sec\n")
        for r in results:
            f.write(
                f"{r['label']},{r['pop_size']},{r['generations']},{r['crossover_rate']},{r['mutation_rate']},{r['mutation_sigma']},"
                f"{r['train_brier']:.6f},{r['val_brier']:.6f},{r['test_brier']:.6f},{r['runtime_sec']:.2f}\n"
            )
    print(f"\nSaved complete tuning table to: {csv_path}")


if __name__ == "__main__":
    main()
