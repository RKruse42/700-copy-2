# Genetic Algorithm (GA) — Naaz

This folder contains the **Real-Coded Genetic Algorithm (RCGA)** implementation for our COS 700 Minesweeper comparative study. 

This component evolves a **Scoring Function** (265 continuous parameters) to predict whether a candidate cell is a mine, trained against the ground-truth probabilities from Simon's CSP solver.

---

## The Files in This Folder

- **`scoring_function.py`**  
  Defines the mathematical scoring model. It takes 24 surrounding cells from a $5 \times 5$ window (excluding the candidate center cell, one-hot encoded into 264 binary inputs), multiplies them by 264 weights, adds a bias term ($264 + 1 = 265$ parameters total), and squashes the result through a Sigmoid function to output a predicted mine probability ($0.0 \to 1.0$).

- **`genetic_algorithm.py`**  
  The core evolutionary engine. Implements a continuous Real-Coded GA using:
  - **Tournament Selection** ($k = 3$) to select parents.
  - **Arithmetic Crossover** ($p_c = 0.85$) to blend parent weights.
  - **Gaussian Mutation** ($p_m = 0.02, \sigma = 0.10$) to introduce small random weight adjustments.
  - **Elitism** ($n = 2$) to preserve the top 2 champions unchanged into each generation.

- **`runner.py`**  
  The main training script. It loads 50,000 samples from Simon's dataset splits (`cos700_dataset_train.npz`, `val`, and `test`), drops the center cell (index 12), evolves weights with early stopping (patience 100, max 10,000 generations), evaluates on the unseen test set, and saves the final weights and convergence log.

- **`tune_hyperparameters.py`**  
  The hyperparameter sensitivity analysis script. It systematically tests 9 configurations across 50 generations (varying population size, mutation rate, step size, and crossover rate) plus a 4,000-generation trajectory, outputting a comparison table for the honours thesis.

- **`policy.py`**  
  Adapter connecting the GA weights into the gameplay environment (`environment/play.py` and `environment/evaluate.py`). Automatically handles 25-cell inputs from the live board by dropping the center cell, one-hot encodes the 24 cells into 264 inputs, and returns $-P(\text{mine})$ so `argmax` selects the lowest-risk square.

- **`evaluate_gameplay.py`**  
  Evaluates the GA champion weights through Richard's exact evaluation harness against the Random and Handcrafted baselines across 500 held-out boards (seeds 5000–5499). Computes win rate, mean score (% and raw), moves survived, and pairwise Mann-Whitney U-tests.

---

## Output Artifacts

- **`ga_best_weights.npz`** — Compressed binary file storing the champion 265 parameters (264 weights + 1 bias) evolved by the GA.
- **`ga_runner_output.txt`** — Verbatim terminal output log captured during training.
- **`ga_convergence.csv`** — Training and validation Brier scores per generation, with the peak generation marked.
- **`ga_final_results.csv`** — Report-ready summary of final Train, Validation, and Test Brier scores along with all hyperparameters.
- **`ga_tuning_results.csv`** — Results table from the hyperparameter sensitivity analysis.
- **`ga_gameplay_results.csv`** — Per-seed evaluation outcomes from the 500 gameplay tests.

---

## How to Run

From the root project directory (`COS700`):

1. **Train the GA model with early stopping:**
   ```bash
   python3 -m naaz.runner
   ```

2. **Run the hyperparameter sensitivity analysis:**
   ```bash
   python3 -m naaz.tune_hyperparameters
   ```

3. **Evaluate GA in the gameplay environment (500 seeds):**
   ```bash
   python3 -m naaz.evaluate_gameplay
   ```

---

## Benchmark Results (Early Stopping: Max 10,000, Patience 100)

- **Early Stopping:** Triggered at Generation 1,802 (no validation improvement for 100 generations).
- **Peak Generation:** Generation 1,702
- **Train Brier Score:** `0.06401`
- **Validation Brier Score:** `0.06375`
- **Test Brier Score:** `0.06542` *(unseen test split, demonstrating zero overfitting)*
- **Hyperparameters:** Population $N = 50$, Crossover $p_c = 0.85$, Mutation $p_m = 0.02$, Mutation Step $\sigma = 0.10$, Tournament $k = 3$, Elites $n = 2$.

---

## Gameplay Benchmark Results (500 Seeds: 5000–5499)

Evaluated across 500 held-out boards using Richard's evaluation harness (`environment/evaluate.py`) on Intermediate ($16 \times 16$, 40 Mines, 216 Safe Cells):

| Agent | Win Rate | Board Cleared (%) | Cells Cleared (Raw) | Moves Survived |
| :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | 0.0% | $24.5\% \pm 14.9\%$ | $52.96 \pm 32.08$ | $2.44 \pm 1.98$ |
| **Handcrafted Baseline** | 0.0% | $42.1\% \pm 21.9\%$ | $90.92 \pm 47.30$ | $9.75 \pm 8.27$ |
| **GA Agent (`load_ga_policy()`)** | **0.0%** | **$44.8\% \pm 22.3\%$** | **$96.73 \pm 48.16$** | **$11.72 \pm 9.41$** |

* **Statistical Significance vs Random:** $U = 26437.0, p < 0.0001$ *(The GA survived **4.8× longer** than random guessing, highly statistically significant under Bonferroni correction)*.
* **Statistical Significance vs Handcrafted:** $U = 141061.5, p = 0.0004$ *(The GA agent statistically significantly outperformed the handcrafted rule agent in moves survived, surviving $11.72$ vs $9.75$ moves)*.

