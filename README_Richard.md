# COS700 – Genetic Programming

This document describes everything built for the **Genetic Programming (GP)** part of the project, with the shared infrastructure GP depends on (the dataset system, the game environment, the baseline agents and the evaluation/statistics code), how to run all of it, and the reasoning behind each design decision.

---

## 1. Overview and headline results

**Two ways to train it.** The same evolutionary machinery can be driven by two different fitness signals, selected with one flag:

| Mode       | Fitness signal                                                                                                                   | Needs the game engine? | Speed              |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------ |
| `dataset`  | Brier score (MSE) between the tree's predicted mine probability and the exact CSP-solver probability, on the precomputed dataset | No                     | Seconds            |
| `gameplay` | Average shaped reward over a fixed set of seeded, fully simulated games                                                          | Yes                    | Absolutely forever |

Training two ways is deliberate: it lets the study ask whether optimising a cheap per-cell proxy (predict the mine probability) transfers to actually playing well, which is a useful result whichever way it comes out.

**Headline result (100 held-out seeds, 8x8 board, 10 mines).**

| Agent                | Score (cells cleared) | Moves survived | Win rate |
| -------------------- | --------------------- | -------------- | -------- |
| Random baseline      | 24.00 ± 11.30         | 2.28 ± 1.61    | 0.0%     |
| Handcrafted baseline | 35.85 ± 13.73         | 6.93 ± 5.24    | 2.0%     |
| GP, dataset-trained  | 22.44 ± 10.42         | 1.42 ± 1.45    | 0.0%     |
| GP, gameplay-trained | 31.90 ± 12.58         | 4.89 ± 3.61    | 0.0%     |

You can see here the winrate is an issue. This would probably need to be fixed. All these values will change later as well

---

## 2. How to run everything

All commands are run **from the repository root**

### 2.1 Setup

```bash
pip install numpy scipy tqdm scikit-learn
```

### 2.2 Commands

| Purpose                                                                          | Command                                 |
| -------------------------------------------------------------------------------- | --------------------------------------- | --- |
| Train GP once (dataset or gameplay, see flag below)                              | `python3 -m richard.runner`             |     |
| Repeated independent training runs, both modes                                   | `python3 -m richard.multi_run`          |
| Baselines only (random + handcrafted)                                            | `python3 -m baselines.runner`           |
| Full pipeline: train GP both ways, evaluate all four agents, pairwise statistics | `python3 -m environment.compare_agents` |

Usually I just run the full pipeline (environment.compare_agents)

### 2.3 Choosing the training mode

At the bottom of `richard/runner.py`:

```python
TRAINING_MODE = "dataset"   # or "gameplay"
```

Flipping this single flag is the only change needed. The evolution loop is identical for both modes; only the fitness function handed to it differs. Note that the `__main__` block of `runner.py` uses small demonstration settings (population 50, 20 generations); the tuned settings live in `richard/best_config.py` and are used by `multi_run.py` / `compare_agents.py`.

I used AI to write that - lmk if that does not make sense

---

## 3. The environment system (`environment/`)

### 3.1 Why it exists

Before this module, every learner was trained on the static dataset: "given this recorded window, predict this cell's mine probability".

The reveal loop inside `corpus/generate_dataset.py` looks similar but is not a substitute: it always reveals the solver's safest cell and silently skips mines to keep harvesting labels. The environment is the opposite: an external policy chooses the cell, and hitting a mine **ends the game**.

It is deliberately **shared infrastructure**, not GP-specific. Every algorithm (GP, GA, Neurocontroller) and both baselines are meant to play through the same seeded games, which is what makes the Mann-Whitney comparison in Section 4.4 of the proposal valid.

That is what AI says. I honestly only made it because my system sucks with the Brier things. Will refine later

### 3.2 Components

**`game.py` - `MinesweeperGame(width, height, mines, seed)`**
A stateful game built directly on the existing `corpus/` functions (nothing reimplemented).

- `step(row, col)` returns a `MoveResult`: `SAFE`, `MINE` (game over, loss), `WIN` (game over, win) or `ALREADY_REVEALED` (invalid no-op). It checks the _true_ board first, because `reveal_section` does not itself check for mines.
- `legal_moves()` returns the frontier when one exists, otherwise every hidden cell. An unguided guess is unavoidable when nothing borders a revealed number.
- `cells_revealed()`, `safe_cell_count`, `moves_made`, `won`, `game_over` expose the outcome.

**`play.py` - the shared play loop**

- `extract_features` pulls the same 5x5 window used at training time from the _live_ board, so play-time and training-time features are identical.
- `choose_move(game, policy_fn)` scores every legal cell in one batch and returns the `argmax`.
- `play_game(policy_fn, width, height, mines, seed)` plays to completion and returns
  `{won, cells_revealed, safe_cell_count, moves_made}`.

`policy_fn` is the only thing that differs between algorithms: it maps a `(n_cells, 25)` array of windows to `(n_cells,)` scores. A GP tree, a neural network, a random scorer and a hand-written rule all plug in identically.

**`evaluate.py` - one evaluation procedure for everyone**
`evaluate_policy`, `run_comparison`, `summarise`, `mann_whitney_compare`, `pairwise_mann_whitney`,
`print_pairwise_table` and `save_results_csv` (details in [Section 8](#8-evaluation-methodology)).

**`compare_agents.py`** ties training, evaluation and statistics together (Section 3.2).

---

## 4. The Genetic Program (`richard/`)

### 4.1 Overview

The population evolves through tournament selection, subtree crossover and subtree mutation, with elitism and a depth cap.

### 4.2 Function set (`functions.py`)

| Function             | Arity | Behaviour                                                         |
| -------------------- | ----- | ----------------------------------------------------------------- |
| `+`, `-`, `*`        | 2     | Arithmetic                                                        |
| `/`                  | 2     | **Protected** division: returns 1.0 when the divisor is near zero |
| `min`, `max`         | 2     | Element-wise                                                      |
| `if_lte(a, b, c, d)` | 4     | `c` if `a <= b`, else `d`                                         |

### 4.3 Terminal set (`terminals.py`, `corpus/derived_features.py`)

Every tree sees **33 features**:

- **25 raw features** `x0`-`x24`: the flattened 5x5 window (values 0-8 = neighbouring mine count, 9 = hidden,
  10 = out of bounds). Index 12 is the candidate cell itself.
- **8 derived features**, computed about the candidate cell:

#### READ THIS: The derived features are added to corpus because GP or GA cannot calculate this realistically. They need them to have any chance of getting anything rihgt

| Name                                                            | Meaning                                                                                                             |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `hidden_neighbour_count`                                        | How many of the 8 neighbours are hidden                                                                             |
| `revealed_neighbour_count`                                      | How many neighbours are revealed numbers                                                                            |
| `out_of_bounds_neighbour_count`                                 | How many neighbours are off the board (edge indicator)                                                              |
| `forced_mine`                                                   | 1 if some touching revealed number's hidden-neighbour count equals its value (the cell is certainly a mine), else 0 |
| `risk_estimate`                                                 | Worst-case (highest) `value / hidden_neighbours` over touching revealed numbers; 0.5 when there is no evidence      |
| `avg_neighbour_value`                                           | Mean value of in-bounds neighbours                                                                                  |
| `min_revealed_neighbour_value` / `max_revealed_neighbour_value` | Range of revealed neighbouring numbers (-1 when none)                                                               |

- **Ephemeral random constants** drawn uniformly from [-1, 1], fixed when a tree is created. A leaf is a feature or a constant with equal probability.

Here is more AI explaining some reasoning for some of the changes made. Without these features the GP sucks so the terminal set had to be expanded. Read below if you want to know why

**Why the derived features exist.** With only raw cell values, a tree that wants to know "how many of my neighbours
are hidden?" or "is this cell certainly a mine?" must _reconstruct_ that from nested arithmetic on individual
numbers. That is exactly what the handcrafted agent computes directly, and it is a very hard structure to find by
random crossover and mutation. Supplying these quantities as ready-made leaves lets evolution spend its effort on
_combining and weighting_ evidence. Effect observed: dataset-mode Brier fell from a plateau near 0.168 to about
0.003, and gameplay score rose from 27.27 to 31.90 ([Section 9.2](#92-effect-of-the-enriched-terminal-set)).

**Where it lives.** The expansion from raw to enriched features happens only inside GP code (`policy.py` for play,
`fitness.py` for the dataset). The shared harness and the baselines continue to operate on the plain 25-value window.
See the fairness caveat in [Section 11](#11-known-limitations-and-caveats).

### 3.4 Tree representation and initialisation (`node.py`)

Honestly you can ignore this. Its actually just useful for me later when explaining

- `Node` is either a function node (name + children) or wraps a terminal. It is **immutable**: nothing changes
  after construction. Operators therefore build new nodes instead of editing in place, so subtrees can be shared
  between parents and children without one individual's mutation corrupting another.
- `evaluate(X)` is **vectorised**: a tree scores a whole `(n_samples, 33)` array in one recursive pass, rather than
  one row at a time. `depth()` and `size()` support depth capping and bloat tracking; `repr` prints the expression
  with the readable feature names.
- **Initialisation: ramped half-and-half.** `generate_tree` supports two methods: `full` (branches always reach the
  target depth) and `grow` (a leaf may be chosen earlier, with probability 0.3 below the root). The initial
  population cycles through a range of depths and alternates the two methods.

**Why ramped half-and-half.** Pure `full` gives near-identical bushy trees; pure `grow` gives mostly tiny trees.
Mixing depths and methods produces a diverse starting population in both shape and size, which matters because
crossover and mutation can only recombine what already exists.

### 3.5 Selection and genetic operators (`operators.py`)

- **Tournament selection**
- **Subtree crossover**
- **Subtree mutation**
- **Depth cap (bloat control).**
- **Elitism** (in `runner.py`)

### 3.6 Fitness: two interchangeable modes (`fitness.py`)

Both modes are reduced to one shape, `fitness_fn(tree) -> float`, **lower is better**, built by a factory that captures its data once.

**Dataset mode**

```
fitness = Brier( csp_probs , sigmoid( clip(tree(features), -60, 60) ) )
```

- The Brier score is the mean squared error between the tree's predicted mine probability and the exact solver probability. It is the same metric that simon's PSO uses, so training numbers are comparable.

**Gameplay mode**

```
reward(game) = (cells_revealed / safe_cells)  -  0.3 if the game was not won
fitness      = - mean( reward ) over a fixed list of seeds
```

- A win scores exactly 1.0. A death costs a fixed
  penalty on top of progress, so clearing half the board before dying still beats dying immediately.
- The value **0.3 is a design choice, not a tuned quantity**; a sensitivity check is listed under remaining work.

### 3.7 The policy adapter (`policy.py`)

AI explaination:

`make_gp_policy(tree)` returns a `policy_fn` that expands the raw `(n, 25)` windows the environment supplies into the
33-feature space, then evaluates the tree. It is the **single place** where a tree is converted into something the
shared harness can play, so gameplay-mode training and the final evaluation use identical features. Any code that
turns a trained tree into a policy must use it instead of calling `tree.evaluate` directly.

---

## 4. The baseline agents (`baselines/`)

These are the random and handcrafted agent. I don't know how smart the handcrafted one should be

---

## 5. Evaluation methodology

### Metrics

- **Score** = `cells_revealed`: the raw count of safe cells cleared when the game ended. No penalty or normalisation.
  Board size and mine count are fixed within a comparison, so the safe-cell total is constant across seeds and raw
  counts are directly comparable.
- **Moves** = `moves_made`: number of clicks made ("length of play").

They are **distinct and can diverge**. One click can reveal many cells through a zero-cascade, and the opening
reveal is free, so score can be high with very few moves. A high score with a very low move count (as for the
dataset-trained tree) means most of the progress came from the automatic opening cascade rather than sustained good
decisions. Reporting both exposes this.

---

## 6. Deviations from the proposal

Now I realise there is a difference from the proposal to the code, so I gave AI the proposal and the codebase and told it to find differences:

| Proposal                                                    | What was done                               | Reason                                                                              |
| ----------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| All agents "perceive an identical feature space" (Sec. 4.1) | GP additionally receives 8 derived features | Removes an artificial handicap; **must be reconciled with GA/PSO** (see Section 11) |
| Penalty for selecting an already-cleared cell (Sec. 4.2)    | Not applied                                 | The environment only offers hidden cells as legal moves, so it can never occur      |
| Handcrafted rule wording (Sec. 4.3)                         | Logically corrected version                 | The literal wording is inverted (equal counts mean mines, not safe cells)           |
| Two GP training modes; Brier-vs-gameplay relationship       | Added                                       | Answers whether a cheap proxy transfers to real play; extra sub-question            |
| Board size "small, larger later" (Sec. 4.1)                 | 8x8, 10 mines throughout                    | Keeps games fast; same mine density as the dataset boards                           |

---

## IGNORE THIS - THIS IS JUST THE STUFF THAT ARE PROBLEMS MAYBE ILL SORT THROUGH IT

## Known limitations and caveats

These matter for the write-up and should be stated honestly.

1. **Feature-set fairness.** Section 4.1 of the proposal requires an identical feature space across algorithms. The
   derived features give GP access to the same reasoning primitives the handcrafted agent uses. Either give GA and the
   Neurocontroller the same derived features, or present **GP-raw** (25 features) and **GP-enriched** (33 features)
   as separate variants and keep the raw variant in the strict three-way comparison. Raw-feature GP results are in
   Section 9.2.
2. **Dataset splits leak.** The split script slices the _augmented_ dataset sequentially, and augmentation stacks the
   eight transforms as consecutive blocks. Consequently **all 1000 seeds appear in train, validation and test**
   (verified: every pair of splits shares all 1000 seeds), and the same board states appear in different splits in
   rotated or reflected form. Dataset-mode "validation" fitness is therefore not a true held-out estimate. The fix is
   to split by seed _before_ augmenting. This is shared infrastructure and affects every algorithm using these files.
   For GP the practical effect is probably small (trees have limited capacity to memorise), but it should be
   disclosed or fixed.
3. **Board-size mismatch.** The dataset was generated on 16x16 boards (40 mines); gameplay and evaluation use 8x8
   (10 mines). Mine density is the same (about 15.6%), but 9x9 windows on an 8x8 board contain far more
   out-of-bounds padding, so this is a plausible contributor to poor dataset-to-play transfer. It is a hypothesis,
   not a demonstrated cause.
4. **Frontier-only training data.** The dataset contains only cells that border a revealed number, so a
   dataset-trained tree has never seen a "blind guess" situation. Blind guesses matter a great deal on this dense
   board, where most games end within a few moves. This is the leading _hypothesis_ for the transfer gap. A
   diagnostic that logs, per move, whether the choice was a frontier decision or a blind guess would confirm or
   refute it; it has not been built.
5. **Subsampled dataset.** Dataset mode uses the first 10,000 rows of each file for speed, not a random sample; rows
   are ordered by seed chunk and game progress, so the subset may be unrepresentative.
6. **Tuned on the dataset only.** Hyperparameters were optimised for dataset-mode Brier and reused for gameplay mode.
7. **Noisy validation in gameplay mode.** Ten validation seeds is small; validation figures there are noisy.
8. **Score includes free cells.** The opening cascade is free, which is why moves is reported alongside score.
9. **Handcrafted agent limits.** It sees only a 5x5 window and single-number constraints (by design), and its
   fallback for cells with no local evidence is crude.
10. **Interpretability versus accuracy.** The tuned depth cap produces very large trees; see 6.9.
11. **Dense board.** 8x8 with 10 mines is as dense as the intermediate real-game setting; even the handcrafted agent
    wins only 2% of games, which compresses differences between agents.
12. **Damage penalty untuned.** `DEATH_PENALTY = 0.3` is a fixed design constant.

---

## Remaining work

- Reconcile the feature sets (Section 11, item 1) before the final three-way comparison.
- Fix the train/validation/test leakage (split by seed before augmenting) and re-run dataset-mode training.
- Build the frontier-versus-blind-guess diagnostic to test the transfer hypothesis.
- Brier-versus-gameplay analysis: compute each trained tree's Brier score and its gameplay score/moves and test the
  relationship with a Spearman correlation.
- Interpretability pass on the best trees from each mode (which features they rely on, comparison with the
  handcrafted logic), including a depth-capped tuning run (about 10-12) to obtain readable trees.
- Sensitivity check on `DEATH_PENALTY`, and optionally a separate tuning pass for gameplay mode.
- Scale evaluation (more seeds, larger or lower-density boards) once the above is settled.
- Integrate the GA and the Neurocontroller into `environment/evaluate.py` (each needs only a `policy_fn`; the
  Neurocontroller has so far only been trained and scored on the static dataset).

---

## 1. Intermediate Board — 16 × 16, 40 Mines

| Agent                 | Win Rate | Board Cleared (%) | Cells Cleared (Mean ± SD) | Moves Survived (Mean ± SD) |
| :-------------------- | -------: | ----------------: | ------------------------: | -------------------------: |
| Random Baseline       |     0.0% |      24.5 ± 14.9% |             52.96 ± 32.08 |                2.44 ± 1.98 |
| Handcrafted Baseline  |     0.0% |      42.1 ± 21.9% |             90.92 ± 47.30 |                9.75 ± 8.27 |
| GP — Dataset-Trained  |     0.0% |      22.7 ± 14.0% |             49.05 ± 30.14 |                1.12 ± 0.58 |
| GP — Gameplay-Trained |     0.0% |      37.7 ± 20.4% |             81.50 ± 44.16 |                7.96 ± 5.94 |
| **GA Agent**          | **0.0%** |  **45.1 ± 22.8%** |         **97.51 ± 49.33** |           **12.03 ± 9.91** |

## 2. Beginner Board — 8 × 8, 10 Mines

| Agent                 | Win Rate | Board Cleared (%) | Cells Cleared (Mean ± SD) | Moves Survived (Mean ± SD) |
| :-------------------- | -------: | ----------------: | ------------------------: | -------------------------: |
| Random Baseline       |     0.0% |      47.1 ± 20.7% |             25.41 ± 11.16 |                2.58 ± 2.12 |
| Handcrafted Baseline  | **7.4%** |  **70.0 ± 23.9%** |         **37.79 ± 12.88** |                7.71 ± 5.51 |
| GP — Dataset-Trained  |     0.0% |      42.3 ± 18.9% |             22.87 ± 10.22 |                1.29 ± 0.95 |
| GP — Gameplay-Trained |     4.2% |      62.4 ± 23.9% |             33.68 ± 12.91 |                5.87 ± 4.43 |
| **GA Agent**          | **2.8%** |  **70.0 ± 24.4%** |         **37.79 ± 13.16** |            **7.97 ± 5.53** |
