"""
GP hyperparameters used by multi_run.py and compare_agents.py.

python3 -m richard.tune writes the winning configuration to best_config.json
next to this file, and it is loaded automatically. Until you have run the
tuning on the CURRENT inputs (the 264 one-hot indicators) and sample sizes,
IS_TUNED is False and the placeholder defaults below are used - they were
chosen for an earlier, different input set and must not be reported as tuned.
"""
import json
import os

_PLACEHOLDER = {
    "population_size": 120,
  "generations": 25,
  "tournament_size": 7,
  "crossover_rate": 0.6837793050475333,
  "mutation_rate": 0.31294526924321114,
  "max_depth": 8,
  "init_max_depth": 4,
}

_PATH = os.path.join(os.path.dirname(__file__), "best_config.json")

if os.path.exists(_PATH):
    with open(_PATH) as f:
        BEST_CONFIG = json.load(f)
    IS_TUNED = True
else:
    BEST_CONFIG = dict(_PLACEHOLDER)
    IS_TUNED = False