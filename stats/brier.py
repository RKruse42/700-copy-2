import math
import numpy as np

# it's just MSE
def brier(target: list[float], actual: list[float]):
    sse: float = 0
    for i in range(len(target)):
        sse += pow(target[i]-actual[i], 2)
    return sse / len(target)

def brier_np(target: np.ndarray, actual: np.ndarray) -> float:
    diff = target - actual
    return np.mean(diff ** 2)
