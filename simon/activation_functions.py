import numpy as np


def sigmoid(net: np.ndarray, ld: float = 1.0) -> np.ndarray:
    if ld != 1.0:
        net *= -ld
    else:
        np.negative(net, out=net)
    np.exp(net, out=net)
    net += 1.0
    np.reciprocal(net, out=net)
    return net
