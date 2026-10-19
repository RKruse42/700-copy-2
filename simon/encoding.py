import numpy as np


def one_hot(input_arr: np.ndarray, num_classes: int = 11, flatten: bool = False) -> np.ndarray:
    # Ensure indices are integers for indexing
    indices = np.asarray(input_arr, dtype=np.intp)

    encoded = np.eye(num_classes, dtype=np.float64)[indices]

    if flatten:
        # Reshapes e.g. (N, 25, 11) -> (N, 275) or (25, 11) -> (275,)
        return encoded.reshape(*indices.shape[:-1], -1) if indices.ndim > 1 else encoded.ravel()

    return encoded
