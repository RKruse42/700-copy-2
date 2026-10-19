import sys
import numpy as np


def compute_radial_average(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    rot0 = matrix
    rot90 = np.rot90(matrix, 1)
    rot180 = np.rot90(matrix, 2)
    rot270 = np.rot90(matrix, 3)

    radial_avg = (rot0 + rot90 + rot180 + rot270) / 4.0
    return radial_avg


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "mutual_information.npz"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "mutual_information_radial.npz"

    with np.load(input_file) as data:
        key = 'mi_matrix' if 'mi_matrix' in data else list(data.keys())[0]
        matrix = data[key]

    radial_matrix = compute_radial_average(matrix)
    np.savez_compressed(output_file, mi_matrix=radial_matrix)

    print(f"Original Matrix:\n{matrix}\n")
    print(f"Radial Average Matrix:\n{radial_matrix}\n")
    print(f"Saved radial average to {output_file}")


if __name__ == "__main__":
    main()
