from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import numpy as np


def compute_feature_mi(args):
    idx, feature_col, target = args
    score = mutual_info_regression(feature_col.reshape(-1, 1), target, random_state=42)[0]
    return idx, score


def calculate_mutual_information(windows_matrix, probs_column, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    if windows_matrix.ndim == 3:
        num_samples, height, width = windows_matrix.shape
        flattened_windows = windows_matrix.reshape(num_samples, height * width)
    else:
        num_samples, num_features = windows_matrix.shape
        width = int(np.sqrt(num_features))
        height = width
        flattened_windows = windows_matrix

    num_features = height * width
    tasks = [
        (i, flattened_windows[:, i], probs_column)
        for i in range(num_features)
    ]

    mi_scores = np.zeros(num_features)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_feature_mi, task): task[0]
            for task in tasks
        }
        with tqdm(total=num_features, desc="Calculating MI Features", unit="cell") as pbar:
            for future in as_completed(futures):
                idx, score = future.result()
                r, c = divmod(idx, width)
                mi_scores[idx] = score
                pbar.set_postfix({"last_cell": f"({r},{c})", "score": f"{score:.4f}"})
                pbar.update(1)

    mi_matrix = mi_scores.reshape(height, width)
    return mi_matrix



def get_mutual_information(input_data='cos700_dataset.npz', output_filename=None, max_workers=None):
    if isinstance(input_data, str):
        with np.load(input_data) as data:
            probs_column = data['csp_probs']
            windows_matrix_column = data['windows']
    else:
        probs_column = input_data['csp_probs']
        windows_matrix_column = input_data['windows']

    mi_matrix = calculate_mutual_information(windows_matrix_column, probs_column, max_workers=max_workers)

    if output_filename:
        np.savez_compressed(output_filename, mi_matrix=mi_matrix)
        print(f"Saved MI matrix of shape {mi_matrix.shape} to {output_filename}")

    return mi_matrix


if __name__ == '__main__':
    NUM_CORES = os.cpu_count() or 4
    matrix = get_mutual_information('cos700_dataset.npz', 'mutual_information.npz', 12)
    print("MI Matrix:\n", matrix)
