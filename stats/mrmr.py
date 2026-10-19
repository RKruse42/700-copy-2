import numpy as np


def calculate_redundancy(windows_matrix, col_a, col_b):
    a = windows_matrix[:, col_a].astype(np.int64, copy=False)
    b = windows_matrix[:, col_b].astype(np.int64, copy=False)

    # 16x16 joint distribution (Minesweeper values 0-12)
    joint = np.bincount(a * 16 + b, minlength=256).reshape(16, 16)
    nz = joint > 0
    if not np.any(nz):
        return 0.0

    total = float(len(a))
    pxy = joint[nz] / total
    px = joint.sum(axis=1) / total
    py = joint.sum(axis=0) / total
    px_py = (px[:, None] * py[None, :])[nz]

    # I(X; Y) = sum( P(x, y) * log( P(x,y) / (P(x)*P(y)) ) )
    mi = np.sum(pxy * np.log(pxy / px_py))
    return max(0.0, float(mi))


def select_next_best(windows_matrix, mi_matrix, selected_cells, exclude_border=True):
    # 1. Flatten windows matrix to (N, num_features) if 3D
    if windows_matrix.ndim == 3:
        n_samples, height, width = windows_matrix.shape
        flat_windows = windows_matrix.reshape(n_samples, -1)
    else:
        flat_windows = windows_matrix
        width = int(np.sqrt(flat_windows.shape[1]))
        height = width

    num_features = flat_windows.shape[1]

    # 2. Flatten mi_matrix to 1D relevance vector (num_features,)
    flat_mi = mi_matrix.ravel()

    # 3. Standardize selected_cells to flat integer indices
    flat_selected = set()
    input_is_tuples = (mi_matrix.ndim == 2)
    for cell in selected_cells:
        if isinstance(cell, (tuple, list)):
            input_is_tuples = True
            r, c = cell
            flat_selected.add(r * width + c)
        else:
            flat_selected.add(int(cell))

    # 4. Identify candidate cells (excluding selected, center target cell, and optionally outer border)
    center_r, center_c = height // 2, width // 2
    candidate_cells = []
    for i in range(num_features):
        if i in flat_selected:
            continue
        r, c = divmod(i, width)
        # Exclude center cell (target being predicted)
        if r == center_r and c == center_c:
            continue
        if exclude_border:
            if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                continue
        candidate_cells.append(i)

    if not candidate_cells:
        return None, None

    # 5. Base case: If no cells have been selected yet, pick the one with max MI (relevance)
    if len(flat_selected) == 0:
        best_idx = max(candidate_cells, key=lambda i: flat_mi[i])
        best_score = float(flat_mi[best_idx])
        return (divmod(best_idx, width) if input_is_tuples else best_idx), best_score

    # 6. MRMR Score calculation for each candidate
    best_cell = None
    best_score = -float('inf')

    for candidate in candidate_cells:
        relevance = flat_mi[candidate]

        # Calculate average redundancy with all currently selected cells
        redundancies = [
            calculate_redundancy(flat_windows, candidate, s)
            for s in flat_selected
        ]
        avg_redundancy = np.mean(redundancies)

        # MRMR score = relevance - redundancy
        mrmr_score = relevance - avg_redundancy

        if mrmr_score > best_score:
            best_score = mrmr_score
            best_cell = candidate

    # Return in the format matching user's input (tuple or flat int)
    if input_is_tuples:
        return divmod(best_cell, width), best_score
    return best_cell, best_score


def mrmr_feature_selection(windows_matrix, mi_matrix, k=10):
    selected = []
    scores = []
    for _ in range(k):
        next_cell, score = select_next_best(windows_matrix, mi_matrix, selected)
        if next_cell is None:
            break
        selected.append(next_cell)
        scores.append(score)
    return selected, scores
