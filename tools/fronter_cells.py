
# gets all the frontier cells (cells that are adjacent to a cell
# that is not hidden)
from corpus.extract_sliding_window import get_sliding_window


def get_frontier_cells(board_hidden):
    rows, cols = board_hidden.shape
    frontier = set()
    for r in range(rows):
        for c in range(cols):
            if board_hidden[r][c] == 9:      # hidden cells can't anchor
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board_hidden[nr][nc] == 9:
                        frontier.add((nr, nc))
    return frontier
