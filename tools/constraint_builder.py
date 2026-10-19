from tools.fronter_cells import get_frontier_cells

def build_constraints(board_hidden):
    constraints = {}
    rows, cols = board_hidden.shape

    frontier_cells = get_frontier_cells(board_hidden)

    for r in range(rows):
        for c in range(cols):
            val = board_hidden[r][c]

            if 1 <= val <= 8:
                adjacent_frontier = []

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue

                        nr, nc = r + dr, c + dc

                        if 0 <= nr < rows and 0 <= nc < cols:
                            if board_hidden[nr][nc] == 9:
                                adjacent_frontier.append((nr, nc))

                if len(adjacent_frontier) > 0:
                    constraints[(r, c)] = (val, adjacent_frontier)

    return frontier_cells, constraints
