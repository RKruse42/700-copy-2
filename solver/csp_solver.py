# solver/csp_solver.py
from collections import deque


def separate_components(frontier_cells, constraints):
    """
    Partitions frontier cells and constraints into independent connected components.
    Two frontier cells are in the same component if they share a constraint or are
    connected through a chain of shared constraints.
    """
    frontier_set = set(frontier_cells)
    if not frontier_set:
        return []

    # Build adjacency between frontier cells sharing any constraint
    adj = {cell: set() for cell in frontier_set}
    cell_to_constraints = {cell: [] for cell in frontier_set}

    for rev_cell, (req_mines, neighbors) in constraints.items():
        valid_neighbors = [n for n in neighbors if n in frontier_set]
        for i, c1 in enumerate(valid_neighbors):
            cell_to_constraints[c1].append(rev_cell)
            for c2 in valid_neighbors[i + 1:]:
                adj[c1].add(c2)
                adj[c2].add(c1)

    visited = set()
    components = []

    for cell in frontier_set:
        if cell not in visited:
            comp_cells = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                curr = queue.popleft()
                comp_cells.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Collect constraints that apply to this component
            comp_constraints = {}
            for c in comp_cells:
                for rev_cell in cell_to_constraints[c]:
                    if rev_cell not in comp_constraints:
                        comp_constraints[rev_cell] = constraints[rev_cell]

            components.append((comp_cells, comp_constraints))

    return components


def _solve_single_component(frontier_cells_list, constraints, max_frontier_size=85):
    """
    Solves a single isolated component using backtracking with early constraint pruning.
    """
    if not frontier_cells_list:
        return {}, 0

    if len(frontier_cells_list) > max_frontier_size:
        return {}, 0

    valid_configurations = []

    def is_partial_valid(current_assignment):
        """Early pruning check: verifies that current assignments don't violate bounds."""
        for revealed_cell, (required_mines, neighbors) in constraints.items():
            assigned_mines = 0
            unassigned_count = 0
            for n in neighbors:
                if n in current_assignment:
                    assigned_mines += current_assignment[n]
                else:
                    unassigned_count += 1

            # Too many mines placed
            if assigned_mines > required_mines:
                return False
            # Not enough remaining unassigned cells to fulfill required mines
            if assigned_mines + unassigned_count < required_mines:
                return False
        return True

    def backtrack(index, current_assignment):
        if index == len(frontier_cells_list):
            valid_configurations.append(current_assignment.copy())
            return

        current_cell = frontier_cells_list[index]

        # Branch A: Assume Safe (0)
        current_assignment[current_cell] = 0
        if is_partial_valid(current_assignment):
            backtrack(index + 1, current_assignment)

        # Branch B: Assume Mine (1)
        current_assignment[current_cell] = 1
        if is_partial_valid(current_assignment):
            backtrack(index + 1, current_assignment)

        del current_assignment[current_cell]

    backtrack(0, {})

    total_valid_states = len(valid_configurations)
    if total_valid_states == 0:
        return {}, 0

    probabilities = {}
    for cell in frontier_cells_list:
        mine_count = sum(state[cell] == 1 for state in valid_configurations)
        probabilities[cell] = mine_count / total_valid_states

    return probabilities, total_valid_states


def calculate_csp_probabilities(frontier_cells, constraints, max_component_size=85):
    """
    Calculates exact probability of each frontier cell containing a mine.
    Automatically decomposes the board into independent connected components and
    solves each component with early constraint pruning.

    Args:
        frontier_cells: An iterable of coordinates (e.g. [(0, 1), (1, 2)]).
        constraints: Dict mapping (row, col) -> (required_mines, [adjacent_frontier_cells]).
        max_component_size: Max cells in any single component before skipping for safety.

    Returns:
        probabilities: Dict mapping each frontier cell to its mine probability.
        total_states: Total valid board configurations across all components.
    """
    components = separate_components(frontier_cells, constraints)
    if not components:
        return {}, 0

    all_probabilities = {}
    total_states_prod = 1

    for comp_cells, comp_constraints in components:
        comp_probs, comp_states = _solve_single_component(
            comp_cells, comp_constraints, max_frontier_size=max_component_size
        )
        if not comp_probs or comp_states == 0:
            return {}, 0

        all_probabilities.update(comp_probs)
        total_states_prod *= comp_states

    return all_probabilities, total_states_prod
