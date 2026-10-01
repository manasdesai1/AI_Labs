from collections import deque
import heapq

# ------------------------------------------------------------
# Goal-Based Agent for Warehouse Navigation
# ------------------------------------------------------------
# Symbols used in the grid:
#   S -> Start position
#   G -> Goal position
#   # -> Obstacle / blocked cell
#   . -> Free cell
#   * -> Path found by the agent
#
# The agent is goal-based: it plans a complete collision-free
# route from S to G before moving, then executes that plan one
# action at a time.
# ------------------------------------------------------------


WAREHOUSE_MAP = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""


# Goal sealed off by shelving, used for the negative test.
SEALED_MAP = """\
#####################
#S....#...........#G#
#.##....##########.##
#....##............##
#.######.###.#.###.##
#........#.........##
#####################"""


# Possible movements: Up, Down, Left, Right.
# One move changes the position by one grid square.
DIRECTIONS = [
    ("Up",    -1,  0),
    ("Down",   1,  0),
    ("Left",   0, -1),
    ("Right",  0,  1),
]


def parse_grid(text):
    """
    Convert an ASCII map into a 2D grid.

    Parameters:
        text : multi-line string describing the warehouse

    Returns:
        2D list of characters.
    """
    return [list(row) for row in text.splitlines()]


def find_position(grid, symbol):
    """
    Find the position of a given symbol in the grid.

    Parameters:
        grid   : 2D list representing the warehouse
        symbol : symbol to search for ('S' or 'G')

    Returns:
        (row, column) of the symbol.

    Raises:
        ValueError if the symbol is not present.
    """
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == symbol:
                return row, col

    raise ValueError(f"{symbol} not found in the warehouse.")


def is_free(grid, cell):
    """
    Check whether a cell can be occupied by the robot.

    Parameters:
        grid : 2D warehouse grid
        cell : (row, column) to test

    Returns:
        True if the cell is inside the grid and is not an obstacle.
    """
    row, col = cell

    if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
        return False

    return grid[row][col] != '#'


def free_cells(grid):
    """
    Count the cells the robot is allowed to occupy.

    Parameters:
        grid : 2D warehouse grid

    Returns:
        Number of free cells.
    """
    return sum(1 for row in grid for cell in row if cell != '#')


def manhattan(a, b):
    """
    Grid distance between two cells ignoring obstacles.

    Parameters:
        a, b : (row, column) positions

    Returns:
        |row difference| + |column difference|.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(parent, goal):
    """
    Reconstruct the path from S to G using the parent dictionary.

    Parameters:
        parent : dictionary containing each cell's predecessor
        goal   : goal position

    Returns:
        List of positions from start to goal.
    """
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    # The path was constructed backwards, so reverse it.
    path.reverse()

    return path


def bfs(grid, start, goal):
    """
    Breadth-First Search for a shortest path from start to goal.

    BFS explores cells in order of increasing number of moves, so
    on a grid where every move costs the same it returns a path
    with the fewest possible moves.

    Parameters:
        grid  : 2D warehouse grid
        start : (row, column) of starting position
        goal  : (row, column) of goal position

    Returns:
        (path, cells_expanded). path is None if no route exists.
    """
    queue = deque([start])
    parent = {start: None}
    expanded = 0

    while queue:

        current = queue.popleft()
        expanded += 1

        if current == goal:
            return reconstruct_path(parent, goal), expanded

        for name, dr, dc in DIRECTIONS:

            neighbor = (current[0] + dr, current[1] + dc)

            if not is_free(grid, neighbor):
                continue

            if neighbor in parent:
                continue

            parent[neighbor] = current
            queue.append(neighbor)

    # Queue became empty without reaching the goal.
    return None, expanded


def dfs(grid, start, goal):
    """
    Depth-First Search, included only for comparison.

    DFS follows one branch as far as it can before backtracking,
    so it finds a route but not necessarily a short one.

    Parameters:
        grid  : 2D warehouse grid
        start : (row, column) of starting position
        goal  : (row, column) of goal position

    Returns:
        (path, cells_expanded). path is None if no route exists.
    """
    stack = [start]
    parent = {start: None}
    expanded = 0

    while stack:

        current = stack.pop()
        expanded += 1

        if current == goal:
            return reconstruct_path(parent, goal), expanded

        for name, dr, dc in DIRECTIONS:

            neighbor = (current[0] + dr, current[1] + dc)

            if not is_free(grid, neighbor):
                continue

            if neighbor in parent:
                continue

            parent[neighbor] = current
            stack.append(neighbor)

    return None, expanded


def astar(grid, start, goal):
    """
    A* search using Manhattan distance as the heuristic.

    Cells are expanded in order of f = g + h, so cells that appear
    to lead towards the goal are preferred. Because Manhattan
    distance never overestimates the real distance on this grid,
    the path returned is still a shortest path.

    Parameters:
        grid  : 2D warehouse grid
        start : (row, column) of starting position
        goal  : (row, column) of goal position

    Returns:
        (path, cells_expanded). path is None if no route exists.
    """
    # The counter keeps the ordering deterministic when two cells
    # have the same f and g values.
    counter = 0

    frontier = [(manhattan(start, goal), 0, counter, start)]
    parent = {start: None}
    cost = {start: 0}
    expanded = 0

    while frontier:

        f, g, _, current = heapq.heappop(frontier)
        expanded += 1

        if current == goal:
            return reconstruct_path(parent, goal), expanded

        for name, dr, dc in DIRECTIONS:

            neighbor = (current[0] + dr, current[1] + dc)

            if not is_free(grid, neighbor):
                continue

            new_cost = g + 1

            # Only queue a cell if this route to it is cheaper.
            if neighbor in cost and new_cost >= cost[neighbor]:
                continue

            cost[neighbor] = new_cost
            parent[neighbor] = current
            counter += 1

            heapq.heappush(
                frontier,
                (new_cost + manhattan(neighbor, goal), new_cost,
                 counter, neighbor)
            )

    return None, expanded


def path_to_actions(path):
    """
    Convert a list of cells into the action names producing it.

    Parameters:
        path : list of (row, column) positions

    Returns:
        List of action names such as 'Up' or 'Right'.
    """
    offsets = {(dr, dc): name for name, dr, dc in DIRECTIONS}
    actions = []

    for current, nxt in zip(path, path[1:]):
        actions.append(offsets[(nxt[0] - current[0], nxt[1] - current[1])])

    return actions


def execute_plan(grid, start, actions):
    """
    Carry out a plan one action at a time.

    This is what makes the agent goal-based rather than a planner
    only: the plan is executed and the state is updated after
    every step.

    Parameters:
        grid    : 2D warehouse grid
        start   : (row, column) of starting position
        actions : list of action names to perform

    Returns:
        List of (step, action, new_position) tuples.
    """
    offsets = {name: (dr, dc) for name, dr, dc in DIRECTIONS}
    current = start
    trace = []

    for step, action in enumerate(actions, start=1):
        dr, dc = offsets[action]
        current = (current[0] + dr, current[1] + dc)
        trace.append((step, action, current))

    return trace


def validate_path(grid, path, start, goal):
    """
    Check that a returned path is actually legal.

    Parameters:
        grid  : 2D warehouse grid
        path  : list of positions from start to goal
        start : expected first cell
        goal  : expected last cell

    Returns:
        List of (description, passed) pairs.
    """
    return [
        ("starts at S", bool(path) and path[0] == start),
        ("ends at G", bool(path) and path[-1] == goal),
        ("every cell is free", all(is_free(grid, c) for c in path)),
        ("one square per step",
         all(manhattan(a, b) == 1 for a, b in zip(path, path[1:]))),
        ("no cell revisited", len(set(path)) == len(path)),
    ]


def display_path(grid, path):
    """
    Display the warehouse grid with the discovered path.

    Parameters:
        grid : original warehouse grid
        path : list of positions in the path
    """
    # Make a copy so that the original grid is not modified.
    result = [row[:] for row in grid]

    # Mark the path using '*'. Keep S and G unchanged.
    for row, col in path:
        if result[row][col] not in ('S', 'G'):
            result[row][col] = '*'

    for row in result:
        print(" ".join(row))


def scale_map(text, factor):
    """
    Enlarge a map by repeating each cell, preserving its layout.

    Used to test how the search behaves when the warehouse grows.

    Parameters:
        text   : multi-line ASCII map
        factor : how many times to repeat each row and column

    Returns:
        2D grid with a single S and a single G.
    """
    rows = []

    for row in text.splitlines():
        wide = "".join(ch * factor for ch in row)
        for _ in range(factor):
            rows.append(list(wide))

    # Duplicating cells also duplicates S and G, so keep only the
    # first of each and turn the copies into free space.
    seen = set()

    for row in range(len(rows)):
        for col in range(len(rows[0])):
            if rows[row][col] in ('S', 'G'):
                if rows[row][col] in seen:
                    rows[row][col] = '.'
                else:
                    seen.add(rows[row][col])

    return rows


def main():
    """
    Run the goal-based agent and the comparisons for the lab.
    """

    # --------------------------------------------------------
    # Task 1: the problem
    # --------------------------------------------------------
    grid = parse_grid(WAREHOUSE_MAP)
    start = find_position(grid, 'S')
    goal = find_position(grid, 'G')

    print("Grid size      :", len(grid), "x", len(grid[0]))
    print("Free cells     :", free_cells(grid))
    print("Start          :", start)
    print("Goal           :", goal)
    print("Manhattan S-G  :", manhattan(start, goal), "moves (lower bound)")

    # --------------------------------------------------------
    # Task 3: plan a route and carry it out
    # --------------------------------------------------------
    path, expanded = bfs(grid, start, goal)

    if path is None:
        print("\nNo collision-free path exists from S to G.")
        return

    actions = path_to_actions(path)
    trace = execute_plan(grid, start, actions)

    print("\nPath found!")
    print("Moves          :", len(path) - 1)
    print("Cells expanded :", expanded)

    print("\nWarehouse with path:")
    display_path(grid, path)

    print("\nActions:", " ".join(actions))

    print("\nFirst five steps:")
    for step, action, position in trace[:5]:
        print(f"  {step}: {action:<5} -> {position}")

    print("Final position :", trace[-1][2])
    print("Goal reached   :", trace[-1][2] == goal)

    print("\nValidation:")
    for description, passed in validate_path(grid, path, start, goal):
        print(f"  {'PASS' if passed else 'FAIL'}  {description}")

    # --------------------------------------------------------
    # Comparison of search strategies
    # --------------------------------------------------------
    print("\nStrategy comparison:")
    print(f"  {'Strategy':<10}{'Moves':>8}{'Expanded':>10}")

    for name, search in (("BFS", bfs), ("DFS", dfs), ("A*", astar)):
        found, count = search(grid, start, goal)
        print(f"  {name:<10}{len(found) - 1:>8}{count:>10}")

    # --------------------------------------------------------
    # Negative test: the goal is unreachable
    # --------------------------------------------------------
    sealed = parse_grid(SEALED_MAP)
    sealed_start = find_position(sealed, 'S')
    sealed_goal = find_position(sealed, 'G')
    sealed_path, sealed_expanded = bfs(sealed, sealed_start, sealed_goal)

    print("\nSealed-goal test:")
    if sealed_path is None:
        print("  No collision-free path exists from S to G.")
    print("  Cells expanded :", sealed_expanded)
    print("  Free cells     :", free_cells(sealed))

    # --------------------------------------------------------
    # How the search behaves as the warehouse grows
    # --------------------------------------------------------
    print("\nScaling behaviour:")
    print(f"  {'Scale':<7}{'Grid':>10}{'Free':>7}{'Moves':>7}"
          f"{'BFS':>7}{'A*':>7}")

    for factor in (1, 2, 4):
        big = scale_map(WAREHOUSE_MAP, factor)
        big_start = find_position(big, 'S')
        big_goal = find_position(big, 'G')

        bfs_path, bfs_expanded = bfs(big, big_start, big_goal)
        _, astar_expanded = astar(big, big_start, big_goal)

        size = f"{len(big)}x{len(big[0])}"
        print(f"  {'x' + str(factor):<7}{size:>10}{free_cells(big):>7}"
              f"{len(bfs_path) - 1:>7}{bfs_expanded:>7}"
              f"{astar_expanded:>7}")


# ------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
