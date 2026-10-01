from collections import deque
import heapq
import math

# ------------------------------------------------------------
# Warehouse Robot Navigation: A* and Blind Search
# ------------------------------------------------------------
# Symbols used in the grid:
#   S -> Start position
#   G -> Goal position
#   # -> Obstacle / blocked cell
#   . -> Free cell
#   * -> Path found by the agent
#
# The search problem is P = (S, A, T, s0, G, c):
#   S  states      -> (row, column) of a free cell
#   A  actions     -> Up, Down, Left, Right            (DIRECTIONS)
#   T  transition  -> successors()
#   s0 initial     -> the cell marked S
#   G  goals       -> the cell marked G                (is_goal)
#   c  cost        -> STEP_COST, one per move
# ------------------------------------------------------------


WAREHOUSE = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""


# Goal immediately next to the start.
TRIVIAL = """\
#####
#SG##
#####"""


# Goal sealed inside a pocket, so no route exists.
NO_SOLUTION = """\
#######
#S....#
###.###
#...#G#
#######"""


# Two routes of different length, to check the shorter is returned.
ALTERNATIVE = """\
#########
#S.....G#
#.#####.#
#.......#
#########"""


# Open hall used only to show what a heuristic does when the map
# gives it room to discriminate. The supplied warehouse above is
# never modified.
OPEN_HALL = """\
###############################
#.............................#
#.............................#
#.............................#
#.............................#
#.............S.......G.......#
#.............................#
#.............................#
#.............................#
#.............................#
###############################"""


DIRECTIONS = [
    ("Up",    -1,  0),
    ("Down",   1,  0),
    ("Left",   0, -1),
    ("Right",  0,  1),
]

STEP_COST = 1


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
        grid   : 2D warehouse grid
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
    Decide whether a cell may be occupied.

    An action is invalid exactly when its target cell fails this
    test, either by leaving the grid or by being an obstacle.

    Parameters:
        grid : 2D warehouse grid
        cell : (row, column) to test

    Returns:
        True if the cell is on the map and is not an obstacle.
    """
    row, col = cell

    if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
        return False

    return grid[row][col] != '#'


def is_goal(cell, goal):
    """
    The goal test.

    Parameters:
        cell : (row, column) being tested
        goal : (row, column) of the goal

    Returns:
        True if the cell is the goal.
    """
    return cell == goal


def successors(grid, cell):
    """
    The transition function, applied to every action in turn.

    Parameters:
        grid : 2D warehouse grid
        cell : (row, column) to expand

    Returns:
        List of (action, next_cell, step_cost) for valid actions.
    """
    result = []

    for name, dr, dc in DIRECTIONS:
        neighbor = (cell[0] + dr, cell[1] + dc)

        if is_free(grid, neighbor):
            result.append((name, neighbor, STEP_COST))

    return result


def free_cells(grid):
    """
    List the cells the robot may occupy.

    Parameters:
        grid : 2D warehouse grid

    Returns:
        List of (row, column) positions.
    """
    return [(r, c)
            for r in range(len(grid))
            for c in range(len(grid[0]))
            if grid[r][c] != '#']


def h_zero(cell, goal):
    """
    The uninformed heuristic h(n) = 0.

    With this heuristic f reduces to g and A* becomes
    uniform-cost search.
    """
    return 0


def h_manhattan(cell, goal):
    """
    Manhattan distance |x - xG| + |y - yG|.

    With four-way unit-cost moves any path must cover the row and
    column differences separately, so this is the exact cost on an
    empty grid and an underestimate once obstacles are added.
    """
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def h_euclidean(cell, goal):
    """
    Straight-line distance. Admissible here, but weaker than
    Manhattan because it underestimates by more.
    """
    return math.hypot(cell[0] - goal[0], cell[1] - goal[1])


def scaled(heuristic, weight):
    """
    Multiply a heuristic by a constant.

    A weight above 1 can overestimate the remaining cost, which
    breaks admissibility and so gives up the optimality guarantee.

    Parameters:
        heuristic : function of (cell, goal)
        weight    : multiplier to apply

    Returns:
        A new heuristic function.
    """
    return lambda cell, goal: weight * heuristic(cell, goal)


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

    path.reverse()

    return path


def search_result(path, expanded, max_frontier, cost):
    """
    Package the figures the lab asks every run to report.

    Parameters:
        path         : list of positions, or None if no route
        expanded     : number of states removed and expanded
        max_frontier : largest frontier size seen
        cost         : total path cost, or None

    Returns:
        Dictionary of results.
    """
    return {
        "found": path is not None,
        "path": path,
        "length": (len(path) - 1) if path else None,
        "cost": cost,
        "expanded": expanded,
        "max_frontier": max_frontier,
    }


def astar(grid, start, goal, heuristic=h_manhattan):
    """
    A* search, expanding states in order of f(n) = g(n) + h(n).

    The frontier is a binary heap. A closed set stops a state
    being expanded twice, and the g-value test stops a worse
    route to a known state being queued at all.

    Parameters:
        grid      : 2D warehouse grid
        start     : (row, column) of starting position
        goal      : (row, column) of goal position
        heuristic : function estimating the cost from a cell to G

    Returns:
        Result dictionary from search_result().
    """
    # The counter breaks ties deterministically and stops Python
    # comparing the position tuples when f and g are both equal.
    counter = 0

    frontier = [(heuristic(start, goal), 0, counter, start)]
    g_cost = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    max_frontier = 1

    while frontier:

        f, g, _, current = heapq.heappop(frontier)

        # A stale duplicate left behind by an earlier push.
        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        # The goal test is applied on removal, not on generation,
        # because a cheap-looking route may be found first.
        if is_goal(current, goal):
            return search_result(reconstruct_path(parent, current),
                                 expanded, max_frontier, g_cost[current])

        for name, neighbor, step in successors(grid, current):

            if neighbor in closed:
                continue

            new_g = g + step

            if neighbor in g_cost and new_g >= g_cost[neighbor]:
                continue

            g_cost[neighbor] = new_g
            parent[neighbor] = current
            counter += 1

            heapq.heappush(
                frontier,
                (new_g + heuristic(neighbor, goal), new_g, counter, neighbor)
            )

        max_frontier = max(max_frontier, len(frontier))

    return search_result(None, expanded, max_frontier, None)


def bfs(grid, start, goal):
    """
    Breadth-First Search: blind, ordered by number of moves.

    The frontier is a FIFO queue, so no cost or heuristic
    information is used to choose what to expand next. On a
    unit-cost grid this still returns a shortest path, because
    depth and cost coincide.

    Parameters:
        grid  : 2D warehouse grid
        start : (row, column) of starting position
        goal  : (row, column) of goal position

    Returns:
        Result dictionary from search_result().
    """
    frontier = deque([start])
    parent = {start: None}
    depth = {start: 0}
    expanded = 0
    max_frontier = 1

    while frontier:

        current = frontier.popleft()
        expanded += 1

        if is_goal(current, goal):
            return search_result(reconstruct_path(parent, current),
                                 expanded, max_frontier, depth[current])

        for name, neighbor, step in successors(grid, current):

            if neighbor in parent:
                continue

            parent[neighbor] = current
            depth[neighbor] = depth[current] + step
            frontier.append(neighbor)

        max_frontier = max(max_frontier, len(frontier))

    return search_result(None, expanded, max_frontier, None)


def greedy(grid, start, goal, heuristic=h_manhattan):
    """
    Greedy best-first search, ordered by h(n) alone.

    Included to show what A* gains by keeping g(n): greedy uses
    the same information about the goal but ignores the cost
    already paid, so it can return a longer path.

    Parameters:
        grid      : 2D warehouse grid
        start     : (row, column) of starting position
        goal      : (row, column) of goal position
        heuristic : function estimating the cost from a cell to G

    Returns:
        Result dictionary from search_result().
    """
    counter = 0

    frontier = [(heuristic(start, goal), counter, start)]
    parent = {start: None}
    closed = set()
    expanded = 0
    max_frontier = 1

    while frontier:

        _, _, current = heapq.heappop(frontier)

        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        if is_goal(current, goal):
            path = reconstruct_path(parent, current)
            return search_result(path, expanded, max_frontier, len(path) - 1)

        for name, neighbor, step in successors(grid, current):

            if neighbor in parent:
                continue

            parent[neighbor] = current
            counter += 1

            heapq.heappush(frontier,
                           (heuristic(neighbor, goal), counter, neighbor))

        max_frontier = max(max_frontier, len(frontier))

    return search_result(None, expanded, max_frontier, None)


def distances_from(grid, source):
    """
    Least cost from one cell to every reachable cell, by BFS.

    Running this backwards from G gives the true remaining cost
    h*(n), which is what admissibility must be checked against.

    Parameters:
        grid   : 2D warehouse grid
        source : (row, column) to measure from

    Returns:
        Dictionary mapping cell to least cost from source.
    """
    distance = {source: 0}
    queue = deque([source])

    while queue:
        current = queue.popleft()

        for name, neighbor, step in successors(grid, current):
            if neighbor not in distance:
                distance[neighbor] = distance[current] + step
                queue.append(neighbor)

    return distance


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
        ("all cells free", all(is_free(grid, c) for c in path)),
        ("one cell per step",
         all(h_manhattan(a, b) == 1 for a, b in zip(path, path[1:]))),
        ("no repeated cell", len(set(path)) == len(path)),
    ]


def display_path(grid, path):
    """
    Display the warehouse grid with the discovered path.

    Parameters:
        grid : original warehouse grid
        path : list of positions in the path
    """
    result = [row[:] for row in grid]

    for row, col in path:
        if result[row][col] not in ('S', 'G'):
            result[row][col] = '*'

    for row in result:
        print(" ".join(row))


def run_map(text, label):
    """
    Run A* on a map and print the figures required by Task 3.

    Parameters:
        text  : multi-line ASCII map
        label : name to print for this test

    Returns:
        (grid, start, goal, result) for any further checks.
    """
    grid = parse_grid(text)
    start = find_position(grid, 'S')
    goal = find_position(grid, 'G')
    result = astar(grid, start, goal)

    print(f"\n{label}")
    print("  solution found :", result["found"])

    if result["found"]:
        print("  path length    :", result["length"], "moves")
        print("  states expanded:", result["expanded"])
    else:
        print("  No path exists from S to G.")
        print("  states expanded:", result["expanded"])
        print("  free cells     :", len(free_cells(grid)))

    return grid, start, goal, result


def main():
    """
    Run the formulation summary, the tests and the experiments.
    """

    grid = parse_grid(WAREHOUSE)
    start = find_position(grid, 'S')
    goal = find_position(grid, 'G')
    cells = free_cells(grid)

    # --------------------------------------------------------
    # Task 0: the search problem
    # --------------------------------------------------------
    print("Grid size     :", len(grid), "x", len(grid[0]))
    print("Free cells    :", len(cells))
    print("Start s0      :", start)
    print("Goal G        :", goal)
    print("Step cost c   :", STEP_COST)
    print("Actions A     :", ", ".join(name for name, dr, dc in DIRECTIONS))
    print("Manhattan s0-G:", h_manhattan(start, goal), "moves (lower bound)")

    # --------------------------------------------------------
    # Task 3: tests
    # --------------------------------------------------------
    result = astar(grid, start, goal)
    optimum = bfs(grid, start, goal)["length"]

    print("\nTest 1: original warehouse")
    print("  solution found :", result["found"])
    print("  path length    :", result["length"], "moves")
    print("  states expanded:", result["expanded"])
    print("  peak frontier  :", result["max_frontier"])

    print("\nWarehouse with path:")
    display_path(grid, result["path"])

    print("\nValidation:")
    for description, passed in validate_path(grid, result["path"],
                                             start, goal):
        print(f"  {'PASS' if passed else 'FAIL'}  {description}")
    print(f"  {'PASS' if result['length'] == optimum else 'FAIL'}  "
          f"length matches the BFS optimum of {optimum}")

    trivial_grid, t_start, t_goal, trivial = run_map(
        TRIVIAL, "Test 2: goal adjacent to the start")
    print("  path           :", trivial["path"])

    run_map(NO_SOLUTION, "Test 3: goal sealed off")

    alt_grid, a_start, a_goal, alt = run_map(
        ALTERNATIVE, "Test 4: two routes of different length")
    alt_optimum = bfs(alt_grid, a_start, a_goal)["length"]
    print(f"  BFS optimum    : {alt_optimum} moves")
    print(f"  {'PASS' if alt['length'] == alt_optimum else 'FAIL'}  "
          f"a shortest path was returned")

    # --------------------------------------------------------
    # Task 5: A* against blind search
    # --------------------------------------------------------
    print("\nTask 5: supplied warehouse")
    print(f"  {'Algorithm':<16}{'Found':>7}{'Moves':>7}{'Expanded':>10}"
          f"{'Frontier':>10}")

    runs = [
        ("BFS", bfs(grid, start, goal)),
        ("A* Manhattan", astar(grid, start, goal, h_manhattan)),
        ("Greedy", greedy(grid, start, goal, h_manhattan)),
    ]

    for name, run in runs:
        print(f"  {name:<16}{str(run['found']):>7}{run['length']:>7}"
              f"{run['expanded']:>10}{run['max_frontier']:>10}")

    # Why no heuristic can prune anything on this map: A* must
    # expand every state whose f* = g*(n) + h(n) is below C*.
    g_star = distances_from(grid, start)
    within = sum(1 for c in cells
                 if g_star[c] + h_manhattan(c, goal) <= optimum)
    degree_two = sum(1 for c in cells if len(successors(grid, c)) == 2)
    edges = sum(len(successors(grid, c)) for c in cells) // 2

    print(f"  cells with f* <= C*={optimum}: {within} of {len(cells)}")
    print(f"  free space: {len(cells)} cells, {edges} edges, "
          f"{degree_two} of degree 2")

    hall = parse_grid(OPEN_HALL)
    hall_start = find_position(hall, 'S')
    hall_goal = find_position(hall, 'G')

    print("\nTask 5: open hall (supplementary)")
    print(f"  {'Algorithm':<16}{'Moves':>7}{'Expanded':>10}")
    for name, run in (("BFS", bfs(hall, hall_start, hall_goal)),
                      ("A* Manhattan",
                       astar(hall, hall_start, hall_goal, h_manhattan))):
        print(f"  {name:<16}{run['length']:>7}{run['expanded']:>10}")
    print("  free cells     :", len(free_cells(hall)))

    # --------------------------------------------------------
    # Task 6: the heuristic
    # --------------------------------------------------------
    heuristics = [
        ("h = 0", h_zero),
        ("Manhattan", h_manhattan),
        ("Euclidean", h_euclidean),
        ("2 x Manhattan", scaled(h_manhattan, 2)),
    ]

    true_cost = distances_from(grid, goal)

    print("\nTask 6: admissibility against h*(n)")
    print(f"  {'Heuristic':<16}{'Admissible':>12}{'Violations':>12}"
          f"{'Worst excess':>14}")

    for name, heuristic in heuristics:
        excess = [heuristic(c, goal) - cost
                  for c, cost in true_cost.items()]
        violations = sum(1 for e in excess if e > 1e-9)
        worst = max(excess)
        print(f"  {name:<16}{('yes' if violations == 0 else 'no'):>12}"
              f"{violations:>12}{worst:>14.2f}")

    print("\nTask 6: behaviour on the supplied warehouse")
    print(f"  {'Heuristic':<16}{'Moves':>7}{'Expanded':>10}{'Optimal':>9}")

    for name, heuristic in heuristics:
        run = astar(grid, start, goal, heuristic)
        print(f"  {name:<16}{run['length']:>7}{run['expanded']:>10}"
              f"{('yes' if run['length'] == optimum else 'no'):>9}")

    print("\nTask 6: weight sweep, h = w x Manhattan")
    print(f"  {'w':<5}{'Admissible':>12}{'Moves':>7}{'Expanded':>10}"
          f"{'Optimal':>9}")

    for weight in (0, 1, 2, 3, 5, 10):
        heuristic = scaled(h_manhattan, weight)
        run = astar(grid, start, goal, heuristic)
        admissible = all(heuristic(c, goal) - cost <= 1e-9
                         for c, cost in true_cost.items())
        print(f"  {weight:<5}{('yes' if admissible else 'no'):>12}"
              f"{run['length']:>7}{run['expanded']:>10}"
              f"{('yes' if run['length'] == optimum else 'no'):>9}")

    hall_optimum = bfs(hall, hall_start, hall_goal)["length"]

    print("\nTask 6: behaviour on the open hall")
    print(f"  {'Heuristic':<16}{'Moves':>7}{'Expanded':>10}{'Optimal':>9}")

    for name, heuristic in heuristics:
        run = astar(hall, hall_start, hall_goal, heuristic)
        print(f"  {name:<16}{run['length']:>7}{run['expanded']:>10}"
              f"{('yes' if run['length'] == hall_optimum else 'no'):>9}")


# ------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
