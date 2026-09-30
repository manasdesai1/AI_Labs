"""
Artificial Intelligence Laboratory: Search - A* and Blind Search
Warehouse Robot Navigation

A robot moves from the loading area S to the delivery area G on an ASCII grid.
It may move Up, Down, Left or Right, and every move costs 1.

The search problem is P = (S, A, T, s0, G, c), implemented here as:

  S  (states)      -> (row, col) coordinates of free cells       -> Grid.free_cells
  A  (actions)     -> Up / Down / Left / Right                   -> ACTIONS
  T  (transition)  -> Grid.result(state, action)                 -> Grid.result
  s0 (initial)     -> the cell marked S                          -> Grid.start
  G  (goals)       -> the cell marked G                          -> Grid.is_goal
  c  (cost)        -> 1 per move                                 -> STEP_COST

Contents:
  Task 0 / Task 1  the formulation and design, printed for the record
  Task 2 / Task 3  A* on the warehouse, plus the trivial, no-solution and
                   alternative-path tests
  Task 5           BFS compared against A* on the same map
  Task 6           heuristic investigation: h = 0, Manhattan, Euclidean, 2 x
                   Manhattan, with an empirical admissibility check

Run:  python search_lab_astar.py
"""

import heapq
import math
from collections import deque

# --------------------------------------------------------------------------
# The warehouse map, exactly as supplied in the laboratory sheet.
# --------------------------------------------------------------------------

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

# Task 3 test maps.
TRIVIAL = """\
#####
#SG##
#####"""

NO_SOLUTION = """\
#######
#S....#
###.###
#...#G#
#######"""

# Two routes from S to G: a short one over the top (via row 1) and a long one
# around the bottom. A correct A* must return the short one.
ALTERNATIVE = """\
#########
#S.....G#
#.#####.#
#.......#
#########"""

# Supplementary map only. The supplied warehouse is never modified; this open
# hall is used in Task 5 and Task 6 to show what a heuristic does when the map
# gives it room to discriminate, which the corridor map does not.
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

ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
OBSTACLE = "#"
STEP_COST = 1


def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------
# The environment: the state space, transition function and cost
# --------------------------------------------------------------------------

class Grid:
    """The warehouse map and the search problem defined over it."""

    def __init__(self, text):
        self.grid = [list(r) for r in text.splitlines()]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == symbol:
                    return (r, c)
        return None

    def is_free(self, cell):
        """An action is invalid if its result is off the map or an obstacle."""
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols \
            and self.grid[r][c] != OBSTACLE

    def result(self, state, action):
        """T: the state reached by applying `action` in `state`."""
        dr, dc = ACTIONS[action]
        return (state[0] + dr, state[1] + dc)

    def is_goal(self, state):
        """The goal test."""
        return state == self.goal

    def successors(self, state):
        """Yield (action, next_state, step_cost) for every valid action."""
        for action in ACTIONS:
            nxt = self.result(state, action)
            if self.is_free(nxt):
                yield action, nxt, STEP_COST

    def free_cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.grid[r][c] != OBSTACLE]

    def render(self, path=None):
        out = [row[:] for row in self.grid]
        for cell in (path or []):
            if cell not in (self.start, self.goal):
                out[cell[0]][cell[1]] = "*"
        return "\n".join("".join(r) for r in out)


# --------------------------------------------------------------------------
# Heuristics
# --------------------------------------------------------------------------

def h_zero(cell, goal):
    """h(n) = 0. A* then ignores the goal and behaves as uniform-cost search."""
    return 0


def h_manhattan(cell, goal):
    """|x - xG| + |y - yG|: the exact cost on an empty 4-connected grid."""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def h_euclidean(cell, goal):
    """Straight-line distance. Admissible here, but weaker than Manhattan."""
    return math.hypot(cell[0] - goal[0], cell[1] - goal[1])


def h_manhattan_x2(cell, goal):
    """2 x Manhattan. Deliberately inadmissible: it can overestimate."""
    return 2 * h_manhattan(cell, goal)


HEURISTICS = [
    ("h = 0", h_zero),
    ("Manhattan", h_manhattan),
    ("Euclidean", h_euclidean),
    ("2 x Manhattan", h_manhattan_x2),
]


# --------------------------------------------------------------------------
# Search algorithms
# --------------------------------------------------------------------------

def astar(grid, heuristic=h_manhattan):
    """A* search. Returns a result dictionary.

    The frontier is a binary heap (priority queue) ordered by
    f(n) = g(n) + h(n). A closed set prevents re-expanding a state, and the
    g-value test prevents queueing a worse route to a state already reached.
    """
    start, goal = grid.start, grid.goal
    counter = 0
    g = {start: 0}
    parent = {start: None}
    # Each frontier entry stores f, g, a tie-breaker and the state itself.
    frontier = [(heuristic(start, goal), 0, counter, start)]
    closed = set()
    expanded = 0
    max_frontier = 1

    while frontier:
        f, g_cur, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if grid.is_goal(current):
            return result(True, reconstruct(parent, current), expanded,
                          max_frontier, g[current])

        for action, nxt, step in grid.successors(current):
            new_g = g_cur + step
            if nxt in closed:
                continue
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                parent[nxt] = current
                counter += 1
                heapq.heappush(
                    frontier,
                    (new_g + heuristic(nxt, goal), new_g, counter, nxt))
        max_frontier = max(max_frontier, len(frontier))

    return result(False, None, expanded, max_frontier, None)


def bfs(grid):
    """Breadth-first search: blind, explores by number of moves.

    The frontier is a FIFO queue, so no cost or heuristic information is used
    to choose the next state. On a unit-cost grid it still returns a shortest
    path, because depth and cost coincide.
    """
    start = grid.start
    frontier = deque([start])
    parent = {start: None}
    depth = {start: 0}
    expanded = 0
    max_frontier = 1

    while frontier:
        current = frontier.popleft()
        expanded += 1
        if grid.is_goal(current):
            return result(True, reconstruct(parent, current), expanded,
                          max_frontier, depth[current])
        for action, nxt, step in grid.successors(current):
            if nxt not in parent:
                parent[nxt] = current
                depth[nxt] = depth[current] + 1
                frontier.append(nxt)
        max_frontier = max(max_frontier, len(frontier))

    return result(False, None, expanded, max_frontier, None)


def greedy(grid, heuristic=h_manhattan):
    """Greedy best-first search: orders the frontier by h(n) alone.

    Included to show what A* gains by keeping g(n): greedy uses the same
    information about the goal but discards the cost already paid.
    """
    start, goal = grid.start, grid.goal
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
        if grid.is_goal(current):
            path = reconstruct(parent, current)
            return result(True, path, expanded, max_frontier, len(path) - 1)
        for action, nxt, step in grid.successors(current):
            if nxt not in parent:
                parent[nxt] = current
                counter += 1
                heapq.heappush(frontier, (heuristic(nxt, goal), counter, nxt))
        max_frontier = max(max_frontier, len(frontier))

    return result(False, None, expanded, max_frontier, None)


def result(found, path, expanded, max_frontier, cost):
    return {
        "found": found,
        "path": path,
        "length": (len(path) - 1) if path else None,
        "cost": cost,
        "expanded": expanded,
        "max_frontier": max_frontier,
    }


def reconstruct(parent, goal):
    """Follow parent pointers back from the goal to build the path."""
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def distances_from(grid, source):
    """Least cost from `source` to every reachable free cell, by BFS."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        for action, nxt, step in grid.successors(cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + step
                queue.append(nxt)
    return dist


def true_distances(grid):
    """h*(n) for every free cell: the real cost to the goal, found by BFS
    backwards from G. Used to check admissibility empirically."""
    return distances_from(grid, grid.goal)


def weighted(h, w):
    """Return the heuristic h scaled by w, for the Task 6 weight sweep."""
    return lambda cell, goal: w * h(cell, goal)


def check_path(grid, path):
    """Confirm a returned path is actually legal."""
    checks = [
        ("starts at S", bool(path) and path[0] == grid.start),
        ("ends at G", bool(path) and path[-1] == grid.goal),
        ("all cells free", all(grid.is_free(c) for c in path)),
        ("steps of exactly one cell",
         all(h_manhattan(a, b) == 1 for a, b in zip(path, path[1:]))),
        ("no repeated cell", len(set(path)) == len(path)),
    ]
    return checks


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------

def task0_and_1(grid):
    line("TASK 0 AND TASK 1 - PROBLEM FORMULATION AND AGENT DESIGN")
    print("The warehouse map:")
    print(grid.render())
    print()
    print("Search problem P = (S, A, T, s0, G, c):")
    print(f"  S  states     : (row, col) of a free cell; "
          f"{len(grid.free_cells())} of {grid.rows * grid.cols} cells are free")
    print(f"  A  actions    : {', '.join(ACTIONS)}")
    print("  T  transition : T(s, a) = s + offset(a), defined only when the")
    print("                  target cell is on the map and is not an obstacle")
    print(f"  s0 initial    : {grid.start}  (the cell marked S)")
    print(f"  G  goals      : {{{grid.goal}}}  (the cell marked G)")
    print(f"  c  cost       : {STEP_COST} per move, so path cost = number of moves")
    print()
    print("(a) A state needs only the robot's (row, col) position. The map is")
    print("    static and the robot carries nothing, so no other information")
    print("    affects which actions are available or what they lead to.")
    print("(b) An action is invalid if its target cell lies outside the grid")
    print("    or contains an obstacle.")
    print("(c) Yes, it is deterministic: each action from a given state leads")
    print("    to exactly one successor state, and the map never changes.")
    print("(d) A solution is a sequence of valid actions from s0 whose final")
    print("    state is G. An optimal solution is one of least total cost,")
    print("    which here means the fewest moves.")
    print()
    print("Design decisions (Task 1):")
    print("  1. state          : a (row, col) tuple - immutable, so it can be")
    print("                      used as a dictionary key and set member")
    print("  2. warehouse      : list of lists of characters, parsed from the")
    print("                      ASCII map rather than hardcoded")
    print("  3. valid actions  : Grid.successors(), which filters the four")
    print("                      offsets through Grid.is_free()")
    print("  4. goal test      : Grid.is_goal(), an equality check against G")
    print("  5. frontier holds : (f, g, tie-breaker, state) so the heap orders")
    print("                      by f and g is available without a lookup")
    print("  6. path rebuilt   : parent dictionary, walked back from G")
    print("  Reported on exit  : solution found or not, the path, its length,")
    print("                      and the number of states expanded.")
    print(f"  Lower bound on any solution: Manhattan(s0, G) = "
          f"{h_manhattan(grid.start, grid.goal)} moves.")


def task3_tests(grid):
    line("TASK 3 - TESTING THE GENERATED PROGRAM")

    print("Test 1 - the original warehouse map")
    res = astar(grid, h_manhattan)
    print(f"  solution found   : {res['found']}")
    print(f"  path length      : {res['length']} moves (cost {res['cost']})")
    print(f"  states expanded  : {res['expanded']}")
    print(f"  peak frontier    : {res['max_frontier']}")
    print("  path drawn on the map:")
    for row in grid.render(res["path"]).splitlines():
        print("    " + row)
    print("  path as coordinates:")
    print("    " + " ".join(str(c) for c in res["path"]))
    print("  validation:")
    for name, ok in check_path(grid, res["path"]):
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    optimal = bfs(grid)["length"]
    print(f"    {'PASS' if res['length'] == optimal else 'FAIL'}  "
          f"length {res['length']} equals the BFS optimum {optimal}")

    print()
    print("Test 2 - trivial case, goal adjacent to the start")
    g2 = Grid(TRIVIAL)
    for row in g2.render().splitlines():
        print("    " + row)
    r2 = astar(g2, h_manhattan)
    print(f"  solution found   : {r2['found']}")
    print(f"  path             : {r2['path']}")
    print(f"  path length      : {r2['length']} move")
    print(f"  states expanded  : {r2['expanded']}")
    print(f"  {'PASS' if r2['length'] == 1 else 'FAIL'}  one-step solution found")

    print()
    print("Test 3 - no solution, goal walled off")
    g3 = Grid(NO_SOLUTION)
    for row in g3.render().splitlines():
        print("    " + row)
    r3 = astar(g3, h_manhattan)
    print(f"  solution found   : {r3['found']}")
    print(f"  reported         : "
          f"{'path ' + str(r3['path']) if r3['found'] else 'No path exists from S to G'}")
    print(f"  states expanded before the frontier emptied: {r3['expanded']}")
    print(f"  free cells on the map: {len(g3.free_cells())}")
    print(f"  {'PASS' if not r3['found'] else 'FAIL'}  terminates and reports "
          f"failure instead of looping")

    print()
    print("Test 4 - alternative paths, is the returned path shortest?")
    g4 = Grid(ALTERNATIVE)
    for row in g4.render().splitlines():
        print("    " + row)
    r4 = astar(g4, h_manhattan)
    b4 = bfs(g4)
    print("  the top corridor gives a 6-move route; going round the bottom")
    print("  loop gives a longer one, so the two routes are not equal")
    print(f"  A* path length   : {r4['length']} moves")
    print(f"  BFS path length  : {b4['length']} moves (independent optimum)")
    print(f"  A* route         : {' '.join(str(c) for c in r4['path'])}")
    print(f"  {'PASS' if r4['length'] == b4['length'] else 'FAIL'}  A* returned "
          f"a shortest path")
    return res


def task5_comparison(grid):
    line("TASK 5 - A* COMPARED WITH BLIND SEARCH")
    rows = []
    for name, fn in [("BFS (blind)", lambda g: bfs(g)),
                     ("A* (Manhattan)", lambda g: astar(g, h_manhattan)),
                     ("Greedy (Manhattan)", lambda g: greedy(g, h_manhattan))]:
        r = fn(grid)
        rows.append((name, r))

    print("On the supplied warehouse map (unchanged):")
    print(f"{'Measure':<22}{'Solution':>10}{'Path length':>14}"
          f"{'States expanded':>18}{'Peak frontier':>16}")
    for name, r in rows:
        print(f"{name:<22}{str(r['found']):>10}{str(r['length']):>14}"
              f"{r['expanded']:>18}{r['max_frontier']:>16}")
    total_free = len(grid.free_cells())
    print(f"  (the map has {total_free} free cells in total)")

    # Why the heuristic cannot help on this particular map.
    print()
    print("Why BFS and A* expand the same number of states here:")
    edges = sum(sum(1 for _ in grid.successors(c))
                for c in grid.free_cells()) // 2
    degrees = [sum(1 for _ in grid.successors(c)) for c in grid.free_cells()]
    print(f"  the free space is {total_free} cells joined by {edges} edges, and "
          f"{degrees.count(2)} of those cells have exactly two neighbours,")
    print("  so the warehouse is one long winding corridor with a single")
    print("  junction rather than an open area with choices.")
    g_star = distances_from(grid, grid.start)
    optimum = bfs(grid)["length"]
    below = sum(1 for c in grid.free_cells()
                if g_star[c] + h_manhattan(c, grid.goal) <= optimum)
    print(f"  A* must expand every state whose f* = g*(n) + h(n) is below the")
    print(f"  optimal cost C* = {optimum}. On this map {below} of the "
          f"{total_free} free cells")
    print("  satisfy f* <= C*, so there is provably nothing for the heuristic")
    print("  to prune. The tie is a property of the map, not a bug.")

    # The same comparison where the map does give the heuristic room to work.
    print()
    print("Supplementary map (open hall, supplied warehouse left untouched):")
    hall = Grid(OPEN_HALL)
    print(f"{'Measure':<22}{'Solution':>10}{'Path length':>14}"
          f"{'States expanded':>18}")
    hall_rows = [("BFS (blind)", bfs(hall)),
                 ("A* (Manhattan)", astar(hall, h_manhattan))]
    for name, r in hall_rows:
        print(f"{name:<22}{str(r['found']):>10}{str(r['length']):>14}"
              f"{r['expanded']:>18}")
    print(f"  (this map has {len(hall.free_cells())} free cells)")

    print()
    print("(a) Yes. BFS, A* and greedy all found a solution on the warehouse,")
    print("    and BFS and A* both found one on the open hall.")
    print("(b) BFS and A* returned paths of the same length, 40 moves. BFS is")
    print("    optimal here because every move costs 1, so depth equals cost,")
    print("    and A* is optimal because Manhattan is admissible. Greedy")
    print("    returned 48 moves, which is 8 moves worse.")
    print("(c) On the supplied map neither expanded fewer - both expanded all")
    print("    64 free cells. Greedy expanded 58. On the open hall A* expanded")
    print("    9 states against 113 for BFS.")
    print("(d) A* can expand fewer states because it orders the frontier by")
    print("    f = g + h, so a state that is cheap to reach but far from the")
    print("    goal is deprioritised, and whole regions pointing away from G")
    print("    are never opened. BFS orders by depth alone and so spreads")
    print("    outwards in every direction equally. That saving only appears")
    print("    when the map offers directions to reject, which is why it shows")
    print("    up on the open hall and not in a corridor. Greedy shows the")
    print("    other half of the story: it uses h but discards g, expands")
    print("    slightly fewer states, and loses optimality as a result.")
    return rows, hall_rows


def task6_heuristics(grid):
    line("TASK 6 - INVESTIGATING THE HEURISTIC")

    dist = true_distances(grid)
    print("Admissibility check against the true cost h*(n), computed by BFS")
    print("backwards from G over all free cells:")
    print(f"{'Heuristic':<16}{'Admissible?':>13}{'Violations':>12}"
          f"{'Worst h - h*':>15}")
    for name, h in HEURISTICS:
        violations = 0
        worst = 0.0
        for cell, true_cost in dist.items():
            over = h(cell, grid.goal) - true_cost
            if over > 1e-9:
                violations += 1
                worst = max(worst, over)
        print(f"{name:<16}{('yes' if violations == 0 else 'no'):>13}"
              f"{violations:>12}{worst:>15.2f}")
    print(f"  ({len(dist)} free cells are reachable from G and were checked.)")

    print()
    print("Search behaviour with each heuristic on the same warehouse map:")
    print(f"{'Heuristic':<16}{'Solution':>10}{'Path length':>14}"
          f"{'States expanded':>18}{'Optimal?':>11}")
    optimum = bfs(grid)["length"]
    results = {}
    for name, h in HEURISTICS:
        r = astar(grid, h)
        results[name] = r
        print(f"{name:<16}{str(r['found']):>10}{str(r['length']):>14}"
              f"{r['expanded']:>18}"
              f"{('yes' if r['length'] == optimum else 'no'):>11}")
    print(f"  (the true optimum, from BFS, is {optimum} moves)")
    print("  Every version returns 40 moves and expands all 64 cells, for the")
    print("  same reason as in Task 5: in a corridor there is nothing to prune")
    print("  and only one sensible route, so the heuristic cannot change the")
    print("  outcome. The interesting behaviour has to be provoked.")

    print()
    print("Weight sweep on the supplied map: A* with h = w x Manhattan")
    print(f"{'w':<6}{'Admissible?':>13}{'Path length':>14}"
          f"{'States expanded':>18}{'Optimal?':>11}")
    for w in (0, 1, 2, 3, 5, 10):
        h = weighted(h_manhattan, w)
        r = astar(grid, h)
        adm = all(h(c, grid.goal) - d <= 1e-9 for c, d in dist.items())
        print(f"{w:<6}{('yes' if adm else 'no'):>13}{str(r['length']):>14}"
              f"{r['expanded']:>18}"
              f"{('yes' if r['length'] == optimum else 'no'):>11}")

    print()
    print("Same four heuristics on the supplementary open hall:")
    hall = Grid(OPEN_HALL)
    hall_opt = bfs(hall)["length"]
    print(f"{'Heuristic':<16}{'Path length':>14}{'States expanded':>18}"
          f"{'Optimal?':>11}")
    for name, h in HEURISTICS:
        r = astar(hall, h)
        print(f"{name:<16}{str(r['length']):>14}{r['expanded']:>18}"
              f"{('yes' if r['length'] == hall_opt else 'no'):>11}")
    print(f"  (BFS on this map: {bfs(hall)['length']} moves, "
          f"{bfs(hall)['expanded']} states expanded)")

    print()
    print("1. h(n) = 0 removes all information about where the goal is, so f")
    print("   reduces to g and A* becomes uniform-cost search. It stays")
    print("   optimal but is uninformed: on the open hall it expands 113")
    print("   states where Manhattan expands 9. On the corridor map it makes")
    print("   no difference, because nothing can be pruned there anyway.")
    print("2. Euclidean distance is admissible on this grid, because a")
    print("   straight line is never longer than a path restricted to")
    print("   horizontal and vertical moves. It is therefore safe, but it")
    print("   underestimates more than Manhattan does, so in general it is a")
    print("   weaker guide. On these two maps it happened to expand the same")
    print("   number of states as Manhattan.")
    print("3. Multiplying Manhattan by 2 makes the heuristic inadmissible -")
    print("   18 of the 64 cells are overestimated, by up to 14 moves. On this")
    print("   map it nevertheless still returned an optimal 40-move path, so")
    print("   inadmissibility permits suboptimality rather than guaranteeing")
    print("   it. The weight sweep locates where the guarantee actually fails:")
    print("   at w = 3 and above the returned path becomes 48 moves, 8 worse")
    print("   than the optimum, and at w = 10 it also expands fewer states.")
    print("   Greedy search, which is the w -> infinity limit, gives the same")
    print("   48-move path. This is the honest version of the answer: the")
    print("   guarantee is lost as soon as w > 1, but the loss only becomes")
    print("   visible on a given map once the weight is large enough.")
    print()
    print("Think About It - too optimistic versus too aggressive:")
    print("  A heuristic that is too optimistic (h = 0 is the extreme case)")
    print("  keeps optimality but throws away guidance, so the search degrades")
    print("  towards blind search and the cost is time and memory.")
    print("  A heuristic that is too aggressive overestimates the remaining")
    print("  cost, which breaks admissibility. The search gets faster, but the")
    print("  answer is no longer guaranteed to be the best one - it trades")
    print("  correctness for speed rather than buying speed for free.")
    return results


def task4_inspection():
    line("TASK 4 - WHERE EACH CONCEPT APPEARS IN THE CODE")
    items = [
        ("State", "a (row, col) tuple; created in Grid._find and Grid.result"),
        ("Action", "the ACTIONS dictionary of four (row, col) offsets"),
        ("Transition", "Grid.result(state, action), filtered by Grid.successors"),
        ("Goal test", "Grid.is_goal(state), called just after a pop in astar"),
        ("g(n)", "the g dictionary and g_cur in astar; new_g = g_cur + step"),
        ("h(n)", "the heuristic argument, e.g. h_manhattan(nxt, goal)"),
        ("f(n)", "new_g + heuristic(nxt, goal), computed in the heappush call"),
        ("Frontier", "the `frontier` list used as a heap via heapq"),
        ("Visited states", "the `closed` set, plus the g-value test on reopening"),
        ("Path reconstruction", "the `parent` dictionary, walked by reconstruct()"),
    ]
    for concept, where in items:
        print(f"  {concept:<20} {where}")
    print()
    print("(a) The frontier is a binary heap (a priority queue) held in a list")
    print("    and managed by heapq, ordered by f and then by g.")
    print("(b) The next state to expand is the one with the smallest f value,")
    print("    taken by heapq.heappop; ties are broken by g and then by an")
    print("    insertion counter, which keeps the ordering deterministic.")
    print("(c) The heuristic is calculated when a successor is pushed onto the")
    print("    frontier, not when it is popped, so each state's h is computed")
    print("    once per insertion.")
    print("(d) Yes. f is formed explicitly as new_g + heuristic(nxt, goal) and")
    print("    stored as the first element of the heap entry.")
    print("(e) Two mechanisms. The `closed` set means a state is expanded at")
    print("    most once, and the test `nxt not in g or new_g < g[nxt]` means a")
    print("    state is only re-queued when a strictly cheaper route to it has")
    print("    been found, which stops the heap filling with worse duplicates.")


def main():
    grid = Grid(WAREHOUSE)
    task0_and_1(grid)
    task3_tests(grid)
    task4_inspection()
    task5_comparison(grid)
    task6_heuristics(grid)
    line("END OF LABORATORY RUN")


if __name__ == "__main__":
    main()
