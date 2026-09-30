"""
Artificial Intelligence - Agents Laboratory Exercise
Constructing a Goal-Based Agent using a Large Language Model

Problem: an autonomous warehouse vehicle must move from the loading bay (S) to
the dispatch area (G) on a grid map containing impassable shelving (#), using
the four moves Up, Down, Left and Right.

This file implements the design specified in Task 2:

  Environment            -> WarehouseEnvironment (the grid and what is passable)
  Current state          -> the agent's (row, col) position
  Goal                   -> the goal test, position == G
  Actions                -> Up / Down / Left / Right, one square each
  Decision-making        -> GoalBasedAgent.plan(), a breadth-first search, then
                            act(), which executes the plan one action at a time

It also runs the tests and diagnostics reported in the lab report:
  - validation that the returned path is legal and actually reaches the goal;
  - a comparison of BFS against DFS and A* on the same map;
  - an unreachable-goal case, to check the "no path exists" message;
  - a scaling experiment for the "what if the warehouse doubles" question.

Run:  python agents_lab_warehouse.py
"""

import heapq
import time
from collections import deque

# --------------------------------------------------------------------------
# The warehouse map, exactly as given in the laboratory sheet.
# --------------------------------------------------------------------------

WAREHOUSE_MAP = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# The four available actions, as (row, col) offsets. One move changes the
# vehicle's position by one grid square.
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

OBSTACLE = "#"


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# The environment
# --------------------------------------------------------------------------

class WarehouseEnvironment:
    """The warehouse: a 2-D grid of free cells and impassable shelving.

    The environment owns the map and answers questions about it. It does not
    decide anything; deciding is the agent's job.
    """

    def __init__(self, text=WAREHOUSE_MAP):
        self.grid = [list(row) for row in text.splitlines()]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        if any(len(r) != self.cols for r in self.grid):
            raise ValueError("map rows have inconsistent lengths")
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == symbol:
                    return (r, c)
        return None

    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell):
        """A cell is passable if it is on the map and is not shelving."""
        r, c = cell
        return self.in_bounds(cell) and self.grid[r][c] != OBSTACLE

    def result(self, state, action):
        """The successor state produced by taking `action` in `state`."""
        dr, dc = ACTIONS[action]
        return (state[0] + dr, state[1] + dc)

    def legal_actions(self, state):
        """The actions that do not walk into shelving or off the map."""
        return [a for a in ACTIONS if self.is_free(self.result(state, a))]

    def free_cells(self):
        return sum(1 for r in range(self.rows) for c in range(self.cols)
                   if self.grid[r][c] != OBSTACLE)

    def render(self, path=None):
        """Return the map as text, with any supplied path drawn in as '*'."""
        out = [row[:] for row in self.grid]
        for cell in (path or []):
            if cell != self.start and cell != self.goal:
                out[cell[0]][cell[1]] = "*"
        return "\n".join("".join(row) for row in out)


# --------------------------------------------------------------------------
# Search strategies used by the decision-making component
# --------------------------------------------------------------------------

def bfs(env, start, goal):
    """Breadth-first search. Returns (path, nodes_expanded).

    BFS explores the frontier in order of increasing number of moves, so on a
    grid where every move has the same cost it returns a path with the fewest
    possible moves. `path` is None if the goal cannot be reached.
    """
    if start is None or goal is None:
        return None, 0
    frontier = deque([start])
    came_from = {start: None}
    expanded = 0
    while frontier:
        current = frontier.popleft()
        expanded += 1
        if current == goal:
            return reconstruct(came_from, goal), expanded
        for action in ACTIONS:
            nxt = env.result(current, action)
            if env.is_free(nxt) and nxt not in came_from:
                came_from[nxt] = current
                frontier.append(nxt)
    return None, expanded


def dfs(env, start, goal):
    """Depth-first search, for comparison only. Returns (path, expanded).

    DFS follows one branch as far as it can before backtracking, so it finds
    *a* path but not necessarily a short one.
    """
    stack = [start]
    came_from = {start: None}
    expanded = 0
    while stack:
        current = stack.pop()
        expanded += 1
        if current == goal:
            return reconstruct(came_from, goal), expanded
        for action in ACTIONS:
            nxt = env.result(current, action)
            if env.is_free(nxt) and nxt not in came_from:
                came_from[nxt] = current
                stack.append(nxt)
    return None, expanded


def manhattan(a, b):
    """Straight-line grid distance ignoring obstacles: an admissible heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(env, start, goal):
    """A* with the Manhattan heuristic. Returns (path, expanded).

    Because the heuristic never overestimates the true remaining distance, A*
    returns a shortest path like BFS, but it expands fewer cells because it
    prefers cells that appear to lead towards the goal.
    """
    counter = 0
    frontier = [(manhattan(start, goal), 0, counter, start)]
    came_from = {start: None}
    cost = {start: 0}
    expanded = 0
    while frontier:
        _, g, _, current = heapq.heappop(frontier)
        expanded += 1
        if current == goal:
            return reconstruct(came_from, goal), expanded
        for action in ACTIONS:
            nxt = env.result(current, action)
            if not env.is_free(nxt):
                continue
            new_cost = g + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                came_from[nxt] = current
                counter += 1
                heapq.heappush(
                    frontier,
                    (new_cost + manhattan(nxt, goal), new_cost, counter, nxt))
    return None, expanded


def reconstruct(came_from, goal):
    """Walk the parent pointers backwards to build the path start -> goal."""
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


# --------------------------------------------------------------------------
# The goal-based agent
# --------------------------------------------------------------------------

class GoalBasedAgent:
    """A goal-based agent for the warehouse navigation problem.

    The agent keeps an explicit goal and an internal model of the warehouse,
    forms a plan that reaches the goal, and then executes that plan one action
    at a time. A simple reflex agent could not do this: choosing a direction
    at the current square from local percepts alone gives no way to tell a
    corridor that leads to the dispatch area from one that dead-ends.
    """

    def __init__(self, env, strategy=bfs):
        self.env = env                  # internal model of the environment
        self.state = env.start          # current state: where the vehicle is
        self.goal = env.goal            # explicit goal
        self.strategy = strategy        # decision-making component
        self.plan = []                  # remaining actions
        self.path = []                  # cells the plan passes through
        self.expanded = 0
        self.trace = []                 # (step, action, state) actually executed

    def goal_reached(self):
        """The goal test."""
        return self.state == self.goal

    def plan_route(self):
        """Run the decision-making component to produce a plan of actions."""
        path, expanded = self.strategy(self.env, self.state, self.goal)
        self.expanded = expanded
        if path is None:
            self.plan, self.path = [], []
            return False
        self.path = path
        self.plan = path_to_actions(path)
        return True

    def act(self):
        """Execute the next action of the plan, updating the internal state."""
        if not self.plan:
            return None
        action = self.plan.pop(0)
        self.state = self.env.result(self.state, action)
        self.trace.append((len(self.trace) + 1, action, self.state))
        return action

    def run(self):
        """Plan, then act until the goal is reached or the plan runs out."""
        if not self.plan_route():
            return False
        while self.plan:
            self.act()
        return self.goal_reached()


def path_to_actions(path):
    """Convert a list of cells into the action names that produce it."""
    offsets = {v: k for k, v in ACTIONS.items()}
    actions = []
    for a, b in zip(path, path[1:]):
        actions.append(offsets[(b[0] - a[0], b[1] - a[1])])
    return actions


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(env, path):
    """Check that a path is actually legal. Returns a list of (check, passed)."""
    checks = []
    checks.append(("starts at S", bool(path) and path[0] == env.start))
    checks.append(("ends at G", bool(path) and path[-1] == env.goal))
    checks.append(("every cell is free",
                   all(env.is_free(c) for c in path)))
    checks.append(("no cell is an obstacle",
                   all(env.grid[r][c] != OBSTACLE for r, c in path)))
    checks.append(("each step moves exactly one square",
                   all(manhattan(a, b) == 1 for a, b in zip(path, path[1:]))))
    checks.append(("no cell is revisited", len(set(path)) == len(path)))
    return checks


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------

def task1(env):
    line("TASK 1 - UNDERSTANDING THE PROBLEM")
    print("The warehouse map:")
    print(env.render())
    print()
    print(f"1. Environment : a {env.rows} x {env.cols} grid of "
          f"{env.rows * env.cols} cells, of which {env.free_cells()} are free")
    print("                 and the rest are impassable shelving. It is fully")
    print("                 observable, static, deterministic and discrete.")
    print(f"2. Goal        : move the vehicle from S at {env.start} to G at "
          f"{env.goal}.")
    print(f"3. Actions     : {', '.join(ACTIONS)} - one grid square per move.")
    print("4. Information : the agent must maintain its current position, its")
    print("                 goal position, a model of which cells are blocked,")
    print("                 the set of cells already visited, and the partial")
    print("                 path or plan built so far.")
    print("5. Goal-based  : the agent selects actions by reasoning about")
    print("                 whether they lead to the goal, not by reacting to")
    print("                 the current square. A reflex rule such as 'move")
    print("                 right if right is free' cannot distinguish a")
    print("                 corridor that reaches dispatch from a dead end.")
    print()
    print(f"Legal actions at the start {env.start}: "
          f"{env.legal_actions(env.start)}")
    print(f"Manhattan distance from S to G (ignoring shelving): "
          f"{manhattan(env.start, env.goal)} moves - a lower bound on any path.")


def task3_run(env):
    line("TASK 3 - RUNNING AND TESTING THE GENERATED PROGRAM")
    agent = GoalBasedAgent(env, strategy=bfs)
    reached = agent.run()

    if not agent.path:
        print("No path exists from S to G.")
        return None

    print(f"Path found: {len(agent.path)} cells, "
          f"{len(agent.path) - 1} moves, "
          f"{agent.expanded} cells expanded during search.")
    print()
    print("Path drawn on the map (* marks the route):")
    print(env.render(agent.path))
    print()
    print("Action sequence:")
    actions = [a for _, a, _ in agent.trace]
    print("  " + " -> ".join(actions))
    print()
    print("First five executed steps:")
    for step, action, state in agent.trace[:5]:
        print(f"  step {step}: {action:<5} -> now at {state}")
    print(f"  ... goal reached at step {len(agent.trace)}, "
          f"final state {agent.state}, goal test = {agent.goal_reached()}")

    print()
    print("Validation of the returned path:")
    for name, passed in validate(env, agent.path):
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"  {'PASS' if reached else 'FAIL'}  agent actually executed the "
          f"plan and reached G")
    lower = manhattan(env.start, env.goal)
    moves = len(agent.path) - 1
    print(f"  {'PASS' if moves >= lower else 'FAIL'}  path length {moves} is "
          f"at least the obstacle-free lower bound {lower}")
    return agent


def compare_strategies(env):
    line("ALGORITHM COMPARISON ON THE SAME MAP")
    print(f"{'Strategy':<10}{'Moves':>8}{'Cells expanded':>18}"
          f"{'Shortest?':>12}")
    best = None
    results = {}
    for name, fn in [("BFS", bfs), ("DFS", dfs), ("A*", astar)]:
        path, expanded = fn(env, env.start, env.goal)
        moves = len(path) - 1 if path else None
        results[name] = (moves, expanded)
        if best is None or (moves is not None and moves < best):
            best = moves if name == "BFS" else best
    for name, (moves, expanded) in results.items():
        shortest = "yes" if moves == results["BFS"][0] else "no"
        print(f"{name:<10}{moves:>8}{expanded:>18}{shortest:>12}")
    print()
    print("BFS and A* both return a shortest path because every move costs the")
    print("same and the Manhattan heuristic never overestimates the distance")
    print("remaining. DFS returns a valid but longer path, because it commits")
    print("to one branch and only backtracks when it gets stuck. A* reaches")
    print("the same answer as BFS while expanding fewer cells, since it")
    print("prefers cells that appear to lead towards the goal.")
    return results


def unreachable_case():
    line("NEGATIVE TEST - GOAL SEALED OFF BY SHELVING")
    sealed = """\
#####################
#S....#...........#G#
#.##....##########.##
#....##............##
#.######.###.#.###.##
#........#.........##
#####################"""
    env = WarehouseEnvironment(sealed)
    print(env.render())
    agent = GoalBasedAgent(env, strategy=bfs)
    reached = agent.run()
    print()
    print(f"Planning succeeded: {reached}")
    if not agent.path:
        print("Reported result: No path exists from S to G.")
    print(f"Cells expanded before exhausting the frontier: {agent.expanded}")
    print(f"Free cells reachable from S: {agent.expanded} of "
          f"{env.free_cells()} free cells on the map.")
    print("The agent terminates and reports failure rather than looping, which")
    print("is the behaviour the specification asked for.")


def scaling_experiment(env):
    line("THINK ABOUT IT - WHAT IF THE WAREHOUSE DOUBLES IN SIZE?")

    def scale(text, k):
        """Enlarge the map by a factor k, preserving its connectivity."""
        rows = text.splitlines()
        out = []
        for row in rows:
            wide = "".join(ch * k for ch in row)
            for _ in range(k):
                out.append(wide)
        # Keep exactly one S and one G after duplication.
        grid = [list(r) for r in out]
        seen = {"S": False, "G": False}
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                ch = grid[r][c]
                if ch in seen:
                    if seen[ch]:
                        grid[r][c] = "."
                    else:
                        seen[ch] = True
        return "\n".join("".join(r) for r in grid)

    print(f"{'Scale':<8}{'Grid':>12}{'Free cells':>13}{'Moves':>8}"
          f"{'BFS expanded':>15}{'A* expanded':>14}")
    for k in (1, 2, 4):
        e = WarehouseEnvironment(scale(WAREHOUSE_MAP, k) if k > 1
                                 else WAREHOUSE_MAP)
        bp, bx = bfs(e, e.start, e.goal)
        ap, ax = astar(e, e.start, e.goal)
        print(f"{'x' + str(k):<8}{f'{e.rows}x{e.cols}':>12}"
              f"{e.free_cells():>13}{len(bp) - 1:>8}{bx:>15}{ax:>14}")
    print()
    print("Doubling each side multiplies the number of cells by about four, and")
    print("the cells BFS expands grows in proportion, because BFS explores")
    print("essentially the whole reachable area before it happens to reach the")
    print("goal. The strategy stays correct but stops being economical.")
    print("A* expands far fewer cells at every size, and its advantage widens")
    print("as the map grows, which is the practical reason to prefer an")
    print("informed search once the warehouse is large.")
    print()
    print("Further difficulties at larger scale: memory, since BFS must hold")
    print("the whole frontier and visited set; other vehicles, which make the")
    print("environment dynamic so a single plan computed once goes stale;")
    print("and turning or acceleration costs, which break the assumption that")
    print("every move costs the same and so require a weighted search.")


def main():
    env = WarehouseEnvironment()
    task1(env)
    task3_run(env)
    compare_strategies(env)
    unreachable_case()
    scaling_experiment(env)
    line("END OF LABORATORY RUN")


if __name__ == "__main__":
    main()
