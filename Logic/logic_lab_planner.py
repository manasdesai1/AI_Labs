from collections import deque

# ------------------------------------------------------------
# Logical Planning Agent for a Warehouse Robot
# ------------------------------------------------------------
# A state is a set of logical propositions, for example
#   At(Robot,A), At(Package,A), Holding(Package)
#
# An action is applicable when the state entails its
# preconditions. Applying it deletes the negative effects and
# adds the positive effects.
#
# Logic decides what is possible, BFS decides what to try.
# ------------------------------------------------------------


LOCATIONS = ['A', 'B', 'C']

# Connected pairs the robot may move between.
CONNECTIONS = [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')]

INITIAL_STATE = frozenset({'At(Robot,A)', 'At(Package,A)'})
GOAL = frozenset({'At(Package,C)'})


def make_action(name, pos_pre=(), neg_pre=(), pos_eff=(), neg_eff=()):
    """
    Build an action description.

    Parameters:
        name    : action name, e.g. 'Move(A,B)'
        pos_pre : propositions that must be present
        neg_pre : propositions that must be absent
        pos_eff : propositions added by the action
        neg_eff : propositions removed by the action

    Returns:
        Dictionary describing the action.
    """
    return {
        'name': name,
        'pos_pre': frozenset(pos_pre),
        'neg_pre': frozenset(neg_pre),
        'pos_eff': frozenset(pos_eff),
        'neg_eff': frozenset(neg_eff),
    }


def move_actions():
    """
    Build every Move action for the connected locations.

    Returns:
        List of action dictionaries.
    """
    return [
        make_action(
            f"Move({x},{y})",
            pos_pre=[f"At(Robot,{x})"],
            pos_eff=[f"At(Robot,{y})"],
            neg_eff=[f"At(Robot,{x})"],
        )
        for x, y in CONNECTIONS
    ]


def pickup_actions():
    """
    Build every PickUp action.

    The negative precondition stops the robot picking up a
    package it is already holding.

    Returns:
        List of action dictionaries.
    """
    return [
        make_action(
            f"PickUp(Package,{loc})",
            pos_pre=[f"At(Robot,{loc})", f"At(Package,{loc})"],
            neg_pre=["Holding(Package)"],
            pos_eff=["Holding(Package)"],
            neg_eff=[f"At(Package,{loc})"],
        )
        for loc in LOCATIONS
    ]


def drop_actions():
    """
    Build every Drop action.

    Returns:
        List of action dictionaries.
    """
    return [
        make_action(
            f"Drop(Package,{loc})",
            pos_pre=[f"At(Robot,{loc})", "Holding(Package)"],
            pos_eff=[f"At(Package,{loc})"],
            neg_eff=["Holding(Package)"],
        )
        for loc in LOCATIONS
    ]


def all_actions():
    """
    The full action set for the warehouse problem.

    Returns:
        List of action dictionaries.
    """
    return move_actions() + pickup_actions() + drop_actions()


def is_applicable(state, action):
    """
    Decide whether the state entails the action's preconditions.

    Parameters:
        state  : frozenset of propositions
        action : action dictionary

    Returns:
        True if every precondition is satisfied.
    """
    return (action['pos_pre'] <= state
            and not (action['neg_pre'] & state))


def apply_action(state, action):
    """
    Apply an action to a state.

    Parameters:
        state  : frozenset of propositions
        action : action dictionary

    Returns:
        The resulting state as a frozenset.
    """
    return frozenset((state - action['neg_eff']) | action['pos_eff'])


def satisfies_goal(state, goal):
    """
    Decide whether a state entails the goal.

    Parameters:
        state : frozenset of propositions
        goal  : frozenset of required propositions

    Returns:
        True if the goal holds in the state.
    """
    return goal <= state


def plan_bfs(initial, actions, goal):
    """
    Search for a shortest sequence of actions achieving the goal.

    Parameters:
        initial : starting state
        actions : list of available actions
        goal    : frozenset of required propositions

    Returns:
        (plan, states, expanded). plan is a list of action names
        and states the state after each action, or (None, None,
        expanded) if no plan exists.
    """
    queue = deque([(initial, [], [])])
    visited = {initial}
    expanded = 0

    while queue:

        state, plan, states = queue.popleft()
        expanded += 1

        if satisfies_goal(state, goal):
            return plan, states, expanded

        for action in actions:

            if not is_applicable(state, action):
                continue

            successor = apply_action(state, action)

            if successor in visited:
                continue

            visited.add(successor)
            queue.append((successor,
                          plan + [action['name']],
                          states + [successor]))

    return None, None, expanded


def validate_plan(initial, plan, actions, goal):
    """
    Re-execute a plan, checking each precondition independently.

    Parameters:
        initial : starting state
        plan    : list of action names
        actions : list of available actions
        goal    : frozenset of required propositions

    Returns:
        (checks, final_state) where checks is a list of
        (step, action, applicable) triples.
    """
    by_name = {action['name']: action for action in actions}
    state = initial
    checks = []

    for step, name in enumerate(plan, start=1):
        action = by_name[name]
        applicable = is_applicable(state, action)
        checks.append((step, name, applicable))

        if not applicable:
            break

        state = apply_action(state, action)

    return checks, state


def format_state(state):
    """
    Format a state for printing.

    Parameters:
        state : frozenset of propositions

    Returns:
        Comma-separated string of sorted propositions.
    """
    return ", ".join(sorted(state))


def run_test(label, initial, actions, goal):
    """
    Run the planner on one problem and print the results.

    Parameters:
        label   : name of the test
        initial : starting state
        actions : list of available actions
        goal    : frozenset of required propositions

    Returns:
        The plan found, or None.
    """
    plan, states, expanded = plan_bfs(initial, actions, goal)

    print(f"\n{label}")
    print("  initial :", format_state(initial))
    print("  goal    :", format_state(goal))
    print("  actions :", len(actions))

    if plan is None:
        print("  No plan found")
        print("  states expanded:", expanded)
        return None

    print("  plan    :", " -> ".join(plan))
    print("  length  :", len(plan))
    print("  states expanded:", expanded)

    print("  S0:", format_state(initial))
    for i, state in enumerate(states, start=1):
        print(f"  S{i}:", format_state(state))

    checks, final = validate_plan(initial, plan, actions, goal)
    for step, name, applicable in checks:
        print(f"  {'PASS' if applicable else 'FAIL'}  step {step}: {name}")
    print(f"  {'PASS' if satisfies_goal(final, goal) else 'FAIL'}  "
          f"goal holds in the final state")

    return plan


def main():
    """
    Run the warehouse planning problem and the three tests.
    """
    actions = all_actions()

    # Task 0: which actions are applicable in the initial state.
    print("Initial state:", format_state(INITIAL_STATE))
    print("Applicable actions in I:")
    for action in actions:
        if is_applicable(INITIAL_STATE, action):
            print("  ", action['name'])

    for name in ("PickUp(Package,A)", "Drop(Package,C)"):
        action = next(a for a in actions if a['name'] == name)
        missing = action['pos_pre'] - INITIAL_STATE
        print(f"  {name}: applicable ="
              f" {is_applicable(INITIAL_STATE, action)}"
              f", unmet preconditions = {sorted(missing) or 'none'}")

    # Test A: the original problem.
    run_test("Test A: original warehouse problem",
             INITIAL_STATE, actions, GOAL)

    # Test B: PickUp removed, so the package cannot be carried.
    run_test("Test B: PickUp removed",
             INITIAL_STATE, move_actions() + drop_actions(), GOAL)

    # Test C: only Move actions, so the robot can reach C but the
    # package cannot.
    run_test("Test C: only Move actions, goal At(Package,C)",
             INITIAL_STATE, move_actions(), GOAL)

    run_test("Test C: only Move actions, goal At(Robot,C)",
             INITIAL_STATE, move_actions(), frozenset({'At(Robot,C)'}))

    # The sequence suggested in the laboratory sheet.
    suggested = ["Move(A,B)", "PickUp(Package,B)",
                 "Move(B,C)", "Drop(Package,C)"]
    checks, final = validate_plan(INITIAL_STATE, suggested, actions, GOAL)

    print("\nChecking the sequence suggested in the lab sheet")
    for step, name, applicable in checks:
        print(f"  {'PASS' if applicable else 'FAIL'}  step {step}: {name}")
    print("  goal reached:", satisfies_goal(final, GOAL))


# ------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
