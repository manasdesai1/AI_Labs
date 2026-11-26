from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Reinforcement Learning and Deep RL: Robots in a Warehouse
# ------------------------------------------------------------
# A 5x5 warehouse with two rooms joined by one doorway at (2,2).
#
#   . . # . .
#   . . # . .
#   S . . . G
#   . . # . .
#   . . # . .
#
# Part I   : tabular Q-learning for one robot
# Part II  : the Q-table replaced by a Deep Q-Network
# Part III : one DQN controller for two robots that must not
#            collide
#
# No path search is given to any learner. The BFS below is used
# only to report the shortest route for comparison.
# ------------------------------------------------------------


SEED = 7
SIZE = 5
OBSTACLES = {(0, 2), (1, 2), (3, 2), (4, 2)}
START = (2, 0)
GOAL = (2, 4)
GAMMA = 0.95
MAX_STEPS = 40

# Action codes 0..4 in the order the laboratory specifies.
ACTION_NAMES = ["up", "down", "left", "right", "wait"]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
N_ACTIONS = 5

# Part I settings
Q_EPISODES = 1500
Q_DECAY_EPISODES = 1200
ALPHA = 0.2
EPS_START = 1.0
EPS_END = 0.05
MA_WINDOW = 50

# Part II and III settings
DQN_STEPS = 12000
TWO_ROBOT_STEPS = 60000
REPLAY_CAPACITY = 20000
BATCH_SIZE = 64
LEARNING_STARTS = 500
TRAIN_EVERY = 4
TARGET_SYNC = 250
DQN_LR = 0.001
EPS_DECAY_FRACTION = 0.8

# Two-robot task
B_START = (2, 4)
B_GOAL = (2, 0)
COLLISION_REWARD = -10.0


def set_seeds(seed):
    """
    Seed every random number generator used by the program.

    Parameters:
        seed : integer seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move(position, action):
    """
    Apply one action to one robot, ignoring other robots.

    Parameters:
        position : (row, column) before the action
        action   : action code 0..4

    Returns:
        (new_position, invalid). A move into an obstacle or off
        the grid leaves the robot in place and is invalid.
    """
    dr, dc = MOVES[action]
    row, col = position[0] + dr, position[1] + dc

    if not (0 <= row < SIZE and 0 <= col < SIZE) or (row, col) in OBSTACLES:
        return position, True

    return (row, col), False


def state_index(position):
    """
    Index of a cell in the 25-row Q-table: 5r + c.
    """
    return SIZE * position[0] + position[1]


class Warehouse:
    """
    Single-robot environment.

    step() returns (next_state, reward, terminated, truncated,
    info). Termination means the goal was reached. Truncation is
    only the 40-step collection cutoff.
    """

    def __init__(self, start=START, max_steps=MAX_STEPS):
        self.start = start
        self.max_steps = max_steps
        self.reset()

    def reset(self, start=None):
        self.position = start if start is not None else self.start
        self.steps = 0
        return self.position

    def set_position(self, position):
        """
        Test mode: place the robot anywhere without training.
        """
        self.position = position
        self.steps = 0

    def step(self, action):
        next_position, invalid = move(self.position, action)
        self.steps += 1

        # The three reward cases are mutually exclusive.
        if next_position == GOAL:
            reward, terminated = 10.0, True
        elif invalid:
            reward, terminated = -1.0, False
        else:
            reward, terminated = -0.1, False

        truncated = (not terminated) and self.steps >= self.max_steps
        self.position = next_position

        return next_position, reward, terminated, truncated, {"invalid": invalid}


class TwoRobotWarehouse:
    """
    Two robots moved simultaneously by one controller.

    The state is (pA, pB). The joint action j = 5*aA + aB. A robot
    at its own goal stays there whatever it is told to do.
    """

    def __init__(self, starts=(START, B_START), max_steps=MAX_STEPS):
        self.starts = starts
        self.goals = (GOAL, B_GOAL)
        self.max_steps = max_steps
        self.reset()

    def reset(self, starts=None):
        self.positions = tuple(starts if starts is not None else self.starts)
        self.steps = 0
        return self.positions

    def set_positions(self, positions):
        """
        Test mode: place both robots without training.
        """
        self.positions = tuple(positions)
        self.steps = 0

    def step(self, joint_action):
        actions = divmod(joint_action, N_ACTIONS)
        old = self.positions
        proposed = []
        invalid = 0

        # Both proposals are computed from the same old state.
        for robot in range(2):
            if old[robot] == self.goals[robot]:
                proposed.append(old[robot])
                continue

            new, bad = move(old[robot], actions[robot])
            proposed.append(new)
            invalid += int(bad)

        same_cell = proposed[0] == proposed[1]
        swap = proposed[0] == old[1] and proposed[1] == old[0]
        self.steps += 1

        if same_cell or swap:
            # Old positions are kept for reporting.
            info = {"collision": True, "invalid": invalid, "arrived": 0}
            return old, COLLISION_REWARD, True, False, info

        arrived = sum(1 for robot in range(2)
                      if old[robot] != self.goals[robot]
                      and proposed[robot] == self.goals[robot])
        both = all(proposed[robot] == self.goals[robot] for robot in range(2))

        reward = -0.1 - invalid + 5 * arrived + 5 * int(both)
        terminated = both
        truncated = (not terminated) and self.steps >= self.max_steps

        self.positions = tuple(proposed)
        info = {"collision": False, "invalid": invalid, "arrived": arrived}

        return self.positions, reward, terminated, truncated, info


# ------------------------------------------------------------
# Part I: tabular Q-learning
# ------------------------------------------------------------

def linear_epsilon(t, decay_length, start=EPS_START, end=EPS_END):
    """
    Epsilon decreases linearly from start to end over decay_length
    and then stays at end.
    """
    fraction = min(t / decay_length, 1.0)
    return start + fraction * (end - start)


def epsilon_greedy(q_values, epsilon, rng):
    """
    Training-time action choice.

    With probability epsilon choose uniformly from all five
    actions; otherwise choose a best action, breaking ties at
    random.
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))

    best = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best))


def greedy_action(q_values):
    """
    Evaluation-time action choice with a fixed tie-break: the
    lowest action code among the best.
    """
    return int(np.argmax(q_values))


def td_target(reward, max_next_q, terminated, gamma=GAMMA):
    """
    y = r on true termination, otherwise r + gamma * max Q(s', a').

    Truncation is not termination, so it still bootstraps.
    """
    if terminated:
        return reward
    return reward + gamma * max_next_q


def q_update(q_table, s, a, reward, s_next, terminated,
             alpha=ALPHA, gamma=GAMMA):
    """
    One Q-learning update of the entry for the action taken.

    Returns:
        (target, new_value)
    """
    target = td_target(reward, q_table[s_next].max(), terminated, gamma)
    q_table[s, a] += alpha * (target - q_table[s, a])
    return target, q_table[s, a]


def train_q_learning(episodes=Q_EPISODES, seed=SEED, explore=True):
    """
    Train a fresh Q-table from the default start.

    Parameters:
        episodes : number of training episodes
        seed     : seed for exploration and tie-breaking
        explore  : False sets epsilon to 0 throughout

    Returns:
        (q_table, log) where log holds per-episode return, success,
        epsilon, invalid moves and timeout.
    """
    rng = np.random.default_rng(seed)
    q_table = np.zeros((SIZE * SIZE, N_ACTIONS))
    env = Warehouse()
    log = {"return": [], "success": [], "epsilon": [],
           "invalid": [], "timeout": []}

    for episode in range(episodes):
        epsilon = linear_epsilon(episode, Q_DECAY_EPISODES) if explore else 0.0
        position = env.reset()
        total, invalid = 0.0, 0

        while True:
            s = state_index(position)
            a = epsilon_greedy(q_table[s], epsilon, rng)
            next_position, reward, terminated, truncated, info = env.step(a)

            # The next state from before any reset is used here.
            q_update(q_table, s, a, reward, state_index(next_position), terminated)

            total += reward
            invalid += int(info["invalid"])
            position = next_position

            if terminated or truncated:
                break

        log["return"].append(total)
        log["success"].append(int(terminated))
        log["epsilon"].append(epsilon)
        log["invalid"].append(invalid)
        log["timeout"].append(int(truncated))

    return q_table, log


def greedy_rollout(env, choose, start=None):
    """
    Run one rollout with a fixed policy and no learning.

    Parameters:
        env    : Warehouse or TwoRobotWarehouse
        choose : function state -> action
        start  : optional start state

    Returns:
        Dictionary with the trajectory, actions, outcome, steps,
        invalid moves and return.
    """
    state = env.reset(start)
    trajectory = [state]
    actions = []
    total, invalid = 0.0, 0
    collision = False

    while True:
        action = choose(state)
        state, reward, terminated, truncated, info = env.step(action)
        trajectory.append(state)
        actions.append(action)
        total += reward
        invalid += info["invalid"]
        collision = collision or info.get("collision", False)

        if terminated or truncated:
            break

    if collision:
        outcome = "collision"
    elif terminated:
        outcome = "success"
    else:
        outcome = "timeout"

    return {"trajectory": trajectory, "actions": actions, "outcome": outcome,
            "steps": len(actions), "invalid": invalid, "return": total}


def random_policy_eval(episodes=100, seed=SEED):
    """
    Uniformly random policy on the same task and cutoff.
    """
    rng = np.random.default_rng(seed)
    env = Warehouse()
    results = [greedy_rollout(env, lambda s: int(rng.integers(N_ACTIONS)))
               for _ in range(episodes)]

    successes = [r for r in results if r["outcome"] == "success"]

    return {
        "success_rate": len(successes) / episodes,
        "timeout_rate": sum(r["outcome"] == "timeout" for r in results) / episodes,
        "mean_steps": np.mean([r["steps"] for r in successes]) if successes else None,
        "mean_invalid": np.mean([r["invalid"] for r in results]),
    }


def bfs_shortest(start_state, successors, is_goal):
    """
    Reference shortest route length, used only for comparison.
    Nothing here is given to any learner.
    """
    frontier = deque([(start_state, 0)])
    seen = {start_state}

    while frontier:
        state, depth = frontier.popleft()
        if is_goal(state):
            return depth
        for nxt in successors(state):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, depth + 1))

    return None


def joint_successors(state):
    """
    Collision-free successors of a two-robot state.
    """
    env = TwoRobotWarehouse()
    result = []
    for j in range(N_ACTIONS * N_ACTIONS):
        env.set_positions(state)
        nxt, _, _, _, info = env.step(j)
        if not info["collision"]:
            result.append(nxt)
    return result


def moving_average(values, window=MA_WINDOW):
    """
    Trailing moving average; early entries average what exists.
    """
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty(len(values))
    for i in range(len(values)):
        lo = max(0, i + 1 - window)
        out[i] = (cumulative[i + 1] - cumulative[lo]) / (i + 1 - lo)
    return out


def grid_picture(trajectory):
    """
    ASCII picture of a single-robot trajectory: digits give the
    time step modulo 10.
    """
    cells = [["#" if (r, c) in OBSTACLES else "." for c in range(SIZE)]
             for r in range(SIZE)]

    for t, (r, c) in enumerate(trajectory):
        cells[r][c] = str(t % 10)

    if GOAL not in trajectory:
        cells[GOAL[0]][GOAL[1]] = "G"
    return "\n".join("  " + " ".join(row) for row in cells)


# ------------------------------------------------------------
# Part II and III: Deep Q-Network
# ------------------------------------------------------------

def one_hot(position):
    vector = torch.zeros(SIZE * SIZE)
    vector[state_index(position)] = 1.0
    return vector


def encode_single(state):
    return one_hot(state)


def encode_pair(state):
    """
    Two concatenated one-hot vectors, A first: 50 inputs.
    """
    return torch.cat([one_hot(state[0]), one_hot(state[1])])


def build_q_network(n_inputs, n_outputs):
    """
    inputs -> 64 ReLU -> 64 ReLU -> linear Q-values (no softmax).
    """
    return nn.Sequential(
        nn.Linear(n_inputs, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, n_outputs),
    )


class ReplayBuffer:
    """
    Fixed-capacity store of (s, a, r, s', terminated) transitions.
    """

    def __init__(self, capacity=REPLAY_CAPACITY):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, float(terminated)))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.memory)


def dqn_loss(online, target, batch, gamma=GAMMA):
    """
    Mean squared TD error for one batch.

    Q(s, a) is taken for the recorded action with gather(). The
    target y = r + gamma * (1 - d) * max Q_target(s', a') is built
    under no_grad, so no gradient reaches the target network.
    d is 1 only for true termination; truncation keeps d = 0.
    """
    states, actions, rewards, next_states, dones = batch

    q_taken = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        max_next = target(next_states).max(dim=1).values
        targets = rewards + gamma * (1.0 - dones) * max_next

    loss = nn.functional.mse_loss(q_taken, targets)
    return loss, q_taken, targets


def train_dqn(env, encode, n_inputs, n_outputs, total_steps, seed=SEED):
    """
    Train a DQN with replay and a target network.

    Parameters:
        env         : environment exposing reset() and step()
        encode      : state -> input tensor
        n_inputs    : input size of the network
        n_outputs   : number of (joint) actions
        total_steps : environment-step budget
        seed        : seed for every generator

    Returns:
        (online_network, log)
    """
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    online = build_q_network(n_inputs, n_outputs)
    target = build_q_network(n_inputs, n_outputs)
    target.load_state_dict(online.state_dict())
    optimiser = torch.optim.Adam(online.parameters(), lr=DQN_LR)
    buffer = ReplayBuffer()

    decay_steps = EPS_DECAY_FRACTION * total_steps
    log = {"return": [], "success": [], "collision": [], "end_step": [],
           "updates": 0, "syncs": 0}

    state = env.reset()
    episode_return = 0.0

    for step in range(total_steps):
        epsilon = linear_epsilon(step, decay_steps)

        with torch.no_grad():
            q_values = online(encode(state)).numpy()
        action = epsilon_greedy(q_values, epsilon, rng)

        next_state, reward, terminated, truncated, info = env.step(action)

        # Stored before any reset, so a truncated transition keeps
        # its real next state and still bootstraps.
        buffer.push(encode(state), action, reward, encode(next_state), terminated)
        episode_return += reward
        state = next_state

        if step >= LEARNING_STARTS and step % TRAIN_EVERY == 0:
            loss, _, _ = dqn_loss(online, target, buffer.sample(BATCH_SIZE))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            log["updates"] += 1

            if log["updates"] % TARGET_SYNC == 0:
                target.load_state_dict(online.state_dict())
                log["syncs"] += 1

        if terminated or truncated:
            log["return"].append(episode_return)
            log["success"].append(int(terminated and not info.get("collision", False)))
            log["collision"].append(int(info.get("collision", False)))
            log["end_step"].append(step + 1)
            state = env.reset()
            episode_return = 0.0

    return online, log


def network_policy(network, encode):
    """
    Greedy policy from a network, with the fixed lowest-index
    tie-break and no gradient tracking.
    """
    def choose(state):
        with torch.no_grad():
            return greedy_action(network(encode(state)).numpy())
    return choose


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def environment_tests():
    """
    The five single-robot cases from Task 4, plus truncation.
    """
    env = Warehouse()
    cases = [
        ((0, 0), 0, (0, 0), -1.0, False),
        ((1, 1), 3, (1, 1), -1.0, False),
        ((2, 1), 3, (2, 2), -0.1, False),
        ((2, 1), 4, (2, 1), -0.1, False),
        ((2, 3), 3, (2, 4), 10.0, True),
    ]

    print("Environment tests (single robot)")
    print("  position  action  next      reward  term   trunc  result")

    for position, action, expect_pos, expect_r, expect_term in cases:
        env.set_position(position)
        nxt, reward, terminated, truncated, _ = env.step(action)
        ok = (nxt == expect_pos and abs(reward - expect_r) < 1e-9
              and terminated == expect_term and not truncated)
        print(f"  {str(position):<9} {ACTION_NAMES[action]:<7} {str(nxt):<9}"
              f" {reward:>6.1f}  {terminated!s:<6} {truncated!s:<6} "
              f"{'PASS' if ok else 'FAIL'}")

    # Waiting at the start forever must truncate after 40 steps.
    env.reset()
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(4)
        steps += 1
        if terminated or truncated:
            break
    ok = truncated and not terminated and steps == MAX_STEPS
    print(f"  always-wait rollout: stopped after {steps} steps, "
          f"terminated={terminated}, truncated={truncated}  "
          f"{'PASS' if ok else 'FAIL'}")


def update_tests():
    """
    Synthetic Q-learning updates with hand-computable values.
    Q(s,a) = 0.4, max Q(s',a') = 2, alpha = 0.2, gamma = 0.95.
    """
    print("\nQ-learning update tests (synthetic, not an environment step)")
    print("  case          r    target   new Q(s,a)  expected")

    cases = [
        ("non-terminal", 1.0, False, 2.9, 0.9),
        ("terminal", 5.0, True, 5.0, 1.32),
        ("truncated", 1.0, False, 2.9, 0.9),
    ]

    for name, reward, terminated, expect_y, expect_q in cases:
        q_table = np.zeros((2, N_ACTIONS))
        q_table[0, 0] = 0.4
        q_table[1] = [2.0, 1.0, 0.0, -1.0, 0.5]

        # Truncation passes terminated=False, so it bootstraps.
        y, new = q_update(q_table, 0, 0, reward, 1, terminated)
        ok = abs(y - expect_y) < 1e-9 and abs(new - expect_q) < 1e-9
        print(f"  {name:<13} {reward:>3.0f}  {y:>7.4f}  {new:>10.4f}"
              f"  {expect_y}, {expect_q}  {'PASS' if ok else 'FAIL'}")


def dqn_loss_test():
    """
    Loss on a batch whose targets are known in advance.

    Linear stand-in networks with zero weights make every output
    equal to the bias, so Q(s,a) and max Q_target(s',a') are set
    exactly. The online biases differ per action, so a wrong
    gather() index would change the loss.
    """
    online = nn.Linear(SIZE * SIZE, N_ACTIONS)
    target = nn.Linear(SIZE * SIZE, N_ACTIONS)

    with torch.no_grad():
        online.weight.zero_()
        online.bias.copy_(torch.tensor([0.4, 0.1, -0.3, 0.7, 0.0]))
        target.weight.zero_()
        target.bias.copy_(torch.tensor([2.0, 1.0, 0.0, -1.0, 0.5]))

    states = torch.stack([one_hot((2, 1)), one_hot((2, 3)), one_hot((1, 1))])
    next_states = torch.stack([one_hot((2, 2)), one_hot((2, 4)), one_hot((0, 1))])
    batch = (states,
             torch.tensor([0, 3, 1]),
             torch.tensor([1.0, 5.0, 1.0]),
             next_states,
             torch.tensor([0.0, 1.0, 0.0]))   # non-terminal, terminal, truncated

    loss, q_taken, targets = dqn_loss(online, target, batch)
    loss.backward()

    expected_targets = [2.9, 5.0, 2.9]
    expected_q = [0.4, 0.7, 0.1]
    expected_loss = sum((q - y) ** 2 for q, y in zip(expected_q, expected_targets)) / 3

    print("\nDQN loss test (gamma = 0.95, max target-network next value 2)")
    print("  case          recorded a  Q(s,a)   target")
    for name, a, q, y in zip(("non-terminal", "terminal", "truncated"),
                             (0, 3, 1), q_taken.tolist(), targets.tolist()):
        print(f"  {name:<13} {ACTION_NAMES[a]:<11} {q:>6.4f}  {y:>7.4f}")

    print(f"  loss {loss.item():.4f}, expected {expected_loss:.4f}  "
          f"{'PASS' if abs(loss.item() - expected_loss) < 1e-5 else 'FAIL'}")
    print("  target network grads are None:",
          all(p.grad is None for p in target.parameters()))
    print("  online network received grads:",
          all(p.grad is not None for p in online.parameters()))


def two_robot_tests():
    """
    Collision and goal rules for simultaneous movement.
    """
    env = TwoRobotWarehouse()
    cases = [
        ((2, 1), (2, 3), 3, 2, "same-cell collision, -10, terminate"),
        ((2, 1), (2, 2), 3, 2, "swap collision, -10, terminate"),
        ((2, 1), (2, 2), 3, 3, "A follows B, no collision, -0.1"),
        ((2, 1), (2, 2), 0, 4, "A up to (1,1), B waits, -0.1"),
    ]

    print("\nTwo-robot environment tests")
    print("  old A, B          A act  B act  new A, B          reward  coll   term")

    for pos_a, pos_b, a_a, a_b, _ in cases:
        env.set_positions((pos_a, pos_b))
        new, reward, terminated, _, info = env.step(N_ACTIONS * a_a + a_b)
        print(f"  {str((pos_a, pos_b)):<17} {ACTION_NAMES[a_a]:<6} "
              f"{ACTION_NAMES[a_b]:<6} {str(new):<17} {reward:>6.1f}  "
              f"{info['collision']!s:<6} {terminated}")

    # B moves into a cell where A is waiting.
    env.set_positions(((2, 1), (2, 2)))
    _, reward, terminated, _, info = env.step(N_ACTIONS * 4 + 2)
    print(f"  B enters waiting A's cell: reward {reward}, "
          f"collision {info['collision']}, terminated {terminated}")

    # A already at its goal stays there and is not paid again.
    env.set_positions((GOAL, (0, 0)))
    rewards = []
    for _ in range(2):
        new, reward, terminated, _, info = env.step(N_ACTIONS * 2 + 4)
        rewards.append(reward)
    print(f"  A at goal told 'left' twice: A stays at {new[0]}, rewards "
          f"{rewards}, invalid {info['invalid']}, arrivals {info['arrived']}")

    # First arrivals: k counts each new arrival once, I is a bonus.
    env.set_positions(((2, 3), (2, 1)))
    _, reward, terminated, _, info = env.step(N_ACTIONS * 3 + 2)
    print(f"  both arrive together: reward {reward:.1f} "
          f"(= -0.1 + 5*2 + 5), terminated {terminated}")

    env.set_positions(((2, 3), (0, 0)))
    _, reward, terminated, _, info = env.step(N_ACTIONS * 3 + 0)
    print(f"  A arrives while B bumps the wall: reward {reward:.1f} "
          f"(= -0.1 - 1 + 5), terminated {terminated}")


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------

def print_training_summary(label, log, blocks=5):
    returns = np.asarray(log["return"])
    success = np.asarray(log["success"])
    n = len(returns)

    print(f"\n{label}: {n} training episodes")
    print(f"  {'episodes':<14}{'mean return':>12}{'success':>9}")
    for part in np.array_split(np.arange(n), blocks):
        lo, hi = part[0], part[-1] + 1
        print(f"  {f'{lo + 1}-{hi}':<14}{returns[lo:hi].mean():>12.3f}"
              f"{success[lo:hi].mean():>9.2f}")


def print_single_rollout(label, result):
    print(f"\n{label}")
    print(f"  outcome {result['outcome']}, steps {result['steps']}, "
          f"invalid moves {result['invalid']}, return {result['return']:.2f}")
    print("  positions:", " -> ".join(str(p) for p in result["trajectory"]))
    print("  actions  :", " ".join(ACTION_NAMES[a] for a in result["actions"]))


def print_joint_rollout(label, result):
    print(f"\n{label}")
    print("  t   A action  B action  A        B")
    print(f"  0   {'':<9} {'':<9} {str(result['trajectory'][0][0]):<8} "
          f"{result['trajectory'][0][1]}")
    for t, (j, (pa, pb)) in enumerate(zip(result["actions"],
                                          result["trajectory"][1:]), start=1):
        a_a, a_b = divmod(j, N_ACTIONS)
        print(f"  {t:<3} {ACTION_NAMES[a_a]:<9} {ACTION_NAMES[a_b]:<9} "
              f"{str(pa):<8} {pb}")
    print(f"  outcome {result['outcome']}, steps {result['steps']}, "
          f"invalid moves {result['invalid']}, return {result['return']:.2f}")


def draw_grid(ax, title):
    for (r, c) in OBSTACLES:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="0.35"))
    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(SIZE - 0.5, -0.5)
    ax.set_xticks(range(SIZE))
    ax.set_yticks(range(SIZE))
    ax.grid(False)
    for k in range(SIZE + 1):
        ax.axhline(k - 0.5, color="0.8", lw=0.8)
        ax.axvline(k - 0.5, color="0.8", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def plot_training(q_log, dqn_log, two_log, path):
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    rows = [("Q-table, one robot", q_log),
            ("DQN, one robot", dqn_log),
            ("DQN, two robots", two_log)]

    for i, (name, log) in enumerate(rows):
        x = np.arange(1, len(log["return"]) + 1)
        axes[i, 0].plot(x, log["return"], color="0.8", lw=0.6, label="episode")
        axes[i, 0].plot(x, moving_average(log["return"]), color="C0",
                        label=f"{MA_WINDOW}-episode moving average")
        axes[i, 0].set_title(f"{name}: training return", fontsize=10)
        axes[i, 0].set_xlabel("episode")
        axes[i, 0].legend(fontsize=8)

        axes[i, 1].plot(x, moving_average(log["success"]), color="C2",
                        label="success")
        if sum(log.get("collision", [])) > 0:
            axes[i, 1].plot(x, moving_average(log["collision"]), color="C3",
                            label="collision")
        if "epsilon" in log:
            axes[i, 1].plot(x, log["epsilon"], color="C1", ls="--", label="epsilon")
        axes[i, 1].set_ylim(-0.05, 1.05)
        axes[i, 1].set_title(f"{name}: {MA_WINDOW}-episode moving average",
                             fontsize=10)
        axes[i, 1].set_xlabel("episode")
        axes[i, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_trajectories(single_runs, joint_runs, path):
    panels = len(single_runs) + len(joint_runs)
    fig, axes = plt.subplots(1, panels, figsize=(3.6 * panels, 3.9))

    for ax, (title, result) in zip(axes, single_runs):
        draw_grid(ax, title)
        rows = [p[0] for p in result["trajectory"]]
        cols = [p[1] for p in result["trajectory"]]
        ax.plot(cols, rows, "-o", color="C0", ms=4)
        for t, (r, c) in enumerate(result["trajectory"]):
            ax.annotate(str(t), (c, r), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)

    for ax, (title, result) in zip(axes[len(single_runs):], joint_runs):
        draw_grid(ax, title)
        for robot, colour, shift in ((0, "C0", -0.12), (1, "C3", 0.12)):
            rows = [p[robot][0] + shift for p in result["trajectory"]]
            cols = [p[robot][1] + shift for p in result["trajectory"]]
            ax.plot(cols, rows, "-o", color=colour, ms=4,
                    label="A" if robot == 0 else "B")
            for t, (r, c) in enumerate(zip(rows, cols)):
                ax.annotate(str(t), (c, r), fontsize=7, color=colour,
                            xytext=(3, 3 if robot == 0 else -9),
                            textcoords="offset points")
        ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    """
    Run every test and experiment required by the laboratory.
    """
    print("numpy", np.__version__, "| torch", torch.__version__,
          "| matplotlib", matplotlib.__version__, "| seed", SEED)

    # --------------------------------------------------------
    # Task 1: reference shortest routes (comparison only)
    # --------------------------------------------------------
    def single_successors(p):
        return [move(p, a)[0] for a in range(N_ACTIONS)]

    print("\nReference shortest routes by BFS (never given to a learner)")
    print("  one robot S -> G      :",
          bfs_shortest(START, single_successors, lambda p: p == GOAL), "steps")
    print("  two robots, joint     :",
          bfs_shortest((START, B_START), joint_successors,
                       lambda s: s == (GOAL, B_GOAL)), "joint steps")
    print("  one robot from (0,0)  :",
          bfs_shortest((0, 0), single_successors, lambda p: p == GOAL), "steps")
    print("  two robots from (0,0),(4,4):",
          bfs_shortest(((0, 0), (4, 4)), joint_successors,
                       lambda s: s == (GOAL, B_GOAL)), "joint steps")

    # --------------------------------------------------------
    # Task 4: test the environment and the update separately
    # --------------------------------------------------------
    print()
    environment_tests()
    update_tests()

    # --------------------------------------------------------
    # Task 5: train, evaluate, compare with random
    # --------------------------------------------------------
    q_table, q_log = train_q_learning()
    print_training_summary("Q-learning (alpha 0.2, gamma 0.95, seed 7)", q_log)
    print(f"  epsilon at episodes 1, 600, 1200, 1500: "
          f"{q_log['epsilon'][0]:.3f}, {q_log['epsilon'][599]:.3f}, "
          f"{q_log['epsilon'][1199]:.3f}, {q_log['epsilon'][1499]:.3f}")
    print(f"  first training success at episode "
          f"{q_log['success'].index(1) + 1}; training timeouts "
          f"{sum(q_log['timeout'])}; training invalid moves "
          f"{sum(q_log['invalid'])}")
    ma = moving_average(q_log["success"])
    print(f"  {MA_WINDOW}-episode success average at episodes 100, 500, 1000, "
          f"1500: {ma[99]:.2f}, {ma[499]:.2f}, {ma[999]:.2f}, {ma[1499]:.2f}")

    q_policy = lambda p: greedy_action(q_table[state_index(p)])
    q_result = greedy_rollout(Warehouse(), q_policy)
    print_single_rollout("Greedy Q-policy from (2,0), epsilon 0, no updates",
                         q_result)
    print(grid_picture(q_result["trajectory"]))

    print("\n  Greedy values along the route, max_a Q(s,a):")
    print("  ", [round(float(q_table[state_index(p)].max()), 3)
                 for p in q_result["trajectory"][:-1]])

    rand = random_policy_eval()
    print("\nRandom policy, 100 episodes, same map and cutoff")
    print(f"  success fraction {rand['success_rate']:.2f}, timeout fraction "
          f"{rand['timeout_rate']:.2f}, mean steps when successful "
          f"{rand['mean_steps'] if rand['mean_steps'] is None else round(rand['mean_steps'], 2)}, "
          f"mean invalid moves per episode {rand['mean_invalid']:.2f}")

    # Optional: no exploration, two seeds.
    print("\nOptional investigation: epsilon = 0 throughout")
    print(f"  {'seed':<6}{'explore':<9}{'train success':>14}"
          f"{'first success':>15}{'greedy outcome':>16}{'steps':>7}")
    for seed in (7, 8):
        for explore in (True, False):
            table, log = train_q_learning(seed=seed, explore=explore)
            res = greedy_rollout(Warehouse(),
                                 lambda p, t=table: greedy_action(t[state_index(p)]))
            first = log["success"].index(1) + 1 if 1 in log["success"] else "-"
            print(f"  {seed:<6}{str(explore):<9}{np.mean(log['success']):>14.3f}"
                  f"{first!s:>15}{res['outcome']:>16}{res['steps']:>7}")

    # --------------------------------------------------------
    # Task 6: DQN for one robot
    # --------------------------------------------------------
    dqn_loss_test()

    dqn, dqn_log = train_dqn(Warehouse(), encode_single, 25, N_ACTIONS, DQN_STEPS)
    print_training_summary("DQN one robot (12,000 steps, seed 7)", dqn_log)
    print(f"  optimizer updates {dqn_log['updates']}, target syncs "
          f"{dqn_log['syncs']}")

    dqn_policy = network_policy(dqn, encode_single)
    dqn_result = greedy_rollout(Warehouse(), dqn_policy)
    print_single_rollout("Greedy DQN policy from (2,0)", dqn_result)
    print("  same route as Q-table:",
          dqn_result["trajectory"] == q_result["trajectory"])

    # --------------------------------------------------------
    # Task 7: two robots
    # --------------------------------------------------------
    two_robot_tests()

    two_runs = []
    for seed, steps in ((7, TWO_ROBOT_STEPS), (8, TWO_ROBOT_STEPS)):
        net, log = train_dqn(TwoRobotWarehouse(), encode_pair, 50,
                             N_ACTIONS * N_ACTIONS, steps, seed=seed)
        print_training_summary(f"DQN two robots ({steps:,} steps, seed {seed})", log)
        print(f"  optimizer updates {log['updates']}, target syncs {log['syncs']}, "
              f"training collisions {sum(log['collision'])}")
        result = greedy_rollout(TwoRobotWarehouse(),
                                network_policy(net, encode_pair))
        print_joint_rollout(f"Greedy joint rollout, seed {seed}", result)
        two_runs.append((seed, net, log, result))

    # --------------------------------------------------------
    # Task 8: additional starts, no retraining
    # --------------------------------------------------------
    print("\nAdditional starts, greedy, no retraining")
    extra_q = greedy_rollout(Warehouse(), q_policy, start=(0, 0))
    print_single_rollout("Q-table from (0,0)", extra_q)
    extra_dqn = greedy_rollout(Warehouse(), dqn_policy, start=(0, 0))
    print_single_rollout("DQN one robot from (0,0)", extra_dqn)

    extra_joint = []
    for seed, net, _, _ in two_runs:
        res = greedy_rollout(TwoRobotWarehouse(), network_policy(net, encode_pair),
                             start=((0, 0), (4, 4)))
        print_joint_rollout(f"DQN two robots (seed {seed}) from A (0,0), B (4,4)",
                            res)
        extra_joint.append((seed, res))

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    main_seed, _, main_log, main_result = two_runs[0]
    plot_training(q_log, dqn_log, main_log, "rl_lab_training_curves.png")
    plot_trajectories(
        [("Q-table, start (2,0)", q_result),
         ("DQN, start (2,0)", dqn_result),
         ("Q-table, start (0,0)", extra_q)],
        [(f"Two robots, seed {seed}", res) for seed, _, _, res in two_runs]
        + [(f"Two robots, seed {extra_joint[0][0]}, (0,0)/(4,4)",
            extra_joint[0][1])],
        "rl_lab_trajectories.png")
    print("\nSaved rl_lab_training_curves.png and rl_lab_trajectories.png")


# ------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
