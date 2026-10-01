import torch
import torch.nn as nn

# ------------------------------------------------------------
# Neural Models: Learning, Depth, Activations, Output Layers
# ------------------------------------------------------------
# A device has two redundant binary sensors x1 and x2 and must
# raise a warning exactly when the sensors disagree. That rule
# is XOR, which is not linearly separable, so the experiments
# below compare:
#
#   - a single affine map plus sigmoid   (cannot represent XOR)
#   - a 2-2-1 network with a nonlinearity (can represent XOR)
#   - zero initialisation                (breaks symmetry)
#   - sigmoid / tanh / ReLU hidden units
#   - a three-class softmax output
# ------------------------------------------------------------


SEED = 0
STEPS = 4000
LEARNING_RATE = 0.1
GRAD_STEP = 10          # training step at which gradients are recorded

# The four sensor readings. These are the whole input space.
INPUTS = torch.tensor([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]])

# XOR targets: warn when the two sensors disagree.
TARGETS = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Three-class relabelling of the same inputs:
#   0 -> both inactive, 1 -> disagreement, 2 -> both active
TARGETS_3 = torch.tensor([0, 1, 1, 2])


def activation_module(name):
    """
    Build a hidden activation by name.

    Parameters:
        name : 'sigmoid', 'tanh' or 'relu'

    Returns:
        The corresponding torch module.
    """
    modules = {
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
    }

    return modules[name]()


def build_network(hidden_activation, hidden_units=2, seed=SEED):
    """
    Build the 2 -> hidden -> 1 network.

    The output is a raw logit. The sigmoid is applied inside
    BCEWithLogitsLoss, which is the numerically stable way of
    writing sigmoid plus binary cross-entropy.

    Parameters:
        hidden_activation : name of the hidden nonlinearity
        hidden_units      : number of hidden units
        seed              : seed for the random initialisation

    Returns:
        A torch Sequential model.
    """
    torch.manual_seed(seed)

    return nn.Sequential(
        nn.Linear(2, hidden_units),
        activation_module(hidden_activation),
        nn.Linear(hidden_units, 1),
    )


def build_linear(seed=SEED):
    """
    Build the control model: one affine map, no hidden layer.

    Any stack of affine layers collapses to a single affine map,
    so this is the honest test of what depth alone can do.

    Parameters:
        seed : seed for the random initialisation

    Returns:
        A torch Sequential model.
    """
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(2, 1))


def build_three_class(hidden_activation="tanh", seed=SEED):
    """
    Build the 2 -> 2 -> 3 network for the multiclass extension.

    Only the output layer differs from the binary network: three
    logits instead of one, with the softmax inside the loss.

    Parameters:
        hidden_activation : name of the hidden nonlinearity
        seed              : seed for the random initialisation

    Returns:
        A torch Sequential model.
    """
    torch.manual_seed(seed)

    return nn.Sequential(
        nn.Linear(2, 2),
        activation_module(hidden_activation),
        nn.Linear(2, 3),
    )


def zero_parameters(model):
    """
    Set every weight and bias to exactly zero.

    Parameters:
        model : the network to modify

    Returns:
        The same model, modified in place.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    return model


def train_binary(model, steps=STEPS, learning_rate=LEARNING_RATE):
    """
    Train a binary model on all four examples at once.

    Parameters:
        model         : network producing one logit per example
        steps         : number of full-batch gradient steps
        learning_rate : Adam learning rate

    Returns:
        Dictionary with the initial and final loss, the four
        probabilities and labels, whether all four are correct,
        the first-layer gradient and its early norm.
    """
    loss_function = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    first_layer = model[0]

    initial_loss = None
    early_grad_norm = None

    for step in range(steps):

        optimiser.zero_grad()
        logits = model(INPUTS)
        loss = loss_function(logits, TARGETS)
        loss.backward()

        if step == 0:
            initial_loss = loss.item()

        # Recorded early, because a converged gradient is small
        # for reasons unrelated to the activation function.
        if step == GRAD_STEP:
            early_grad_norm = first_layer.weight.grad.norm().item()

        optimiser.step()

    # One last forward and backward pass so that the reported
    # gradient belongs to the reported parameters.
    optimiser.zero_grad()
    logits = model(INPUTS)
    final_loss = loss_function(logits, TARGETS)
    final_loss.backward()

    probabilities = torch.sigmoid(logits).detach().squeeze(1)
    labels = (probabilities >= 0.5).float()

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": final_loss.item(),
        "probabilities": probabilities,
        "labels": labels,
        "all_correct": bool((labels == TARGETS.squeeze(1)).all().item()),
        "weight_grad": first_layer.weight.grad.clone(),
        "early_grad_norm": early_grad_norm,
    }


def finite_difference_check(model, epsilon=1e-4):
    """
    Compare the autograd gradient against central differences.

    This is what turns a printed gradient tensor into evidence:
    it checks the number really is the derivative it claims to be.

    Parameters:
        model   : trained network whose first layer is checked
        epsilon : step size for the numerical derivative

    Returns:
        The largest absolute disagreement found.
    """
    loss_function = nn.BCEWithLogitsLoss()
    weight = model[0].weight
    autograd_grad = weight.grad
    worst = 0.0

    def current_loss():
        with torch.no_grad():
            return loss_function(model(INPUTS), TARGETS).item()

    for row in range(weight.shape[0]):
        for col in range(weight.shape[1]):

            original = weight.data[row][col].item()

            weight.data[row][col] = original + epsilon
            loss_plus = current_loss()

            weight.data[row][col] = original - epsilon
            loss_minus = current_loss()

            weight.data[row][col] = original

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            worst = max(worst, abs(numerical - autograd_grad[row][col].item()))

    return worst


def print_binary_run(label, result):
    """
    Print the loss and the four predictions for one run.

    Parameters:
        label  : name of the run
        result : dictionary returned by train_binary()
    """
    print(f"\n{label}")
    print(f"  initial loss : {result['initial_loss']:.6f}")
    print(f"  final loss   : {result['final_loss']:.6f}")
    print("  x1 x2  target  p(warn)  label")

    for i in range(4):
        print(f"   {int(INPUTS[i][0])}  {int(INPUTS[i][1])}"
              f"       {int(TARGETS[i][0])}"
              f"   {result['probabilities'][i]:.4f}"
              f"      {int(result['labels'][i])}")

    print("  all four correct:", result["all_correct"])


def symmetry_experiment():
    """
    Train the same network with every parameter set to zero.

    Identical units compute the same function and so receive the
    same gradient, which a deterministic update cannot separate.

    Returns:
        Dictionary with the final loss and whether the two hidden
        rows are still identical.
    """
    model = zero_parameters(build_network("tanh"))
    loss_function = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    weight = model[0].weight

    print("\nSymmetry experiment, all parameters zeroed")
    print("  step   row 0        row 1        identical  loss")

    for step in range(6):
        optimiser.zero_grad()
        loss = loss_function(model(INPUTS), TARGETS)
        loss.backward()

        print(f"  {step:<6} {weight[0].tolist()}  {weight[1].tolist()}"
              f"  {torch.allclose(weight[0], weight[1])!s:<9}"
              f"  {loss.item():.6f}")

        optimiser.step()

    # Train out to the same budget as the random-initialised run.
    for _ in range(STEPS):
        optimiser.zero_grad()
        loss_function(model(INPUTS), TARGETS).backward()
        optimiser.step()

    with torch.no_grad():
        logits = model(INPUTS)
        final_loss = loss_function(logits, TARGETS).item()
        probabilities = torch.sigmoid(logits).squeeze(1)

    labels = (probabilities >= 0.5).float()
    correct = bool((labels == TARGETS.squeeze(1)).all().item())

    print(f"  after {STEPS} more steps: loss {final_loss:.6f}, "
          f"all correct {correct}")
    print("  probabilities:", [round(p, 4) for p in probabilities.tolist()])
    print("  rows identical:", torch.allclose(weight[0], weight[1]))

    return {"final_loss": final_loss, "all_correct": correct}


def activation_experiment():
    """
    Train the network three times, changing only the activation.

    Returns:
        List of (name, final_loss, all_correct, early_grad_norm).
    """
    rows = []

    print("\nActivation experiment (seed 0)")
    print(f"  {'Activation':<12}{'Final loss':>12}{'4/4':>6}"
          f"{'Early grad norm':>18}")

    for name in ("sigmoid", "tanh", "relu"):
        result = train_binary(build_network(name))
        rows.append((name, result["final_loss"], result["all_correct"],
                     result["early_grad_norm"]))

        print(f"  {name:<12}{result['final_loss']:>12.6f}"
              f"{('yes' if result['all_correct'] else 'no'):>6}"
              f"{result['early_grad_norm']:>18.6f}")

    return rows


def seed_sweep(seeds=5):
    """
    Repeat each activation over several seeds.

    One run cannot distinguish a property of the activation from
    a lucky or unlucky initialisation.

    Parameters:
        seeds : how many seeds to try per activation
    """
    print(f"\nRepeated runs over {seeds} seeds")
    print(f"  {'Activation':<12}{'Solved':>8}  Final losses")

    for name in ("sigmoid", "tanh", "relu"):
        solved = 0
        losses = []

        for seed in range(seeds):
            result = train_binary(build_network(name, seed=seed))
            solved += int(result["all_correct"])
            losses.append(round(result["final_loss"], 4))

        print(f"  {name:<12}{f'{solved}/{seeds}':>8}  {losses}")


def saturation_snapshot():
    """
    Show pre-activations and activations at initialisation.

    A saturated sigmoid sits near 0 or 1 but never exactly at
    either, so its derivative is small but nonzero. A dead ReLU
    sits exactly at 0, so its derivative is exactly 0.
    """
    print("\nPre-activations and activations at initialisation")

    for name in ("sigmoid", "relu"):
        model = build_network(name)
        pre_activation = model[0](INPUTS)
        activation = model[1](pre_activation)

        print(f"  {name}:")
        print("    pre :", [[round(v, 4) for v in row]
                            for row in pre_activation.detach().tolist()])
        print("    post:", [[round(v, 4) for v in row]
                            for row in activation.detach().tolist()])


def three_class_experiment():
    """
    Replace the single output with three logits and retrain.

    Checks the shapes, that the softmax sums to one, that the
    logit gradient equals p - y, and that softmax is unchanged by
    adding a constant to every logit.
    """
    model = build_three_class()
    loss_function = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    initial_loss = None

    for step in range(STEPS):
        optimiser.zero_grad()
        logits = model(INPUTS)
        loss = loss_function(logits, TARGETS_3)

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimiser.step()

    with torch.no_grad():
        logits = model(INPUTS)
        final_loss = loss_function(logits, TARGETS_3).item()
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    print("\nThree-class extension")
    print("  output weight shape:", tuple(model[2].weight.shape))
    print("  output bias shape  :", tuple(model[2].bias.shape))
    print("  logits shape       :", tuple(logits.shape))
    print(f"  initial loss       : {initial_loss:.6f}")
    print(f"  final loss         : {final_loss:.6f}")

    print("  x1 x2  true  p(0)      p(1)      p(2)      predicted")
    for i in range(4):
        row = probabilities[i]
        print(f"   {int(INPUTS[i][0])}  {int(INPUTS[i][1])}"
              f"     {int(TARGETS_3[i])}"
              f"   {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}"
              f"         {int(predictions[i])}")

    print("  all four correct   :",
          bool((predictions == TARGETS_3).all().item()))

    # Softmax normalises by construction, so this must be 1.
    one_example = probabilities[1]
    print(f"  sum for (0,1)      : {one_example.sum().item():.10f}")

    # Check the p - y claim directly on a single example.
    logit = model(INPUTS[1:2]).detach().requires_grad_(True)
    nn.CrossEntropyLoss()(logit, TARGETS_3[1:2]).backward()
    one_hot = torch.nn.functional.one_hot(TARGETS_3[1:2], num_classes=3)
    p_minus_y = torch.softmax(logit, dim=1).detach() - one_hot.float()

    print("  autograd dL/dz     :", [f"{v:.6e}" for v in logit.grad[0].tolist()])
    print("  p - y              :", [f"{v:.6e}" for v in p_minus_y[0].tolist()])
    print("  match              :",
          bool(torch.allclose(logit.grad, p_minus_y, atol=1e-6)))

    # Adding a constant to every logit cancels in the softmax.
    shifted = torch.softmax(logits[1] + 100.0, dim=0)
    difference = (torch.softmax(logits[1], dim=0) - shifted).abs().max()
    print(f"  shift by +100, max probability change: {difference.item():.3e}")

    # Without the max-subtraction trick, large logits overflow.
    large = torch.tensor([1000.0, 999.0, 998.0])
    print("  naive exp of large logits :", torch.exp(large).tolist())
    print("  stable softmax            :",
          [f"{v:.6f}" for v in torch.softmax(large, dim=0).tolist()])


def main():
    """
    Run every experiment required by the laboratory.
    """
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # Part A: can a linear model represent XOR?
    # --------------------------------------------------------
    linear = train_binary(build_linear())
    print_binary_run("Control: single affine map plus sigmoid", linear)
    print("  ln(2) = 0.693147 is the loss of a constant 0.5 predictor")

    network = train_binary(build_network("tanh"))
    print_binary_run("2-2-1 network, tanh hidden, sigmoid output", network)

    # --------------------------------------------------------
    # Part B: what backpropagation actually produced
    # --------------------------------------------------------
    print("\nFirst-layer weight gradient after backward()")
    print("  dL/dW1:", [[f"{v:.4e}" for v in row]
                        for row in network["weight_grad"].tolist()])
    print(f"  Frobenius norm: {network['weight_grad'].norm().item():.8f}")
    print(f"  finite-difference check, worst error: "
          f"{finite_difference_check(network['model']):.2e}")

    # --------------------------------------------------------
    # Part C and D: symmetry and activations
    # --------------------------------------------------------
    symmetry_experiment()
    activation_experiment()
    seed_sweep()
    saturation_snapshot()

    # --------------------------------------------------------
    # Task 5: three-class output
    # --------------------------------------------------------
    three_class_experiment()


# ------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
