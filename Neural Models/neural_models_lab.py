"""
Laboratory - Neural Models: Learning, Depth, Activations, and Output Layers
===========================================================================

Motivating scenario: a device has two redundant binary sensors x1, x2 and must
raise a "sensor-disagreement" warning exactly when the sensors differ (XOR).

This single file covers the experiments requested in the lab sheet:

  Task 3 / Task 4 Part A : 2-2-1 network, sigmoid output via BCEWithLogitsLoss,
                           random init, full-batch training, final loss +
                           4 probabilities + thresholded labels.
  Task 4 Part B          : inspect .grad of the first-layer weights after
                           backward(), i.e. dL/dW1.
  Task 4 Part C          : symmetry experiment with all weights (and biases)
                           initialised to zero.
  Task 4 Part D          : activation experiment (sigmoid / tanh / ReLU) with
                           final loss, 4/4 correctness, and early ||dL/dW1||_2.
  Task 5                 : three-class extension (3 logits + cross-entropy),
                           softmax probabilities, sum-to-one check, and the
                           logit-shift invariance diagnostic.

Also included: a linear (affine-only) baseline, which is the control that makes
the Task 1 prediction testable, and a finite-difference gradient check that
verifies autograd numerically.

Run:  python neural_models_lab.py
"""

import math

import torch
import torch.nn as nn

# Reproducibility for the whole script.
SEED = 0

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Input space X = {0,1}^2, output space Y = {0,1} for the binary task.
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

# XOR targets: warn exactly when the two sensors disagree.
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Task 5 relabelling: 0 = both inactive, 1 = disagree, 2 = both active.
Y3 = torch.tensor([0, 1, 1, 2])


def line(title):
    """Print a section banner so the transcript is easy to read."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def activation(name):
    """Return the hidden nonlinearity module for a given name."""
    return {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "relu": nn.ReLU()}[name]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_mlp(hidden_act="tanh", hidden=2, seed=SEED):
    """2 inputs -> `hidden` nonlinear hidden units -> 1 logit.

    The output is a raw logit; the sigmoid lives inside BCEWithLogitsLoss,
    which is the numerically stable way to write sigmoid + binary
    cross-entropy in PyTorch.
    """
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, hidden),      # W1 (hidden x 2), b1
        activation(hidden_act),    # the scientifically necessary nonlinearity
        nn.Linear(hidden, 1),      # W2 (1 x hidden), b2 -> logit
    )


def make_linear_baseline(seed=SEED):
    """Control model: a single affine map followed by a sigmoid output.

    Any stack of affine layers without hidden nonlinearities collapses to one
    affine map, so this is the honest test of the Task 1 prediction.
    """
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 1))


def zero_init(model):
    """Set every weight and bias to exactly zero (Task 4 Part C)."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


# ---------------------------------------------------------------------------
# Training loop (binary task)
# ---------------------------------------------------------------------------

def train_binary(model, steps=4000, lr=0.1, record_grad_at=10, verbose=False):
    """Full-batch training of a binary model on the four XOR examples.

    Returns a dict of the evidence the lab asks us to collect: initial loss,
    final loss, the gradient norm of the first-layer weights at an early step,
    and the first-layer gradient tensor from the last backward pass.
    """
    loss_fn = nn.BCEWithLogitsLoss()               # sigmoid + BCE, stable form
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    first_layer = model[0]
    initial_loss = None
    early_grad_norm = None
    history = []

    for step in range(steps):
        opt.zero_grad()
        logits = model(X)                          # forward pass
        loss = loss_fn(logits, Y)                  # scalar loss
        loss.backward()                            # reverse-mode AD

        if step == 0:
            initial_loss = loss.item()
        if step == record_grad_at:
            early_grad_norm = first_layer.weight.grad.norm().item()
        if step % max(1, steps // 5) == 0:
            history.append((step, loss.item()))

        opt.step()                                 # parameter update

    # One final forward/backward at the learned parameters so that .grad holds
    # a gradient that corresponds to the reported loss and predictions.
    opt.zero_grad()
    logits = model(X)
    final_loss = loss_fn(logits, Y)
    final_loss.backward()

    probs = torch.sigmoid(logits).detach()
    preds = (probs >= 0.5).float()
    correct = int((preds == Y).all().item())

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": final_loss.item(),
        "probs": probs.squeeze(1),
        "preds": preds.squeeze(1),
        "all_correct": bool(correct),
        "early_grad_norm": early_grad_norm,
        "W1_grad": first_layer.weight.grad.clone(),
        "history": history,
    }


def report_binary(tag, res):
    """Print the Part A evidence for one binary run."""
    print(f"[{tag}]")
    print(f"  initial loss : {res['initial_loss']:.6f}")
    print(f"  final loss   : {res['final_loss']:.6f}")
    print("  x1 x2 | target |  p(warn)  | predicted")
    for i in range(4):
        print(f"   {int(X[i,0])}  {int(X[i,1])}  |   {int(Y[i,0])}    "
              f"|  {res['probs'][i]:.4f}   |     {int(res['preds'][i])}")
    print(f"  all four correct: {res['all_correct']}")


# ---------------------------------------------------------------------------
# Task 1 / Task 2 written specification (printed so the run is self-contained)
# ---------------------------------------------------------------------------

def task1_specification():
    line("TASK 1 - Problem specification and linear-model prediction")
    print("Input space  X = {0,1}^2  (x1 = sensor A, x2 = sensor B)")
    print("Output space Y = {0,1}    (1 = raise sensor-disagreement warning)")
    print("Labelled examples:")
    for i in range(4):
        print(f"  ({int(X[i,0])}, {int(X[i,1])}) -> {int(Y[i,0])}")
    print()
    print("Sketch of the four points in the (x1, x2) plane:")
    print("  x2")
    print("   1 |  1(0,1)        0(1,1)")
    print("     |")
    print("   0 |  0(0,0)        1(1,0)")
    print("     +-------------------------- x1")
    print("        0               1")
    print()
    print("Why one straight boundary cannot work: the positive class {(0,1),")
    print("(1,0)} and the negative class {(0,0),(1,1)} each sit on a diagonal,")
    print("and the two diagonals cross. Any line w1*x1 + w2*x2 + b = 0 that puts")
    print("both positives on one side necessarily puts both negatives on that")
    print("same side, because (0,0)+(1,1) = (0,1)+(1,0) means the two class")
    print("means coincide at (0.5, 0.5). The classes are not linearly separable.")
    print()
    print("Prediction for a single affine map + sigmoid: it cannot fit XOR. The")
    print("loss should stall near ln 2 = 0.6931 and every probability should sit")
    print("near 0.5, giving at best 2/4 correct.")


def task2_specification():
    line("TASK 2 - Model design and validation criteria")
    print("Architecture     : 2 inputs -> 2 hidden units -> 1 output")
    print("Hidden activation: sigmoid / tanh / ReLU (nonlinear, compared later)")
    print("Output           : one logit, sigmoid applied through the loss")
    print("Loss             : binary cross-entropy (BCEWithLogitsLoss)")
    print("Optimiser        : Adam, full-batch, lr = 0.1")
    print()
    print("1. The hidden nonlinearity is necessary because composing affine")
    print("   maps gives another affine map: W2(W1 x + b1) + b2 = Wx + b. Depth")
    print("   alone adds no expressive power, so a nonlinear hidden layer is")
    print("   what lets the network bend the input space until the two XOR")
    print("   classes become linearly separable in hidden coordinates.")
    print("2. Sigmoid + BCE is a sensible pairing because the target is one")
    print("   yes/no answer: sigmoid maps a logit to a Bernoulli probability,")
    print("   BCE is the negative log-likelihood of that Bernoulli, and the")
    print("   pairing makes the logit gradient reduce to (p - y), which is")
    print("   well-scaled and does not vanish from the loss side.")
    print()
    print("Validation criteria (evidence that counts as learning):")
    print("  C1 final loss clearly below the ln 2 = 0.6931 chance level;")
    print("  C2 all four thresholded labels match the XOR truth table;")
    print("  C3 first-layer gradients are nonzero during training and shrink")
    print("     only as the loss converges;")
    print("  C4 the result reproduces across several random seeds;")
    print("  C5 autograd agrees with a finite-difference gradient estimate.")


# ---------------------------------------------------------------------------
# Task 4
# ---------------------------------------------------------------------------

def task4_part_a():
    line("TASK 4 PART A - Basic learning check (and the linear control)")

    lin = make_linear_baseline()
    lin_res = train_binary(lin, steps=4000, lr=0.1)
    report_binary("Control: single affine map + sigmoid", lin_res)
    print("  ln(2) = 0.693147 is the loss of a constant p = 0.5 predictor.")

    print()
    mlp = make_mlp("tanh")
    mlp_res = train_binary(mlp, steps=4000, lr=0.1)
    report_binary("2-2-1 network, tanh hidden, sigmoid output", mlp_res)
    return lin_res, mlp_res


def task4_part_b(res):
    line("TASK 4 PART B - Backpropagation check")
    W1_grad = res["W1_grad"]
    print("first_layer.weight.grad after backward()  (this is dL/dW1):")
    print(W1_grad)
    print(f"Frobenius norm: {W1_grad.norm().item():.8f}")
    print()
    print("Reading: entry [j, i] is the partial derivative of the scalar loss")
    print("with respect to the weight connecting input i to hidden unit j. It")
    print("is produced by the chain rule: the loss sensitivity is propagated")
    print("back to the hidden pre-activation as delta_j, and then dL/dW1[j,i]")
    print("= delta_j * x_i accumulated over the batch.")
    print()
    print("Because the loss is a MEAN over the four examples, L = (1/4) sum_n")
    print("L_n, and differentiation is linear, so dL/dW1 = (1/4) sum_n dL_n/dW1")
    print("- exactly the average of the per-example gradients. The batch size")
    print("therefore changes the gradient scale but not its direction.")
    print()

    # C5: numerical verification that autograd is computing what we claim.
    model = res["model"]
    loss_fn = nn.BCEWithLogitsLoss()

    def loss_value():
        with torch.no_grad():
            return loss_fn(model(X), Y).item()

    eps = 1e-4
    W = model[0].weight
    max_err = 0.0
    for j in range(W.shape[0]):
        for i in range(W.shape[1]):
            original = W.data[j, i].item()
            W.data[j, i] = original + eps
            lp = loss_value()
            W.data[j, i] = original - eps
            lm = loss_value()
            W.data[j, i] = original
            numeric = (lp - lm) / (2 * eps)
            max_err = max(max_err, abs(numeric - W1_grad[j, i].item()))
    print(f"Finite-difference check on W1: max |numeric - autograd| = {max_err:.2e}")


def task4_part_c():
    line("TASK 4 PART C - Symmetry experiment (all weights initialised to zero)")
    model = zero_init(make_mlp("tanh"))
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    W1 = model[0].weight

    print("Hidden-layer weight matrix W1 over the first few steps:")
    for step in range(6):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        rows_equal = torch.allclose(W1[0], W1[1])
        print(f"  step {step}: row0 = {W1[0].tolist()}, row1 = {W1[1].tolist()}, "
              f"rows identical = {rows_equal}, loss = {loss.item():.6f}")
        opt.step()

    # Train it out to the same budget as the random-init run.
    for _ in range(4000):
        opt.zero_grad()
        loss_fn(model(X), Y).backward()
        opt.step()

    with torch.no_grad():
        logits = model(X)
        final_loss = loss_fn(logits, Y).item()
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).float()
    correct = bool((preds == Y.squeeze(1)).all().item())

    print()
    print(f"After 4000 more steps: final loss = {final_loss:.6f}, "
          f"all four correct = {correct}")
    print(f"  probabilities: {[round(p, 4) for p in probs.tolist()]}")
    print(f"  rows still identical: {torch.allclose(W1[0], W1[1])}")
    print()
    print("Explanation: with identical (here, zero) parameters the two hidden")
    print("units compute the same function of every input, so backpropagation")
    print("sends them the same delta and therefore the same gradient. Identical")
    print("parameters plus identical gradients stay identical under a")
    print("deterministic update, so the pair can never differentiate. The")
    print("network behaves as if it had one hidden unit and collapses to a")
    print("linear model, which cannot represent XOR. Random initialisation is")
    print("what breaks this symmetry.")
    return {"final_loss": final_loss, "all_correct": correct,
            "rows_identical": bool(torch.allclose(W1[0], W1[1]))}


def task4_part_d():
    line("TASK 4 PART D - Activation experiment (sigmoid / tanh / ReLU)")
    rows = []
    for act in ["sigmoid", "tanh", "relu"]:
        res = train_binary(make_mlp(act), steps=4000, lr=0.1, record_grad_at=10)
        rows.append((act, res["final_loss"], res["all_correct"],
                     res["early_grad_norm"]))
        print(f"[{act}] final loss = {res['final_loss']:.6f}, "
              f"4/4 correct = {res['all_correct']}, "
              f"early ||dL/dW1||_2 = {res['early_grad_norm']:.6f}")
        print(f"       probabilities: "
              f"{[round(p, 4) for p in res['probs'].tolist()]}")

    print()
    print("Result table")
    print(f"{'Hidden activation':<20}{'Final loss':>14}"
          f"{'4/4 correct?':>16}{'Early ||grad W1||':>20}")
    for act, fl, ok, gn in rows:
        print(f"{act:<20}{fl:>14.6f}{('yes' if ok else 'no'):>16}{gn:>20.6f}")

    # Repeated-run behaviour (validation criterion C4).
    print()
    print("Reproducibility across seeds (criterion C4), 5 seeds per activation:")
    for act in ["sigmoid", "tanh", "relu"]:
        successes = 0
        losses = []
        for seed in range(5):
            r = train_binary(make_mlp(act, seed=seed), steps=4000, lr=0.1)
            successes += int(r["all_correct"])
            losses.append(r["final_loss"])
        print(f"  {act:<8}: {successes}/5 runs solved XOR, "
              f"losses = {[round(l, 4) for l in losses]}")
    return rows


def saturation_diagnostic():
    line("DIAGNOSTIC - Distinguishing sigmoid saturation from dead ReLU units")
    for act in ["sigmoid", "relu"]:
        model = make_mlp(act)
        pre = model[0](X)                   # pre-activations a(1)
        post = model[1](pre)                # activations h(1)
        print(f"[{act}] pre-activations:\n{pre.detach()}")
        print(f"[{act}] activations:\n{post.detach()}")
    print()
    print("How to tell them apart: a saturated sigmoid has an activation pinned")
    print("near 0 or 1 while its pre-activation is large in magnitude - the unit")
    print("is still responding to input, just flatly. A dead ReLU has a negative")
    print("pre-activation and an activation of exactly 0, and its derivative is")
    print("exactly 0 rather than merely small, so it passes no gradient at all")
    print("until some other unit moves its pre-activation back above zero.")


# ---------------------------------------------------------------------------
# Task 5 - three-class extension
# ---------------------------------------------------------------------------

def task5(hidden_act="tanh", steps=4000, lr=0.1):
    line("TASK 5 - Three-class extension (3 logits + cross-entropy)")
    print("Predictions made before running:")
    print("  1. the final weight matrix W2 has shape (3, 2): 3 classes x 2")
    print("     hidden units; its bias has shape (3,);")
    print("  2. three logits per example, i.e. a (4, 3) logit matrix;")
    print("  3. softmax probabilities sum to one because each exponential is")
    print("     positive and they are divided by their own sum - softmax is a")
    print("     normalisation onto the probability simplex;")
    print("  4. the logit gradient is p - y because for cross-entropy with a")
    print("     one-hot target, dL/dz_k = p_k - y_k: the predicted distribution")
    print("     minus the observed one, so a correct confident prediction gives")
    print("     a near-zero gradient and a confident mistake gives a large one.")
    print()

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(2, 2),
        activation(hidden_act),
        nn.Linear(2, 3),          # three logits; softmax lives in the loss
    )
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    initial_loss = None
    for step in range(steps):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y3)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(X)
        final_loss = loss_fn(logits, Y3).item()
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    print(f"Shape check - W2 weight: {tuple(model[2].weight.shape)}, "
          f"bias: {tuple(model[2].bias.shape)}, logits: {tuple(logits.shape)}")
    print(f"initial loss = {initial_loss:.6f}  "
          f"(ln 3 = {math.log(3):.6f} is the chance level)")
    print(f"final loss   = {final_loss:.6f}")
    print()
    print("x1 x2 | true |   p(class 0)  p(class 1)  p(class 2)  | predicted")
    for i in range(4):
        p = probs[i]
        print(f" {int(X[i,0])}  {int(X[i,1])}  |  {int(Y3[i])}   |"
              f"   {p[0]:.6f}    {p[1]:.6f}    {p[2]:.6f}   |     "
              f"{int(preds[i])}")
    print(f"all four correct: {bool((preds == Y3).all().item())}")

    print()
    print("Sum-to-one verification for example (0,1):")
    p = probs[1]
    print(f"  probability vector = {[f'{v:.6f}' for v in p.tolist()]}")
    print(f"  sum = {p.sum().item():.10f}  "
          f"(close to 1: {bool(torch.isclose(p.sum(), torch.tensor(1.0)))})")

    # Verify the p - y logit gradient claim directly on a single example.
    print()
    print("Direct check of the dL/dlogits = p - y claim on example (0,1):")
    z = model(X[1:2]).detach().requires_grad_(True)
    nn.CrossEntropyLoss()(z, Y3[1:2]).backward()
    p_minus_y = torch.softmax(z, dim=1).detach() - torch.nn.functional.one_hot(
        Y3[1:2], num_classes=3).float()
    print(f"  autograd dL/dz : {[f'{v:.6e}' for v in z.grad[0].tolist()]}")
    print(f"  p - y          : {[f'{v:.6e}' for v in p_minus_y[0].tolist()]}")
    print(f"  match: {bool(torch.allclose(z.grad, p_minus_y, atol=1e-6))}")

    # Optional diagnostic: shift invariance of softmax.
    print()
    print("Optional diagnostic - add 100 to every logit of example (0,1):")
    base = logits[1]
    shifted = base + 100.0
    p_base = torch.softmax(base, dim=0)
    p_shift = torch.softmax(shifted, dim=0)
    print(f"  original logits : {[f'{v:.6f}' for v in base.tolist()]}")
    print(f"  shifted logits  : {[f'{v:.6f}' for v in shifted.tolist()]}")
    print(f"  p(original)     : {[f'{v:.8f}' for v in p_base.tolist()]}")
    print(f"  p(shifted)      : {[f'{v:.8f}' for v in p_shift.tolist()]}")
    print(f"  max abs difference: {(p_base - p_shift).abs().max().item():.3e}")
    print()
    print("Why: softmax(z + c)_k = e^(z_k + c) / sum_j e^(z_j + c), and the")
    print("common factor e^c cancels, so the distribution is invariant to a")
    print("constant shift. Stable implementations exploit this by subtracting")
    print("max(z) before exponentiating: the largest exponent becomes e^0 = 1,")
    print("which cannot overflow, and the underflowing small terms only lose")
    print("mass that was negligible anyway. Without the shift, a logit such as")
    print("1000 makes e^z overflow to inf and the result becomes nan.")

    naive = torch.exp(torch.tensor([1000.0, 999.0, 998.0]))
    print(f"  naive exp of large logits -> {naive.tolist()} (overflow)")
    print(f"  stable softmax of the same logits -> "
          f"{[f'{v:.6f}' for v in torch.softmax(torch.tensor([1000.0, 999.0, 998.0]), 0).tolist()]}")

    return {"final_loss": final_loss, "probs": probs, "preds": preds}


# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    task1_specification()
    task2_specification()
    lin_res, mlp_res = task4_part_a()
    task4_part_b(mlp_res)
    task4_part_c()
    task4_part_d()
    saturation_diagnostic()
    task5()
    line("END OF LABORATORY RUN")


if __name__ == "__main__":
    main()
