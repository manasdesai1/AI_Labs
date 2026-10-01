import random
from collections import Counter, defaultdict

# ------------------------------------------------------------
# Autoregressive Language Model as a Bayesian Network
# ------------------------------------------------------------
# The chain rule factorises a sequence as
#   P(x1..xT) = prod_t P(xt | x1..x(t-1))
#
# A first-order model assumes P(xt | x1..x(t-1)) = P(xt | x(t-1)),
# giving the network X1 -> X2 -> ... -> XT. A second-order model
# conditions on the previous two tokens instead.
#
# Each sequence is padded with `order` <START> tokens so that
# every position has a full context, and ends with <END>.
# ------------------------------------------------------------


START = "<START>"
END = "<END>"
SEED = 0
MAX_LENGTH = 20

CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenise(corpus, order):
    """
    Split sentences into tokens and add the special tokens.

    Parameters:
        corpus : list of sentence strings
        order  : model order, which sets the number of <START> pads

    Returns:
        List of token lists.
    """
    return [[START] * order + sentence.lower().split() + [END]
            for sentence in corpus]


def vocabulary(sequences):
    """
    Collect every distinct token.

    Parameters:
        sequences : list of token lists

    Returns:
        Sorted list of tokens.
    """
    return sorted({token for sequence in sequences for token in sequence})


def count_transitions(sequences, order):
    """
    Count how often each token follows each context.

    Parameters:
        sequences : list of token lists
        order     : number of preceding tokens in a context

    Returns:
        Dictionary mapping a context tuple to a Counter of
        following tokens.
    """
    counts = defaultdict(Counter)

    for sequence in sequences:
        for i in range(len(sequence) - order):
            context = tuple(sequence[i:i + order])
            counts[context][sequence[i + order]] += 1

    return counts


def build_cpt(counts):
    """
    Normalise transition counts into conditional distributions.

    Parameters:
        counts : context -> Counter of following tokens

    Returns:
        Dictionary mapping a context to {token: probability}.
    """
    cpt = {}

    for context, followers in counts.items():
        total = sum(followers.values())
        cpt[context] = {token: count / total
                        for token, count in followers.items()}

    return cpt


def distribution(cpt, context):
    """
    Look up one conditional distribution.

    Parameters:
        cpt     : conditional probability table
        context : tuple of preceding tokens

    Returns:
        {token: probability}, empty if the context was never seen.
    """
    return cpt.get(context, {})


def most_probable(cpt, context):
    """
    The argmax of a conditional distribution.

    Ties are broken alphabetically so the result is repeatable.

    Parameters:
        cpt     : conditional probability table
        context : tuple of preceding tokens

    Returns:
        The most probable token, or None for an unseen context.
    """
    dist = distribution(cpt, context)

    if not dist:
        return None

    return min(dist.items(), key=lambda item: (-item[1], item[0]))[0]


def check_normalisation(cpt):
    """
    Check that every conditional distribution sums to one.

    Parameters:
        cpt : conditional probability table

    Returns:
        List of (context, total) pairs that are not within 1e-9
        of 1.0.
    """
    return [(context, sum(dist.values()))
            for context, dist in cpt.items()
            if abs(sum(dist.values()) - 1.0) > 1e-9]


def generate(cpt, order, mode, rng, max_length=MAX_LENGTH):
    """
    Generate one sentence from the model.

    Parameters:
        cpt        : conditional probability table
        order      : model order
        mode       : 'greedy' or 'sample'
        rng        : random.Random instance
        max_length : cap on generated tokens

    Returns:
        (tokens, terminated) where terminated says whether <END>
        was reached rather than the cap.
    """
    context = tuple([START] * order)
    tokens = []

    while len(tokens) < max_length:

        dist = distribution(cpt, context)

        if not dist:
            return tokens, False

        if mode == "greedy":
            token = most_probable(cpt, context)
        else:
            token = rng.choices(list(dist), weights=list(dist.values()))[0]

        if token == END:
            return tokens, True

        tokens.append(token)
        context = tuple(list(context[1:]) + [token])

    return tokens, False


def model_statistics(counts, cpt, vocab, order):
    """
    Summarise the size and sparsity of a model.

    Parameters:
        counts : context -> Counter of following tokens
        cpt    : conditional probability table
        vocab  : list of tokens
        order  : model order

    Returns:
        Dictionary of counts and totals.
    """
    possible_contexts = len(vocab) ** order
    parameters = sum(len(dist) for dist in cpt.values())

    return {
        "possible_contexts": possible_contexts,
        "observed_contexts": len(counts),
        "unseen_contexts": possible_contexts - len(counts),
        "parameters": parameters,
        "table_cells": possible_contexts * len(vocab),
        "zero_cells": possible_contexts * len(vocab) - parameters,
    }


def print_cpt(cpt, contexts, label):
    """
    Print selected conditional distributions.

    Parameters:
        cpt      : conditional probability table
        contexts : list of context tuples to show
        label    : heading to print
    """
    print(f"\n{label}")

    for context in contexts:
        dist = distribution(cpt, context)
        key = " ".join(context)

        if not dist:
            print(f"  P(. | {key}) : unseen context")
            continue

        ordered = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
        shown = ", ".join(f"{token} {prob:.4f}" for token, prob in ordered)
        print(f"  P(. | {key}) : {shown}")


def main():
    """
    Build both models, test them and generate text.
    """
    rng = random.Random(SEED)

    # --------------------------------------------------------
    # First-order model
    # --------------------------------------------------------
    sequences1 = tokenise(CORPUS, 1)
    vocab = vocabulary(sequences1)
    counts1 = count_transitions(sequences1, 1)
    cpt1 = build_cpt(counts1)

    print("Sentences      :", len(CORPUS))
    print("Vocabulary     :", len(vocab))
    print("Tokens         :", " ".join(vocab))
    print("Example sequence:", " ".join(sequences1[0]))

    required = ["the", "cat", "dog", "sat", "ran"]
    print_cpt(cpt1, [(w,) for w in required],
              "First-order CPT for the required words")

    print("\nRaw counts for 'the'")
    for token, count in sorted(counts1[("the",)].items(),
                               key=lambda i: (-i[1], i[0])):
        print(f"  C(the, {token}) = {count}")
    print("  total =", sum(counts1[("the",)].values()))

    # --------------------------------------------------------
    # Normalisation test
    # --------------------------------------------------------
    failures = check_normalisation(cpt1)
    print("\nNormalisation test, first-order")
    print("  contexts checked :", len(cpt1))
    print("  contexts failing :", len(failures))
    for context, total in failures:
        print("  ", " ".join(context), total)

    # --------------------------------------------------------
    # Next-word prediction
    # --------------------------------------------------------
    print("\nMost probable next word")
    for word in required:
        context = (word,)
        token = most_probable(cpt1, context)
        prob = distribution(cpt1, context).get(token, 0.0)
        print(f"  argmax P(w | {word}) = {token} ({prob:.4f})")

    # --------------------------------------------------------
    # Generation, 20 sampled sentences
    # --------------------------------------------------------
    print("\nTwenty sampled sentences, first-order")
    sampled1 = []

    for i in range(20):
        tokens, terminated = generate(cpt1, 1, "sample", rng)
        sampled1.append(" ".join(tokens))
        print(f"  {i + 1:2}. {' '.join(tokens)}"
              f"{'' if terminated else '   [cut off]'}")

    print("  distinct sentences:", len(set(sampled1)), "of 20")

    # --------------------------------------------------------
    # Greedy against sampling
    # --------------------------------------------------------
    print("\nGreedy, first-order, five runs")
    for i in range(5):
        tokens, terminated = generate(cpt1, 1, "greedy", rng)
        print(f"  {i + 1}. {' '.join(tokens)}"
              f"{'' if terminated else '   [cut off]'}")

    print("\nSampling, first-order, five runs")
    for i in range(5):
        tokens, terminated = generate(cpt1, 1, "sample", rng)
        print(f"  {i + 1}. {' '.join(tokens)}"
              f"{'' if terminated else '   [cut off]'}")

    # --------------------------------------------------------
    # Second-order model
    # --------------------------------------------------------
    sequences2 = tokenise(CORPUS, 2)
    counts2 = count_transitions(sequences2, 2)
    cpt2 = build_cpt(counts2)

    print_cpt(cpt2,
              [(START, "the"), ("the", "cat"), ("the", "dog"),
               ("cat", "sat"), ("on", "the"), ("to", "the")],
              "Second-order CPT for selected contexts")

    failures2 = check_normalisation(cpt2)
    print("\nNormalisation test, second-order")
    print("  contexts checked :", len(cpt2))
    print("  contexts failing :", len(failures2))

    print("\nGreedy, second-order, five runs")
    for i in range(5):
        tokens, terminated = generate(cpt2, 2, "greedy", rng)
        print(f"  {i + 1}. {' '.join(tokens)}"
              f"{'' if terminated else '   [cut off]'}")

    print("\nSampling, second-order, five runs")
    sampled2 = []
    for i in range(5):
        tokens, terminated = generate(cpt2, 2, "sample", rng)
        sampled2.append(" ".join(tokens))
        print(f"  {i + 1}. {' '.join(tokens)}"
              f"{'' if terminated else '   [cut off]'}")

    print("\nTwenty sampled sentences, second-order")
    sampled2_all = []
    for i in range(20):
        tokens, terminated = generate(cpt2, 2, "sample", rng)
        sampled2_all.append(" ".join(tokens))
    print("  distinct sentences:", len(set(sampled2_all)), "of 20")

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------
    stats1 = model_statistics(counts1, cpt1, vocab, 1)
    stats2 = model_statistics(counts2, cpt2, vocab, 2)

    print("\nModel comparison")
    print(f"  {'Measure':<22}{'First':>10}{'Second':>10}")
    for key in ("possible_contexts", "observed_contexts",
                "unseen_contexts", "parameters", "table_cells",
                "zero_cells"):
        print(f"  {key:<22}{stats1[key]:>10}{stats2[key]:>10}")

    in_corpus1 = sum(1 for s in sampled1 if s in CORPUS)
    in_corpus2 = sum(1 for s in sampled2_all if s in CORPUS)
    print(f"  {'distinct of 20':<22}{len(set(sampled1)):>10}"
          f"{len(set(sampled2_all)):>10}")
    print(f"  {'in training corpus':<22}{in_corpus1:>10}{in_corpus2:>10}")

    # --------------------------------------------------------
    # Save the generated sentences
    # --------------------------------------------------------
    with open("bn_lab_generated.txt", "w", encoding="utf-8") as handle:
        handle.write("First-order, 20 sampled sentences\n")
        for sentence in sampled1:
            handle.write(sentence + "\n")
        handle.write("\nSecond-order, 20 sampled sentences\n")
        for sentence in sampled2_all:
            handle.write(sentence + "\n")


# ------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
