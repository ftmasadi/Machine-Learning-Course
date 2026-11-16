"""Concept-learning building blocks for the EnjoySport task (Mitchell, Chapter 2).

Everything is written from scratch (no ML libraries):
  * hypothesis representation
  * the covers (matches) test
  * the "more general than or equal to" test
  * Find-S
  * a random instance generator
  * one experiment run with the two stopping conditions of the assignment
"""
import random
from collections import namedtuple

# ------------------------------------------------------------------ representation
# Instance   : tuple of 6 attribute values, e.g. ("Sunny", "Warm", "Normal", ...)
# Hypothesis : tuple of 6 constraints. Each constraint is
#                - a specific value     (only that value is accepted)
#                - ANY  = "?"           (every value is accepted)
#                - EMPTY = None         (no value is accepted, written ∅ in the text)
ATTRIBUTES = ("Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast")
VALUES = {
    "Sky": ("Sunny", "Cloudy", "Rainy"),
    "AirTemp": ("Warm", "Cold"),
    "Humidity": ("Normal", "High"),
    "Wind": ("Strong", "Weak"),
    "Water": ("Warm", "Cool"),
    "Forecast": ("Same", "Change"),
}
ANY = "?"
EMPTY = None
N_ATTR = len(ATTRIBUTES)


def most_specific():
    """<∅,∅,∅,∅,∅,∅>: rejects every instance."""
    return (EMPTY,) * N_ATTR


def fmt(h):
    """Pretty string such as <Sunny,Warm,?,Strong,?,?>."""
    return "<" + ",".join("∅" if c is EMPTY else c for c in h) + ">"


# ------------------------------------------------------------------ tests
def covers(h, x):
    """True if hypothesis h classifies instance x as positive."""
    for c, v in zip(h, x):
        if c is EMPTY:
            return False
        if c != ANY and c != v:
            return False
    return True


def more_general_or_equal(h1, h2):
    """True if h1 >=_g h2, i.e. every instance covered by h2 is covered by h1."""
    if EMPTY in h2:        # h2 covers nothing, so any h1 is at least as general
        return True
    if EMPTY in h1:        # h1 covers nothing but h2 covers something
        return False
    # both without ∅: h1 must be "?" or equal to h2 in every position
    return all(c1 == ANY or c1 == c2 for c1, c2 in zip(h1, h2))


# ------------------------------------------------------------------ Find-S
def find_s_step(h, x, label):
    """One Find-S update. Negative examples are ignored."""
    if not label:
        return h
    new_h = []
    for c, v in zip(h, x):
        if c is EMPTY:
            new_h.append(v)       # first positive example: copy its value
        elif c == ANY or c == v:
            new_h.append(c)       # already satisfied: keep
        else:
            new_h.append(ANY)     # minimal generalization
    return tuple(new_h)


def find_s(examples, verbose=False):
    """Run Find-S on a list of (instance, label) pairs.

    Returns the list of hypotheses [h0, h1, ..., hn]; the last one is the output.
    """
    h = most_specific()
    trace = [h]
    if verbose:
        print("h0 =", fmt(h))
    for i, (x, label) in enumerate(examples, 1):
        h = find_s_step(h, x, label)
        trace.append(h)
        if verbose:
            print(f"h{i} = {fmt(h)}   ({'+' if label else '-'})")
    return trace


# ------------------------------------------------------------------ generator
def random_instance(rng):
    """Draw one instance uniformly at random from the 96 possible instances."""
    return tuple(rng.choice(VALUES[a]) for a in ATTRIBUTES)


def labelled_stream(target, rng):
    """Infinite stream of (instance, label) pairs labelled by the target concept."""
    while True:
        x = random_instance(rng)
        yield x, covers(target, x)


# ------------------------------------------------------------------ experiment
RunResult = namedtuple("RunResult", "n_examples n_positive stop_reason")
STOP_EXACT = "(i) exact target"
STOP_NMAX = "(ii) Nmax reached"


def make_target(base, k):
    """Target = `base` with its last k constraints replaced by '?'."""
    return tuple(base[: N_ATTR - k]) + (ANY,) * k


def run_once(target, seed, n_max=1000, check_invariants=False):
    """Feed random labelled examples to Find-S until its output equals `target`
    (condition i) or n_max examples were used (condition ii), whichever comes first.

    If check_invariants is True, assert after every example that the hypothesis only
    gets more general and never becomes more general than the target.
    """
    rng = random.Random(seed)
    h = most_specific()
    n_pos = 0
    for n, (x, label) in enumerate(labelled_stream(target, rng), 1):
        n_pos += bool(label)
        new_h = find_s_step(h, x, label)
        if check_invariants:
            assert more_general_or_equal(new_h, h)
            assert more_general_or_equal(target, new_h)
        h = new_h
        if h == target:
            return RunResult(n, n_pos, STOP_EXACT)
        if n >= n_max:
            return RunResult(n, n_pos, STOP_NMAX)
