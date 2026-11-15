"""Concept learning building blocks for the EnjoySport task (Mitchell, Chapter 2).

Everything here is written from scratch (no ML libraries):
  * hypothesis representation
  * the "more general than or equal to" test
  * the covers (matches) test
  * Find-S
  * a random instance generator
"""
import random

# ---------------------------------------------------------------- representation
# An instance is a tuple of 6 attribute values.
# A hypothesis is a tuple of 6 constraints; each constraint is a specific value,
# "?" (any value is acceptable) or "0" (no value is acceptable, written ∅ in the text).
ATTRIBUTES = ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"]
VALUES = {
    "Sky": ["Sunny", "Cloudy", "Rainy"],
    "AirTemp": ["Warm", "Cold"],
    "Humidity": ["Normal", "High"],
    "Wind": ["Strong", "Weak"],
    "Water": ["Warm", "Cool"],
    "Forecast": ["Same", "Change"],
}
ANY = "?"
NONE = "0"  # stands for the empty constraint ∅


def most_specific():
    """<0,0,0,0,0,0>: rejects every instance."""
    return (NONE,) * len(ATTRIBUTES)


def covers(h, x):
    """True if hypothesis h classifies instance x as positive."""
    for c, v in zip(h, x):
        if c == NONE:
            return False
        if c != ANY and c != v:
            return False
    return True


def more_general_or_equal(h1, h2):
    """True if h1 >=_g h2: every instance covered by h2 is also covered by h1."""
    if NONE in h2:  # h2 covers nothing, so any h1 is at least as general
        return True
    for c1, c2 in zip(h1, h2):
        if c1 == ANY:
            continue
        if c1 == NONE:
            return False
        if c2 != c1:  # c1 is a value; c2 is ANY or a different value
            return False
    return True


# ---------------------------------------------------------------- Find-S
def find_s_step(h, x, label):
    """One Find-S update. Negative examples are ignored."""
    if not label:
        return h
    new_h = []
    for c, v in zip(h, x):
        if c == NONE:
            new_h.append(v)          # first positive example: copy its value
        elif c == ANY or c == v:
            new_h.append(c)          # constraint already satisfied: keep
        else:
            new_h.append(ANY)        # minimal generalization
    return tuple(new_h)


def find_s(examples, verbose=False):
    """Run Find-S on a list of (instance, label) pairs; return the final hypothesis."""
    h = most_specific()
    if verbose:
        print("h0 =", fmt(h))
    for i, (x, label) in enumerate(examples, 1):
        h = find_s_step(h, x, label)
        if verbose:
            print(f"h{i} =", fmt(h), "  (", "+" if label else "-", ")")
    return h


# ---------------------------------------------------------------- generator
def random_instance(rng):
    """Draw one instance uniformly at random from the 96 possible instances."""
    return tuple(rng.choice(VALUES[a]) for a in ATTRIBUTES)


def labelled_stream(target, rng):
    """Infinite stream of (instance, label) pairs, labelled by the target concept."""
    while True:
        x = random_instance(rng)
        yield x, covers(target, x)


def fmt(h):
    return "<" + ",".join("∅" if c == NONE else c for c in h) + ">"


# ---------------------------------------------------------------- experiment
def run_once(target, seed, n_max=1000):
    """Feed random labelled examples to Find-S until it outputs exactly `target`
    or n_max examples were used. Returns (n_examples, n_positive, stop_reason)."""
    rng = random.Random(seed)
    h = most_specific()
    n_pos = 0
    for n, (x, label) in enumerate(labelled_stream(target, rng), 1):
        if label:
            n_pos += 1
        h = find_s_step(h, x, label)
        if h == target:
            return n, n_pos, "(i) exact target"
        if n >= n_max:
            return n, n_pos, "(ii) Nmax reached"
