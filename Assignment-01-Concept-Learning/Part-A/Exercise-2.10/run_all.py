"""Reproduces the Find-S trace and every number/figure of Exercise 2.10.

Usage (from anywhere):  python run_all.py
Outputs are written to ./results/ next to this script.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.concept_learning import (  # noqa: E402
    ANY, STOP_EXACT, STOP_NMAX, covers, find_s, fmt, make_target,
    more_general_or_equal, run_once,
)

SEEDS = list(range(20))      # recorded seeds 0..19
N_MAX = 1000
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

TABLE_2_1 = [
    (("Sunny", "Warm", "Normal", "Strong", "Warm", "Same"), True),
    (("Sunny", "Warm", "High", "Strong", "Warm", "Same"), True),
    (("Rainy", "Cold", "High", "Strong", "Warm", "Change"), False),
    (("Sunny", "Warm", "High", "Strong", "Cool", "Change"), True),
]


def summarize(values):
    """Mean, sample std (ddof=1), min, max, median, IQR (numpy linear percentiles)."""
    a = np.array(values, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(mean=a.mean(), std=a.std(ddof=1) if len(a) > 1 else 0.0,
                min=a.min(), max=a.max(), median=med, iqr=q3 - q1)


# ------------------------------------------------------------------ 1. trace on Table 2.1
print("=== 1. Find-S trace on Table 2.1 ===")
trace = find_s(TABLE_2_1, verbose=True)
EXPECTED = [
    "<∅,∅,∅,∅,∅,∅>",
    "<Sunny,Warm,Normal,Strong,Warm,Same>",
    "<Sunny,Warm,?,Strong,Warm,Same>",
    "<Sunny,Warm,?,Strong,Warm,Same>",
    "<Sunny,Warm,?,Strong,?,?>",
]
assert [fmt(h) for h in trace] == EXPECTED, "trace differs from Section 2.4 of the textbook"
print("OK: every step matches Section 2.4 of the textbook\n")

# ------------------------------------------------------------------ 2. small sanity checks
# Part B, question 3: two positive examples
ex_ab = [(("Rainy", "Warm", "High", "Weak", "Warm", "Same"), True),
         (("Rainy", "Cold", "High", "Weak", "Cool", "Same"), True)]
assert fmt(find_s(ex_ab)[-1]) == "<Rainy,?,High,Weak,?,Same>"
# Part B, question 5: label of example 1 flipped to "No"
flipped = [(TABLE_2_1[0][0], False)] + TABLE_2_1[1:]
assert fmt(find_s(flipped)[-1]) == "<Sunny,Warm,High,Strong,?,?>"
# the output is consistent with all data, and so is the more general <Sunny,Warm,?,?,?,?>
other = ("Sunny", "Warm", ANY, ANY, ANY, ANY)
assert all(covers(other, x) == lab for x, lab in TABLE_2_1)
assert more_general_or_equal(other, trace[-1]) and not more_general_or_equal(trace[-1], other)
print("Part B checks passed.\n")

# ------------------------------------------------------------------ 3. main experiment
TARGET = ("Sunny", "Warm", ANY, ANY, ANY, ANY)
print(f"=== 3. Experiment: target {fmt(TARGET)}, {len(SEEDS)} runs, Nmax={N_MAX} ===")
rows = []
for s in SEEDS:
    r = run_once(TARGET, s, N_MAX, check_invariants=True)
    rows.append((s, r.n_examples, r.n_positive, r.stop_reason))
    print(f"seed {s:2d}: {r.n_examples:4d} examples ({r.n_positive} positive)  stopped by {r.stop_reason}")

with open(OUT / "exp_2_10_runs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["seed", "n_examples", "n_positive", "stop_reason"])
    w.writerows(rows)

exact = [r for r in rows if r[3] == STOP_EXACT]
capped = [r for r in rows if r[3] == STOP_NMAX]
print(f"\nRuns ended by (i): {len(exact)}   runs ended by (ii): {len(capped)} "
      f"(reported separately, not in the statistics)")
if exact:
    S = summarize([r[1] for r in exact])
    print("examples : mean ± std = {mean:.2f} ± {std:.2f}   median = {median:.1f}   "
          "IQR = {iqr:.2f}   min = {min:.0f}   max = {max:.0f}".format(**S))
    P = summarize([r[2] for r in exact])
    print("positives: mean = {mean:.2f}, median = {median:.1f}".format(**P))

# ------------------------------------------------------------------ 4. more '?' in the target
print("\n=== 4. Effect of the number of '?' in the target ===")
BASE = ("Sunny", "Warm", "Normal", "Strong", "Warm", "Same")
table = []
for k in range(0, 7):
    target = make_target(BASE, k)
    res = [run_once(target, s, N_MAX) for s in SEEDS]
    good = [r.n_examples for r in res if r.stop_reason == STOP_EXACT]
    n_cap = sum(r.stop_reason == STOP_NMAX for r in res)
    S = summarize(good) if good else None
    table.append((k, fmt(target), len(good), n_cap, S))
    if S:
        print(f"k={k}: {fmt(target):38s} mean={S['mean']:7.2f} ± {S['std']:6.2f}  "
              f"median={S['median']:6.1f}  IQR={S['iqr']:6.2f}  "
              f"min={S['min']:.0f} max={S['max']:.0f}  (capped runs: {n_cap})")
    else:
        print(f"k={k}: {fmt(target):38s} all {n_cap} runs hit Nmax")

with open(OUT / "exp_2_10_vs_question_marks.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["k_question_marks", "target", "runs_exact", "runs_capped",
                "mean", "std", "median", "iqr", "min", "max"])
    for k, t, n_ok, n_cap, S in table:
        if S:
            w.writerow([k, t, n_ok, n_cap, f"{S['mean']:.2f}", f"{S['std']:.2f}",
                        f"{S['median']:.1f}", f"{S['iqr']:.2f}", f"{S['min']:.0f}", f"{S['max']:.0f}"])
        else:
            w.writerow([k, t, n_ok, n_cap] + [""] * 6)

# ------------------------------------------------------------------ 5. figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(k, S) for k, _, _, _, S in table if S]
    plt.figure(figsize=(6, 4))
    plt.errorbar([k for k, _ in pts], [S["mean"] for _, S in pts],
                 yerr=[S["std"] for _, S in pts], marker="o", capsize=3)
    plt.xlabel('Number of "?" constraints in the target')
    plt.ylabel("Training examples until Find-S = target (mean ± std)")
    plt.title(f"Find-S sample need ({len(SEEDS)} runs per point, seeds 0-{SEEDS[-1]})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_examples_vs_question_marks.png", dpi=150)
    print(f"\nFigure saved to {OUT / 'fig_examples_vs_question_marks.png'}")
except ImportError:
    print("\nmatplotlib not installed: figure skipped (pip install matplotlib)")
