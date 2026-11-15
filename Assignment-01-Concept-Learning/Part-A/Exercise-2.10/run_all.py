"""Reproduces every number and figure for Exercise 2.10 from recorded seeds.

Usage:  python run_all.py
Outputs go to ./results/
"""
import csv
import os
import statistics as st

import numpy as np

from src.concept_learning import find_s, run_once, fmt, ANY

SEEDS = list(range(20))   # recorded seeds 0..19
N_MAX = 1000
OUT = "results"
os.makedirs(OUT, exist_ok=True)

TABLE_2_1 = [
    (("Sunny", "Warm", "Normal", "Strong", "Warm", "Same"), True),
    (("Sunny", "Warm", "High", "Strong", "Warm", "Same"), True),
    (("Rainy", "Cold", "High", "Strong", "Warm", "Change"), False),
    (("Sunny", "Warm", "High", "Strong", "Cool", "Change"), True),
]


def summarize(values):
    a = np.array(values, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(mean=a.mean(), std=a.std(ddof=1) if len(a) > 1 else 0.0,
                min=a.min(), max=a.max(), median=med, iqr=q3 - q1)


# ---------------------------------------------------------------- part 1: trace
print("=== Find-S trace on Table 2.1 ===")
final = find_s(TABLE_2_1, verbose=True)
assert fmt(final) == "<Sunny,Warm,?,Strong,?,?>", "trace does not match the textbook"
print("OK: matches Section 2.4 of the textbook\n")

# ---------------------------------------------------------------- part 3: main experiment
TARGET = ("Sunny", "Warm", ANY, ANY, ANY, ANY)
print(f"=== Experiment: target {fmt(TARGET)}, {len(SEEDS)} runs, Nmax={N_MAX} ===")
rows = []
for s in SEEDS:
    n, n_pos, why = run_once(TARGET, s, N_MAX)
    rows.append((s, n, n_pos, why))
    print(f"seed {s:2d}: {n:4d} examples ({n_pos} positive)  stopped by {why}")

with open(f"{OUT}/exp_2_10_runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "n_examples", "n_positive", "stop_reason"])
    w.writerows(rows)

ok = [r[1] for r in rows if r[3].startswith("(i)")]
capped = [r for r in rows if r[3].startswith("(ii)")]
print(f"\nRuns ended by (i): {len(ok)}   runs ended by (ii): {len(capped)} (reported separately, not in the mean)")
if ok:
    S = summarize(ok)
    print("mean ± std = {mean:.2f} ± {std:.2f}   median = {median:.1f}   IQR = {iqr:.2f}   min = {min:.0f}   max = {max:.0f}".format(**S))
    pos = summarize([r[2] for r in rows if r[3].startswith("(i)")])
    print("positive examples used: mean = {mean:.2f}, median = {median:.1f}".format(**pos))

# ---------------------------------------------------------------- part 4: more '?' in the target
print("\n=== Effect of the number of '?' in the target ===")
BASE = ("Sunny", "Warm", "Normal", "Strong", "Warm", "Same")
table = []
for k in range(0, 7):                       # k = number of "?" constraints
    target = BASE[:6 - k] + (ANY,) * k
    res = [run_once(target, s, N_MAX) for s in SEEDS]
    good = [r[0] for r in res if r[2].startswith("(i)")]
    cap = sum(1 for r in res if r[2].startswith("(ii)"))
    if good:
        S = summarize(good)
        table.append((k, fmt(target), len(good), cap, S))
        print(f"k={k}: {fmt(target):40s} mean={S['mean']:7.2f} ± {S['std']:6.2f}  median={S['median']:6.1f}  IQR={S['iqr']:6.2f}  min={S['min']:.0f} max={S['max']:.0f}  (capped runs: {cap})")
    else:
        table.append((k, fmt(target), 0, cap, None))
        print(f"k={k}: {fmt(target):40s} all {cap} runs hit Nmax")

with open(f"{OUT}/exp_2_10_vs_question_marks.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k_question_marks", "target", "runs_exact", "runs_capped", "mean", "std", "median", "iqr", "min", "max"])
    for k, t, ne, nc, S in table:
        if S:
            w.writerow([k, t, ne, nc, f"{S['mean']:.2f}", f"{S['std']:.2f}", S["median"], f"{S['iqr']:.2f}", S["min"], S["max"]])
        else:
            w.writerow([k, t, ne, nc, "", "", "", "", "", ""])

# ---------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [t[0] for t in table if t[4]]
    means = [t[4]["mean"] for t in table if t[4]]
    stds = [t[4]["std"] for t in table if t[4]]
    plt.figure(figsize=(6, 4))
    plt.errorbar(ks, means, yerr=stds, marker="o", capsize=3)
    plt.xlabel('Number of "?" constraints in the target')
    plt.ylabel("Training examples until Find-S = target (mean ± std)")
    plt.title(f"Find-S sample need ({len(SEEDS)} runs per point, seeds 0-{SEEDS[-1]})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_examples_vs_question_marks.png", dpi=150)
    print(f"\nFigure saved to {OUT}/fig_examples_vs_question_marks.png")
except ImportError:
    print("\nmatplotlib not installed: figure skipped (pip install matplotlib)")
