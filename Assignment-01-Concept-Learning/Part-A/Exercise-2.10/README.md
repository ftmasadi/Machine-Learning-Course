# Concept Learning – Find-S (Mitchell, Ch. 2, EnjoySport)

Machine Learning course, Shahid Bahonar University of Kerman.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/colab_demo.ipynb)


## Files
- `src/concept_learning.py` – hypothesis representation, `covers`, `more_general_or_equal`,
  Find-S, random instance generator, `run_once` (stopping conditions).
- `colab_demo.ipynb` – runs everything in Google Colab (clone, run, show tables/figure).
- `run_all.py` – reproduces the Find-S trace on Table 2.1, the Exercise 2.10 experiment
  (20 runs, seeds 0–19, Nmax = 1000) and the "number of ?" study, tables and figure.

## Run
```
pip install -r requirements.txt
python run_all.py
```
Results (CSV files and figure) are written to `results/`.
Python version used: <write your version here>

## Notes
- No ML libraries are used for Find-S, the hypothesis representation, the
  "more general than" test or the instance generator (NumPy is only used for statistics, Matplotlib for the plot).
- Standard deviation is the sample standard deviation (ddof = 1); median/IQR use NumPy's linear percentiles.
- Runs stopped by Nmax are reported separately and excluded from the statistics (none occurred).
