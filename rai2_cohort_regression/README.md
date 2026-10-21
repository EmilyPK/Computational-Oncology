# Cohort-aware RAI2 regression

This is a follow-up experiment based on the public TCGA and SU2C expression
table. It asks whether fusion status and CTBP1/2 contain information about RAI2
expression within each cohort.

I kept TCGA and SU2C separate because they differ in disease stage, sample
composition and expression processing.

## Model

- Target: corrected log2 RAI2 expression
- Main predictors: fusion status, CTBP1 and CTBP2
- Comparisons: CTBP1/2 without fusion status, and fusion status alone
- Model: ridge regression fitted with PyTorch
- Evaluation: nested cross-validation performed separately in each cohort

The linear model is intentional: with at most three predictors it is easier to
interpret than a multilayer network. The analysis is observational, so neither
the coefficients nor prediction performance establish causality.

## Files

- [`rai2_regression_walkthrough.ipynb`](rai2_regression_walkthrough.ipynb) — guided analysis
- [`rai2_regression.py`](rai2_regression.py) — command-line implementation
- `data/expression_values.csv` — input copied from the cohort-analysis project
- `tests/test_rai2_regression.py` — lightweight tests

## Run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python rai2_regression.py
```

Generated metrics, predictions and figures are written to `results/`. To open
the notebook, install `requirements-notebook.txt` and run JupyterLab.

## Tests

```bash
python -m unittest discover -s tests -v
```
