# Computational oncology code examples

I prepared these examples while translating the public-cohort part of my
master's thesis from R into Python and then testing small PyTorch models. The
repository keeps the original analysis and the later learning projects together
so that the progression is easy to follow.

## Suggested starting point

The main example is in
[`prostate_fusion_cohort_analysis/`](prostate_fusion_cohort_analysis/):

1. [`original_thesis_analysis.R`](prostate_fusion_cohort_analysis/original_thesis_analysis.R)
   is the R script used for the thesis analysis.
2. [`Figure_Violin_SUC_TCGA_FINAL_compact_axes_new.pdf`](prostate_fusion_cohort_analysis/Figure_Violin_SUC_TCGA_FINAL_compact_axes_new.pdf)
   is the corresponding figure.
3. [`analysis_walkthrough.ipynb`](prostate_fusion_cohort_analysis/analysis_walkthrough.ipynb)
   explains the later Python implementation.
4. [`pytorch_ml_walkthrough.ipynb`](prostate_fusion_cohort_analysis/pytorch_ml_walkthrough.ipynb)
   contains a small PyTorch extension.

The R script uses local cBioPortal exports. The Python version was written
afterwards and retrieves the public data through the cBioPortal API. The Python
and PyTorch code was not used to generate the submitted thesis.

## Additional experiments

- [`rai2_cohort_regression/`](rai2_cohort_regression/) explores whether fusion
  status and CTBP1/2 contain information about RAI2 expression within each
  cohort.
- [`ML-Beispiel/`](ML-Beispiel/) compares linear and small neural-network models
  for predicting fusion status in held-out TCGA samples.

These folders document later learning experiments. They are optional; the main
R-to-Python project is sufficient for a short review.

## Scope

The analyses use public cohort data and are not clinical diagnostic tools. TCGA
and SU2C differ in disease stage, sample composition and expression processing,
so results are interpreted within each cohort and cross-cohort differences are
reported. No data from previous employers or directly identifying patient data
are included.

Each folder has its own README, notebook, dependency file and tests. A typical
notebook setup is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
python -m jupyter lab
```

## Repository

This repository is a technical portfolio for academic and research
applications. It contains public research data and code written for the thesis
analysis or for the later learning projects described above.
