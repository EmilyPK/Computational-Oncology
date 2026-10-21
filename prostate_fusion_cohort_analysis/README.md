# TMPRSS2–ERG cohort analysis

This folder follows the public-cohort part of my master's thesis from the
original R analysis to later Python and PyTorch implementations. It focuses on
ERG, RAI2, CTBP1 and CTBP2 expression in prostate-cancer samples grouped by
detected TMPRSS2–ERG fusion status.

## Files to review

| File | Purpose |
|---|---|
| [`original_thesis_analysis.R`](original_thesis_analysis.R) | Original R analysis and figure layout |
| [`Figure_Violin_SUC_TCGA_FINAL_compact_axes_new.pdf`](Figure_Violin_SUC_TCGA_FINAL_compact_axes_new.pdf) | Figure produced from the R workflow |
| [`analysis_walkthrough.ipynb`](analysis_walkthrough.ipynb) | Step-by-step Python analysis |
| [`analysis.py`](analysis.py) | Tested command-line Python implementation |
| [`pytorch_ml_walkthrough.ipynb`](pytorch_ml_walkthrough.ipynb) | PyTorch learning experiment |
| [`ml_extension.py`](ml_extension.py) | Command-line PyTorch implementation |

For a short review, the R script and the two notebooks are the most useful
entry points.

## Original R analysis

The R script reads five local exports from cBioPortal, compares the two fusion
groups with Wilcoxon tests and creates the combined TCGA/SU2C figure. Its use of
`SUC` in names and comments refers to the SU2C cohort.

Expected input files:

- `mRNA expression new.txt`
- `prad_su2c_2019_clinical_data.tsv`
- `structural_variants.txt`
- `mRNA Expression TCGA.txt`
- `structural_variants TCGA.txt`

Required R packages are `ggplot2`, `gridExtra` and `gtable`. The exported source
tables are not redistributed in this repository.

## Python implementation

I wrote the Python version after the thesis to make the analysis easier to
repeat and inspect. It downloads the public records from cBioPortal, constructs
the TCGA and SU2C groups, calculates the group comparisons and writes tidy CSV
files plus an HTML report.

```bash
python analysis.py
```

Outputs are written to `results/`:

- `expression_values.csv`
- `summary.csv`
- `report.html`
- `metadata.json`

The supplied R script removes decimal points from TCGA expression strings before
conversion. Because the thesis figure depends on that step, the Python output
keeps a `thesis_analysis_value` column for comparison. A separate
`log2_expression` column uses the direct transformation and is used for the
later modelling examples.

The reproduced sample counts are 304 comparison and 189 fusion-positive samples
for TCGA, and 85 comparison and 96 fusion-positive samples for SU2C.

## PyTorch extension

The extension tests whether RAI2, CTBP1 and CTBP2 expression can predict the
recorded fusion label. TCGA is split into training, validation and internal-test
sets; SU2C is used only as an external comparison. ERG is excluded from the main
model because it is a direct marker of the fusion and is included only in a
separate control model.

```bash
python analysis.py
python -m pip install -r requirements-ml.txt
python ml_extension.py
```

The main model showed weak discrimination in both the TCGA test set and SU2C.
The ERG control ranked samples well, but its probabilities did not transfer to
SU2C at the fixed threshold. I retained this result because it illustrates the
effect of cohort and processing differences rather than a successful clinical
classifier.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover grouping, the historical conversion, group statistics, data
splitting, training-only scaling and basic classification metrics. API responses
are cached locally, and `metadata.json` records the profiles used for each run.

Public data are accessed through the
[cBioPortal REST API](https://docs.cbioportal.org/web-api-and-clients/).
