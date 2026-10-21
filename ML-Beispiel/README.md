# TMPRSS2–ERG prediction example

This is an additional machine-learning experiment based on public TCGA
prostate-cancer data. The question is whether fusion status can be predicted
for a held-out TCGA patient from a small expression panel and basic clinical
variables.

I treat this as a methods example, not as a clinical model. The dataset contains
493 patients, so simple models are important comparison points for the neural
networks.

## Main choices

- The split is performed by patient and stratified by fusion status.
- `ERG`, `TMPRSS2` and the clinical `SUBTYPE` field are not used as predictors,
  because they would reveal or closely encode the target.
- Imputation, scaling and feature selection are fitted on training data only.
- Model settings and the classification threshold are chosen within the
  development set. The test set is used for the final comparison.
- The expression panel was fixed before the train/test split.

The models are a clinical logistic regression, an expression elastic net, an
expression-only PyTorch MLP and a small multimodal MLP. Results include ROC-AUC,
precision-recall AUC, balanced accuracy, calibration and bootstrap confidence
intervals.

## Files

```text
ML-Beispiel/
├── data/model_input.csv          # prepared public TCGA table
├── data/data_manifest.json       # source and sample counts
├── data_pipeline.py              # download and table preparation
├── gene_panel.txt                # fixed expression panel
├── modeling.py                   # model training and evaluation
├── ML_walkthrough.ipynb          # notebook version of the analysis
├── requirements.txt
├── requirements-notebook.txt
└── tests/
```

## Run the notebook

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name ml-beispiel --display-name "Python (ML-Beispiel)"
jupyter lab
```

Open `ML_walkthrough.ipynb`, select the `Python (ML-Beispiel)` kernel and run
the cells in order. The modelling table is included, so this step does not need
an internet connection.

The same workflow can be run from the terminal:

```bash
python data_pipeline.py
python modeling.py
```

For a shorter installation check:

```bash
python modeling.py --quick
```

The cBioPortal responses are cached in `.cache/`. Unit tests do not require the
ML dependencies:

```bash
python -m unittest discover -s tests -v
```

## Limits

The held-out result measures performance within TCGA. It does not show that the
model will transfer to another institution, assay or disease stage. SU2C would
therefore be a domain-shift check, not an interchangeable validation cohort.

Data source: [cBioPortal](https://www.cbioportal.org/), study
`prad_tcga_pan_can_atlas_2018`. The checked-in table contains processed public
research data and no direct identifiers.
