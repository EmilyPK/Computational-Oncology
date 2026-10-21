#!/usr/bin/env python3
"""Fit cohort-specific ridge models for RAI2 expression.

TCGA and SU2C are analysed separately. Nested cross-validation selects the
ridge penalty and produces one out-of-fold prediction per sample. The analysis
tests association and prediction; it does not establish a causal effect.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TCGA_STUDY_ID = "prad_tcga_pan_can_atlas_2018"
SU2C_STUDY_ID = "prad_su2c_2019"
COHORT_ORDER = (TCGA_STUDY_ID, SU2C_STUDY_ID)
GENES = ("ERG", "RAI2", "CTBP1", "CTBP2")
TARGET_GENE = "RAI2"
FUSION_FEATURE = "fusion_status"
ML_VALUE_COLUMN = "log2_expression"

# Compare a CtBP-only baseline, fusion status alone and the combined model.
MODEL_SPECS: dict[str, tuple[str, ...]] = {
    "ctbp_baseline": ("CTBP1", "CTBP2"),
    "fusion_only": (FUSION_FEATURE,),
    "fusion_ctbp_primary": (FUSION_FEATURE, "CTBP1", "CTBP2"),
}
MODEL_DESCRIPTIONS = {
    "ctbp_baseline": "CTBP1 and CTBP2 only; comparison model without fusion status.",
    "fusion_only": "TMPRSS2-ERG status only; a cross-validated expression-group contrast.",
    "fusion_ctbp_primary": "Primary model: fusion status, CTBP1 and CTBP2.",
}
DEFAULT_RIDGE_PENALTIES = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class RegressionSample:
    study: str
    study_id: str
    sample_id: str
    fusion_status: int
    expression_values: tuple[float, ...]

    def expression(self, gene: str) -> float:
        try:
            index = GENES.index(gene)
        except ValueError as exc:
            raise ValueError(f"Unknown gene: {gene}") from exc
        return self.expression_values[index]

    def predictor(self, name: str) -> float:
        if name == FUSION_FEATURE:
            return float(self.fusion_status)
        return self.expression(name)

    @property
    def target(self) -> float:
        return self.expression(TARGET_GENE)


@dataclass(frozen=True)
class FeatureStandardizer:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    def transform(self, sample: RegressionSample) -> list[float]:
        return [
            (sample.predictor(name) - mean) / standard_deviation
            for name, mean, standard_deviation in zip(
                self.feature_names, self.means, self.standard_deviations
            )
        ]


@dataclass(frozen=True)
class TargetStandardizer:
    mean: float
    standard_deviation: float

    def transform(self, value: float) -> float:
        return (float(value) - self.mean) / self.standard_deviation

    def inverse_transform(self, value: float) -> float:
        return float(value) * self.standard_deviation + self.mean


@dataclass(frozen=True)
class RidgeModel:
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    ridge_penalty: float
    feature_standardizer: FeatureStandardizer
    target_standardizer: TargetStandardizer

    def predict_standardized(self, sample: RegressionSample) -> float:
        transformed = self.feature_standardizer.transform(sample)
        return self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, transformed)
        )

    def predict(self, sample: RegressionSample) -> float:
        return self.target_standardizer.inverse_transform(
            self.predict_standardized(sample)
        )


def load_regression_samples(
    csv_path: Path,
    value_column: str = ML_VALUE_COLUMN,
) -> list[RegressionSample]:
    """Pivot tidy expression rows to complete sample-level regression records."""
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "study",
            "study_id",
            "sample_id",
            "gene",
            "fusion_status",
            value_column,
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
        for row in reader:
            gene = str(row["gene"])
            if gene not in GENES:
                continue
            key = (str(row["study_id"]), str(row["sample_id"]))
            record = records.setdefault(
                key,
                {
                    "study": str(row["study"]),
                    "fusion_status": str(row["fusion_status"]),
                    "expressions": {},
                },
            )
            if record["fusion_status"] != str(row["fusion_status"]):
                raise ValueError(f"Conflicting fusion labels for sample {key[1]}")
            if gene in record["expressions"]:
                raise ValueError(f"Duplicate {gene} value for sample {key[1]}")
            record["expressions"][gene] = float(row[value_column])

    incomplete = [
        sample_id
        for (_, sample_id), record in records.items()
        if set(record["expressions"]) != set(GENES)
    ]
    if incomplete:
        preview = ", ".join(sorted(incomplete)[:5])
        raise ValueError(f"Samples with incomplete expression vectors: {preview}")

    samples = []
    for (study_id, sample_id), record in sorted(records.items()):
        fusion_status = str(record["fusion_status"])
        if fusion_status not in {"TMPRSS2-ERG detected", "not detected"}:
            raise ValueError(f"Unexpected fusion label: {fusion_status}")
        samples.append(
            RegressionSample(
                study=str(record["study"]),
                study_id=study_id,
                sample_id=sample_id,
                fusion_status=int(fusion_status.startswith("TMPRSS2")),
                expression_values=tuple(
                    float(record["expressions"][gene]) for gene in GENES
                ),
            )
        )
    if not samples:
        raise ValueError(f"No complete samples found in {csv_path}")
    return samples


def stratified_kfold(
    samples: Sequence[RegressionSample],
    number_of_folds: int,
    seed: int,
) -> list[list[RegressionSample]]:
    """Create deterministic folds stratified by TMPRSS2--ERG status."""
    if number_of_folds < 2:
        raise ValueError("number_of_folds must be at least two")
    by_label: dict[int, list[RegressionSample]] = {0: [], 1: []}
    for sample in samples:
        if sample.fusion_status not in by_label:
            raise ValueError("Fusion status must be binary")
        by_label[sample.fusion_status].append(sample)
    if any(len(group) < number_of_folds for group in by_label.values()):
        raise ValueError("Each fusion group must contain at least one sample per fold")

    folds: list[list[RegressionSample]] = [[] for _ in range(number_of_folds)]
    for label, group in sorted(by_label.items()):
        ordered = sorted(group, key=lambda sample: sample.sample_id)
        random.Random(seed + label).shuffle(ordered)
        for index, sample in enumerate(ordered):
            folds[index % number_of_folds].append(sample)
    for fold in folds:
        fold.sort(key=lambda sample: sample.sample_id)
    return folds


def _population_mean_and_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot standardize an empty sequence")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, max(math.sqrt(variance), 1e-12)


def fit_feature_standardizer(
    samples: Sequence[RegressionSample],
    feature_names: Sequence[str],
) -> FeatureStandardizer:
    """Fit expression scaling on a training fold; keep fusion as an indicator."""
    if not samples:
        raise ValueError("Cannot fit a feature standardizer without samples")
    names = tuple(feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be non-empty and unique")
    means = []
    standard_deviations = []
    for name in names:
        if name == TARGET_GENE:
            raise ValueError("The RAI2 target must not also be used as a predictor")
        if name == FUSION_FEATURE:
            # A 0/1 coefficient then represents the adjusted change in target
            # standard deviations associated with fusion-positive status.
            means.append(0.0)
            standard_deviations.append(1.0)
            continue
        if name not in GENES:
            raise ValueError(f"Unknown predictor: {name}")
        mean, standard_deviation = _population_mean_and_sd(
            [sample.expression(name) for sample in samples]
        )
        means.append(mean)
        standard_deviations.append(standard_deviation)
    return FeatureStandardizer(names, tuple(means), tuple(standard_deviations))


def fit_target_standardizer(
    samples: Sequence[RegressionSample],
) -> TargetStandardizer:
    mean, standard_deviation = _population_mean_and_sd(
        [sample.target for sample in samples]
    )
    return TargetStandardizer(mean, standard_deviation)


def regression_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float | int]:
    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed and predicted must have the same non-zero length")
    observed_values = [float(value) for value in observed]
    predicted_values = [float(value) for value in predicted]
    residuals = [
        actual - estimate
        for actual, estimate in zip(observed_values, predicted_values)
    ]
    squared_error = sum(residual * residual for residual in residuals)
    observed_mean = sum(observed_values) / len(observed_values)
    predicted_mean = sum(predicted_values) / len(predicted_values)
    total_sum_of_squares = sum(
        (value - observed_mean) ** 2 for value in observed_values
    )
    covariance = sum(
        (actual - observed_mean) * (estimate - predicted_mean)
        for actual, estimate in zip(observed_values, predicted_values)
    )
    correlation_denominator = math.sqrt(
        sum((value - observed_mean) ** 2 for value in observed_values)
        * sum((value - predicted_mean) ** 2 for value in predicted_values)
    )
    return {
        "n": len(observed_values),
        "r_squared": (
            1.0 - squared_error / total_sum_of_squares
            if total_sum_of_squares
            else 0.0
        ),
        "rmse": math.sqrt(squared_error / len(observed_values)),
        "mae": sum(abs(residual) for residual in residuals) / len(residuals),
        "pearson_r": (
            covariance / correlation_denominator
            if correlation_denominator
            else 0.0
        ),
    }


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for the regression extension. Install a "
            "PyTorch-supported Python version and then run "
            "`python -m pip install -r requirements-ml.txt`."
        ) from exc
    return torch


def fit_ridge_model(
    train_samples: Sequence[RegressionSample],
    feature_names: Sequence[str],
    ridge_penalty: float,
) -> RidgeModel:
    """Fit an interpretable linear ridge regression with PyTorch tensors."""
    if ridge_penalty < 0.0:
        raise ValueError("ridge_penalty cannot be negative")
    torch = _require_torch()
    feature_standardizer = fit_feature_standardizer(train_samples, feature_names)
    target_standardizer = fit_target_standardizer(train_samples)
    transformed = [
        feature_standardizer.transform(sample) for sample in train_samples
    ]
    design = torch.tensor(
        [[1.0, *row] for row in transformed], dtype=torch.float64
    )
    target = torch.tensor(
        [target_standardizer.transform(sample.target) for sample in train_samples],
        dtype=torch.float64,
    )
    system = design.T @ design
    penalty = torch.eye(system.shape[0], dtype=torch.float64) * ridge_penalty
    penalty[0, 0] = 0.0
    right_hand_side = design.T @ target
    try:
        solution = torch.linalg.solve(system + penalty, right_hand_side)
    except RuntimeError:
        solution = torch.linalg.pinv(system + penalty) @ right_hand_side
    values = [float(value) for value in solution.tolist()]
    return RidgeModel(
        feature_names=tuple(feature_names),
        intercept=values[0],
        coefficients=tuple(values[1:]),
        ridge_penalty=float(ridge_penalty),
        feature_standardizer=feature_standardizer,
        target_standardizer=target_standardizer,
    )


def select_ridge_penalty(
    samples: Sequence[RegressionSample],
    feature_names: Sequence[str],
    penalties: Sequence[float],
    *,
    number_of_folds: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    """Select the penalty by inner-fold RMSE without using an outer test fold."""
    if not penalties:
        raise ValueError("At least one ridge penalty is required")
    folds = stratified_kfold(samples, number_of_folds, seed)
    scores = []
    for penalty in penalties:
        squared_errors = []
        for validation_index, validation_samples in enumerate(folds):
            training_samples = [
                sample
                for fold_index, fold in enumerate(folds)
                if fold_index != validation_index
                for sample in fold
            ]
            model = fit_ridge_model(training_samples, feature_names, float(penalty))
            squared_errors.extend(
                (sample.target - model.predict(sample)) ** 2
                for sample in validation_samples
            )
        scores.append(
            {
                "ridge_penalty": float(penalty),
                "inner_rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
            }
        )
    selected = min(
        scores,
        key=lambda row: (row["inner_rmse"], row["ridge_penalty"]),
    )["ridge_penalty"]
    return float(selected), scores


def nested_cross_validated_regression(
    samples: Sequence[RegressionSample],
    feature_names: Sequence[str],
    penalties: Sequence[float] = DEFAULT_RIDGE_PENALTIES,
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Return outer-fold predictions and metrics from nested cross-validation."""
    outer = stratified_kfold(samples, outer_folds, seed)
    predictions = []
    coefficient_rows = []
    selected_penalties = []
    for outer_index, test_samples in enumerate(outer):
        training_samples = [
            sample
            for fold_index, fold in enumerate(outer)
            if fold_index != outer_index
            for sample in fold
        ]
        selected_penalty, inner_scores = select_ridge_penalty(
            training_samples,
            feature_names,
            penalties,
            number_of_folds=inner_folds,
            seed=seed + 100 + outer_index,
        )
        model = fit_ridge_model(
            training_samples, feature_names, selected_penalty
        )
        selected_penalties.append(selected_penalty)
        coefficient_rows.append(
            {
                "outer_fold": outer_index + 1,
                "intercept": model.intercept,
                "coefficients": dict(zip(model.feature_names, model.coefficients)),
                "selected_penalty": selected_penalty,
                "inner_scores": inner_scores,
            }
        )
        for sample in test_samples:
            estimate = model.predict(sample)
            predictions.append(
                {
                    "outer_fold": outer_index + 1,
                    "study": sample.study,
                    "study_id": sample.study_id,
                    "sample_id": sample.sample_id,
                    "fusion_status": sample.fusion_status,
                    "observed_rai2": sample.target,
                    "predicted_rai2": estimate,
                    "residual": sample.target - estimate,
                }
            )

    predictions.sort(key=lambda row: str(row["sample_id"]))
    metrics = regression_metrics(
        [float(row["observed_rai2"]) for row in predictions],
        [float(row["predicted_rai2"]) for row in predictions],
    )
    coefficient_summary = {}
    for name in feature_names:
        values = [
            float(row["coefficients"][name]) for row in coefficient_rows
        ]
        coefficient_summary[name] = {
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "features": list(feature_names),
        "target": TARGET_GENE,
        "metrics": metrics,
        "coefficient_summary": coefficient_summary,
        "outer_fold_models": coefficient_rows,
        "selected_penalty_counts": {
            str(penalty): count
            for penalty, count in sorted(Counter(selected_penalties).items())
        },
        "predictions": predictions,
    }


def _svg_coordinate(
    value: float,
    lower: float,
    upper: float,
    start: float,
    end: float,
) -> float:
    span = max(upper - lower, 1e-12)
    return start + (value - lower) / span * (end - start)


def render_prediction_plot(
    path: Path,
    cohort_results: dict[str, Any],
) -> None:
    """Render observed versus out-of-fold RAI2 for the primary model."""
    width, height = 760, 350
    panel_width = 315
    panel_top, panel_bottom = 45, 300
    panel_starts = (70, 430)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Observed and predicted RAI2 expression">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="24" y="22" font-family="sans-serif" font-size="14" font-weight="700">Out-of-fold RAI2 predictions</text>',
    ]
    for panel_index, study_id in enumerate(COHORT_ORDER):
        result = cohort_results[study_id]["models"]["fusion_ctbp_primary"]
        rows = result["predictions"]
        values = [
            float(row[key])
            for row in rows
            for key in ("observed_rai2", "predicted_rai2")
        ]
        lower, upper = min(values), max(values)
        padding = max((upper - lower) * 0.05, 0.1)
        lower -= padding
        upper += padding
        left = panel_starts[panel_index]
        right = left + panel_width
        pieces.extend(
            [
                f'<rect x="{left}" y="{panel_top}" width="{panel_width}" height="{panel_bottom - panel_top}" fill="#f8fbfc" stroke="#d8e2e8"/>',
                f'<line x1="{left}" y1="{panel_bottom}" x2="{right}" y2="{panel_top}" stroke="#9db0bc" stroke-dasharray="6 5"/>',
            ]
        )
        for row in rows:
            x = _svg_coordinate(
                float(row["observed_rai2"]), lower, upper, left, right
            )
            y = _svg_coordinate(
                float(row["predicted_rai2"]), lower, upper, panel_bottom, panel_top
            )
            color = "#d45b5b" if int(row["fusion_status"]) else "#4f91b5"
            pieces.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{color}" fill-opacity="0.62"/>'
            )
        label = str(cohort_results[study_id]["study"])
        r_squared = float(result["metrics"]["r_squared"])
        pieces.extend(
            [
                f'<text x="{left + panel_width / 2}" y="38" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700">{html.escape(label)}</text>',
                f'<text x="{left + 8}" y="{panel_top + 17}" font-family="sans-serif" font-size="10">outer-CV R² = {r_squared:.3f}</text>',
                f'<text x="{left + panel_width / 2}" y="329" text-anchor="middle" font-family="sans-serif" font-size="10">Observed log2 RAI2 expression</text>',
            ]
        )
    pieces.append(
        '<text x="15" y="176" text-anchor="middle" transform="rotate(-90 15 176)" font-family="sans-serif" font-size="10">Predicted log2 RAI2 expression</text>'
    )
    pieces.append("</svg>")
    path.write_text("".join(pieces), encoding="utf-8")


def render_coefficient_plot(
    path: Path,
    cohort_results: dict[str, Any],
) -> None:
    """Compare primary-model coefficients on the standardized scale."""
    features = MODEL_SPECS["fusion_ctbp_primary"]
    colors = {TCGA_STUDY_ID: "#145da0", SU2C_STUDY_ID: "#d45b5b"}
    values = [
        float(
            cohort_results[study_id]["models"]["fusion_ctbp_primary"]
            ["coefficient_summary"][feature]["mean"]
        )
        for study_id in COHORT_ORDER
        for feature in features
    ]
    maximum = max(max(abs(value) for value in values) * 1.25, 0.25)
    zero_x = 365.0
    scale = 270.0 / maximum
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 330" role="img" aria-label="Primary-model coefficient comparison">',
        '<rect width="760" height="330" fill="white"/>',
        '<text x="24" y="23" font-family="sans-serif" font-size="14" font-weight="700">Primary-model coefficients across outer folds</text>',
        '<line x1="365" y1="42" x2="365" y2="280" stroke="#7f929e"/>',
    ]
    labels = {
        FUSION_FEATURE: "TMPRSS2-ERG status",
        "CTBP1": "CTBP1 expression",
        "CTBP2": "CTBP2 expression",
    }
    for feature_index, feature in enumerate(features):
        base_y = 82 + feature_index * 72
        pieces.append(
            f'<text x="190" y="{base_y + 5}" text-anchor="end" font-family="sans-serif" font-size="11">{html.escape(labels[feature])}</text>'
        )
        for cohort_index, study_id in enumerate(COHORT_ORDER):
            summary = (
                cohort_results[study_id]["models"]["fusion_ctbp_primary"]
                ["coefficient_summary"][feature]
            )
            value = float(summary["mean"])
            minimum = float(summary["minimum"])
            maximum_value = float(summary["maximum"])
            y = base_y - 9 + cohort_index * 20
            x_value = zero_x + value * scale
            x_minimum = zero_x + minimum * scale
            x_maximum = zero_x + maximum_value * scale
            color = colors[study_id]
            pieces.extend(
                [
                    f'<line x1="{x_minimum:.2f}" y1="{y}" x2="{x_maximum:.2f}" y2="{y}" stroke="{color}" stroke-width="2"/>',
                    f'<circle cx="{x_value:.2f}" cy="{y}" r="4" fill="{color}"/>',
                ]
            )
    pieces.extend(
        [
            '<text x="365" y="307" text-anchor="middle" font-family="sans-serif" font-size="10">Standardized coefficient (point: mean; line: outer-fold range)</text>',
            '<text x="545" y="45" fill="#145da0" font-family="sans-serif" font-size="10">TCGA</text>',
            '<text x="600" y="45" fill="#d45b5b" font-family="sans-serif" font-size="10">SU2C</text>',
            '</svg>',
        ]
    )
    path.write_text("".join(pieces), encoding="utf-8")


def render_html_report(
    path: Path,
    cohort_results: dict[str, Any],
    generated_at: str,
) -> None:
    sections = []
    for study_id in COHORT_ORDER:
        cohort = cohort_results[study_id]
        rows = []
        for model_name in MODEL_SPECS:
            result = cohort["models"][model_name]
            metrics = result["metrics"]
            rows.append(
                "<tr>"
                f"<td>{html.escape(model_name.replace('_', ' '))}</td>"
                f"<td>{', '.join(map(html.escape, result['features']))}</td>"
                f"<td>{metrics['n']}</td>"
                f"<td>{metrics['r_squared']:.3f}</td>"
                f"<td>{metrics['rmse']:.3f}</td>"
                f"<td>{metrics['mae']:.3f}</td>"
                f"<td>{metrics['pearson_r']:.3f}</td>"
                "</tr>"
            )
        baseline_r2 = float(
            cohort["models"]["ctbp_baseline"]["metrics"]["r_squared"]
        )
        primary_r2 = float(
            cohort["models"]["fusion_ctbp_primary"]["metrics"]["r_squared"]
        )
        delta = primary_r2 - baseline_r2
        sections.append(
            f"""<section><div class="eyebrow">Independent cohort analysis</div>
<h2>{html.escape(str(cohort['study']))}</h2>
<p>Samples: {cohort['sample_count']} ({cohort['fusion_positive']} fusion-positive, {cohort['fusion_negative']} comparison). Adding fusion status to CTBP1/CTBP2 changed outer-CV R² by <strong>{delta:+.3f}</strong>.</p>
<table><thead><tr><th>Model</th><th>Predictors</th><th>n</th><th>R²</th><th>RMSE</th><th>MAE</th><th>Pearson r</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"""
        )

    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cohort-aware RAI2 regression</title><style>
:root{{--ink:#12263a;--muted:#607181;--blue:#145da0;--line:#d8e2e8}}*{{box-sizing:border-box}}body{{margin:0;background:#f7fafb;color:var(--ink);font:15px/1.55 system-ui,sans-serif}}header,main,footer{{width:min(1050px,calc(100% - 40px));margin:auto}}header{{padding:56px 0 28px}}h1{{font-size:clamp(34px,6vw,60px);line-height:1.04;margin:.2em 0}}header p,section p,footer{{color:var(--muted)}}aside{{padding:16px 20px;background:#fff4dc;border:1px solid #e7c878;border-radius:12px;margin-bottom:20px}}section{{margin:18px 0;padding:24px;background:white;border:1px solid var(--line);border-radius:14px}}.eyebrow{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}}h2{{margin:.2em 0}}table{{width:100%;border-collapse:collapse;margin-top:18px}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}.figures{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:28px 0}}.figures img{{width:100%;background:white;border:1px solid var(--line);border-radius:14px}}footer{{padding:28px 0 60px;border-top:1px solid var(--line)}}@media(max-width:780px){{.figures{{grid-template-columns:1fr}}table{{font-size:12px}}}}
</style></head><body><header><div class="eyebrow">PyTorch extension · revised biological question</div><h1>Explaining RAI2 expression within each cohort</h1><p>The target is continuous corrected log2 RAI2 expression. The primary predictors are TMPRSS2-ERG status, CTBP1 and CTBP2. TCGA and SU2C are modelled separately with nested stratified cross-validation and training-fold-only scaling.</p></header>
<main><aside><strong>Why a separate analysis was created:</strong> the earlier TCGA-trained fusion classifier classified every SU2C sample as negative at the 0.5 threshold. TCGA and SU2C differ in tumour stage, patient composition, laboratory workflow and expression processing. This analysis therefore treats them as independent cohorts rather than assuming a shared probability scale.</aside>
{''.join(sections)}
<section><div class="eyebrow">Interpretation</div><h2>What this model can and cannot show</h2><p>The outer-fold metrics quantify how much sample-to-sample RAI2 variation can be predicted within each cohort. The difference between the CTBP-only baseline and the fusion-plus-CTBP primary model estimates the incremental predictive information associated with fusion status. Agreement across cohorts strengthens evidence for a reproducible association; disagreement is itself informative about cohort context. These observational models do not establish causal regulation and are not clinical predictors.</p></section>
<div class="figures"><img src="rai2_predictions.svg" alt="Observed versus cross-validated RAI2 predictions"><img src="coefficient_comparison.svg" alt="Primary-model coefficient comparison"></div></main>
<footer>Generated {html.escape(generated_at)}. Ridge penalties were selected inside each outer training fold; all reported predictions are out of fold.</footer></body></html>"""
    path.write_text(document, encoding="utf-8")


def run_experiments(
    expression_csv: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    outer_folds: int = 5,
    inner_folds: int = 4,
    ridge_penalties: Sequence[float] = DEFAULT_RIDGE_PENALTIES,
) -> dict[str, Any]:
    """Run the cohort-specific nested-CV analyses and write reusable outputs."""
    torch = _require_torch()
    samples = load_regression_samples(expression_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    cohort_results: dict[str, Any] = {}
    prediction_rows = []

    for study_id in COHORT_ORDER:
        cohort_samples = [sample for sample in samples if sample.study_id == study_id]
        if not cohort_samples:
            raise ValueError(f"No samples found for cohort {study_id}")
        model_results = {}
        for model_name, features in MODEL_SPECS.items():
            result = nested_cross_validated_regression(
                cohort_samples,
                features,
                ridge_penalties,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                seed=seed,
            )
            final_penalty, full_inner_scores = select_ridge_penalty(
                cohort_samples,
                features,
                ridge_penalties,
                number_of_folds=outer_folds,
                seed=seed + 1000,
            )
            final_model = fit_ridge_model(cohort_samples, features, final_penalty)
            checkpoint = {
                "model_type": "ridge_linear_regression",
                "target": TARGET_GENE,
                "features": list(features),
                "study_id": study_id,
                "input_value_column": ML_VALUE_COLUMN,
                "ridge_penalty": final_penalty,
                "intercept": final_model.intercept,
                "coefficients": dict(
                    zip(final_model.feature_names, final_model.coefficients)
                ),
                "feature_standardizer": asdict(final_model.feature_standardizer),
                "target_standardizer": asdict(final_model.target_standardizer),
            }
            torch.save(
                checkpoint,
                output_dir / f"{study_id}_{model_name}_ridge.pt",
            )
            result["description"] = MODEL_DESCRIPTIONS[model_name]
            result["final_model"] = checkpoint
            result["full_data_penalty_selection"] = full_inner_scores
            for row in result.pop("predictions"):
                prediction_rows.append({"model": model_name, **row})
            # Keep predictions in metadata for rendering, then remove them
            # before JSON serialization to avoid duplicating predictions.csv.
            result["predictions"] = [
                row for row in prediction_rows
                if row["model"] == model_name and row["study_id"] == study_id
            ]
            model_results[model_name] = result

        cohort_results[study_id] = {
            "study": cohort_samples[0].study,
            "study_id": study_id,
            "sample_count": len(cohort_samples),
            "fusion_positive": sum(sample.fusion_status for sample in cohort_samples),
            "fusion_negative": len(cohort_samples)
            - sum(sample.fusion_status for sample in cohort_samples),
            "models": model_results,
        }

    render_prediction_plot(output_dir / "rai2_predictions.svg", cohort_results)
    render_coefficient_plot(
        output_dir / "coefficient_comparison.svg", cohort_results
    )
    render_html_report(output_dir / "report.html", cohort_results, generated_at)

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "model",
            "outer_fold",
            "study",
            "study_id",
            "sample_id",
            "fusion_status",
            "observed_rai2",
            "predicted_rai2",
            "residual",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)

    serializable_cohorts = {}
    for study_id, cohort in cohort_results.items():
        serializable_models = {}
        for model_name, result in cohort["models"].items():
            serializable_models[model_name] = {
                key: value for key, value in result.items() if key != "predictions"
            }
        serializable_cohorts[study_id] = {
            **{key: value for key, value in cohort.items() if key != "models"},
            "models": serializable_models,
        }

    metadata = {
        "generated_at": generated_at,
        "pytorch_version": torch.__version__,
        "question": (
            "How much out-of-fold information about continuous RAI2 expression "
            "is provided by TMPRSS2-ERG status and CTBP1/CTBP2 within each cohort?"
        ),
        "target": TARGET_GENE,
        "input_value_column": ML_VALUE_COLUMN,
        "cohort_strategy": "TCGA and SU2C modelled independently",
        "evaluation": "nested stratified cross-validation",
        "configuration": {
            "seed": seed,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "ridge_penalties": [float(value) for value in ridge_penalties],
        },
        "cohorts": serializable_cohorts,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expression-csv",
        type=Path,
        default=Path("data/expression_values.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    metadata = run_experiments(
        arguments.expression_csv,
        arguments.output_dir,
        seed=arguments.seed,
        outer_folds=arguments.outer_folds,
        inner_folds=arguments.inner_folds,
    )
    for study_id, cohort in metadata["cohorts"].items():
        primary = cohort["models"]["fusion_ctbp_primary"]["metrics"]
        baseline = cohort["models"]["ctbp_baseline"]["metrics"]
        print(
            f"{study_id}: primary outer-CV R2={primary['r_squared']:.3f}; "
            f"delta versus CTBP baseline="
            f"{primary['r_squared'] - baseline['r_squared']:+.3f}"
        )
    print(f"Report: {arguments.output_dir / 'report.html'}")
