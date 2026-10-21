#!/usr/bin/env python3
"""Run the PyTorch experiment based on the cohort-analysis table.

The main model uses RAI2, CTBP1 and CTBP2. ERG is left out because it is a
direct marker of the fusion; an ERG-inclusive model is kept as a control. TCGA
is used for development and internal testing, while SU2C is evaluated only as
an external cohort.
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


TRAIN_STUDY_ID = "prad_tcga_pan_can_atlas_2018"
EXTERNAL_STUDY_ID = "prad_su2c_2019"
ML_VALUE_COLUMN = "log2_expression"
PRIMARY_FEATURES = ("RAI2", "CTBP1", "CTBP2")
MARKER_INCLUSIVE_FEATURES = ("ERG", "RAI2", "CTBP1", "CTBP2")
FEATURE_SETS = {
    "primary": PRIMARY_FEATURES,
    "marker_inclusive_control": MARKER_INCLUSIVE_FEATURES,
}


@dataclass(frozen=True)
class Sample:
    study: str
    study_id: str
    sample_id: str
    label: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class Standardizer:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    def transform(self, values: Sequence[float]) -> list[float]:
        if len(values) != len(self.feature_names):
            raise ValueError("Feature vector has the wrong length")
        return [
            (float(value) - mean) / standard_deviation
            for value, mean, standard_deviation in zip(
                values, self.means, self.standard_deviations
            )
        ]


def load_samples(
    csv_path: Path,
    feature_names: Sequence[str],
    value_column: str = ML_VALUE_COLUMN,
) -> list[Sample]:
    """Pivot corrected expression rows into one feature vector per sample."""
    requested = tuple(feature_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("feature_names must contain unique gene symbols")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if value_column not in (reader.fieldnames or ()):
            raise ValueError(f"Missing ML value column: {value_column}")
        for row in reader:
            gene = str(row["gene"])
            if gene not in requested:
                continue
            key = (str(row["study_id"]), str(row["sample_id"]))
            record = records.setdefault(
                key,
                {
                    "study": str(row["study"]),
                    "status": str(row["fusion_status"]),
                    "values": {},
                },
            )
            if record["status"] != str(row["fusion_status"]):
                raise ValueError(f"Conflicting labels for sample {key[1]}")
            if gene in record["values"]:
                raise ValueError(f"Duplicate {gene} value for sample {key[1]}")
            record["values"][gene] = float(row[value_column])

    incomplete = [
        sample_id
        for (_, sample_id), record in records.items()
        if set(record["values"]) != set(requested)
    ]
    if incomplete:
        preview = ", ".join(sorted(incomplete)[:5])
        raise ValueError(f"Samples with incomplete feature vectors: {preview}")

    samples = []
    for (study_id, sample_id), record in sorted(records.items()):
        status = str(record["status"])
        if status not in {"TMPRSS2-ERG detected", "not detected"}:
            raise ValueError(f"Unexpected fusion label: {status}")
        samples.append(
            Sample(
                study=str(record["study"]),
                study_id=study_id,
                sample_id=sample_id,
                label=int(status.startswith("TMPRSS2")),
                values=tuple(float(record["values"][gene]) for gene in requested),
            )
        )
    if not samples:
        raise ValueError(f"No complete samples found in {csv_path}")
    return samples


def stratified_split(
    samples: Sequence[Sample], validation_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Make a deterministic split while preserving both classes."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    by_label: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        if sample.label not in by_label:
            raise ValueError(f"Expected a binary label, received {sample.label}")
        by_label[sample.label].append(sample)
    if any(len(group) < 2 for group in by_label.values()):
        raise ValueError("Each class needs at least two samples for a stratified split")

    rng = random.Random(seed)
    train: list[Sample] = []
    validation: list[Sample] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda sample: sample.sample_id)
        rng.shuffle(group)
        validation_count = min(
            len(group) - 1,
            max(1, round(len(group) * validation_fraction)),
        )
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def stratified_development_split(
    samples: Sequence[Sample],
    validation_fraction: float,
    internal_test_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Create disjoint training, early-stopping and untouched test subsets.

    Both fractions refer to the complete development cohort. The second split
    therefore converts the requested validation fraction to a fraction of the
    samples remaining after the internal test set has been reserved.
    """
    if validation_fraction <= 0.0 or internal_test_fraction <= 0.0:
        raise ValueError("Validation and internal-test fractions must be positive")
    if validation_fraction + internal_test_fraction >= 1.0:
        raise ValueError("Validation and internal-test fractions must sum to less than one")
    training_pool, internal_test = stratified_split(
        samples, internal_test_fraction, seed
    )
    relative_validation_fraction = validation_fraction / (1.0 - internal_test_fraction)
    train, validation = stratified_split(
        training_pool, relative_validation_fraction, seed + 1
    )
    return train, validation, internal_test


def class_counts(samples: Sequence[Sample]) -> dict[str, int]:
    positives = sum(sample.label for sample in samples)
    return {
        "fusion_positive": positives,
        "fusion_negative": len(samples) - positives,
    }


def fit_standardizer(samples: Sequence[Sample], feature_names: Sequence[str]) -> Standardizer:
    """Fit z-score parameters using training samples only."""
    if not samples:
        raise ValueError("Cannot fit a standardizer without samples")
    feature_names = tuple(feature_names)
    width = len(feature_names)
    if any(len(sample.values) != width for sample in samples):
        raise ValueError("Feature vector width does not match feature_names")
    means = tuple(
        sum(sample.values[index] for sample in samples) / len(samples)
        for index in range(width)
    )
    standard_deviations = []
    for index, mean in enumerate(means):
        variance = sum(
            (sample.values[index] - mean) ** 2 for sample in samples
        ) / len(samples)
        standard_deviations.append(max(math.sqrt(variance), 1e-12))
    return Standardizer(feature_names, means, tuple(standard_deviations))


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Calculate ROC-AUC from average probability ranks, including ties."""
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires observations from both classes")
    ranks = average_ranks([float(value) for value in probabilities])
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5
) -> dict[str, float | int]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    predictions = [int(probability >= threshold) for probability in probabilities]
    true_positive = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    true_negative = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    false_positive = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    false_negative = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision_denominator = true_positive + false_positive
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = sensitivity
    f1_denominator = precision + recall
    return {
        "n": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "accuracy": (true_positive + true_negative) / len(labels),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0,
        "roc_auc": roc_auc(labels, probabilities),
        "brier_score": sum(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        )
        / len(labels),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def roc_points(
    labels: Sequence[int], probabilities: Sequence[float]
) -> list[tuple[float, float]]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC curve requires observations from both classes")
    ordered = sorted(zip(probabilities, labels), reverse=True)
    points = [(0.0, 0.0)]
    true_positive = false_positive = 0
    position = 0
    while position < len(ordered):
        threshold = ordered[position][0]
        while position < len(ordered) and ordered[position][0] == threshold:
            if ordered[position][1] == 1:
                true_positive += 1
            else:
                false_positive += 1
            position += 1
        points.append((false_positive / negatives, true_positive / positives))
    return points


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for the ML extension. Install it with "
            "`python -m pip install -r requirements-ml.txt`."
        ) from exc
    return torch


def _build_model(torch: Any, input_features: int, hidden_units: int, dropout: float) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(input_features, hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_units, 1),
    )


def _as_tensors(
    torch: Any, samples: Sequence[Sample], standardizer: Standardizer
) -> tuple[Any, Any]:
    features = torch.tensor(
        [standardizer.transform(sample.values) for sample in samples],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[float(sample.label)] for sample in samples], dtype=torch.float32
    )
    return features, labels


def _predict_probabilities(
    torch: Any, model: Any, samples: Sequence[Sample], standardizer: Standardizer
) -> list[float]:
    features, _ = _as_tensors(torch, samples, standardizer)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(features)).flatten().tolist()


def train_model(
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    standardizer: Standardizer,
    *,
    seed: int,
    hidden_units: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
) -> tuple[Any, list[dict[str, float | int]], int]:
    """Train an MLP with class weighting and validation-loss early stopping."""
    torch = _require_torch()
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    train_features, train_labels = _as_tensors(torch, train_samples, standardizer)
    validation_features, validation_labels = _as_tensors(
        torch, validation_samples, standardizer
    )
    model = _build_model(
        torch, len(standardizer.feature_names), hidden_units, dropout
    )
    positives = sum(sample.label for sample in train_samples)
    negatives = len(train_samples) - positives
    positive_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        observations = 0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_features)
            observations += len(batch_features)

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                criterion(model(validation_features), validation_labels).item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / observations,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def _line_path(
    points: Iterable[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: float = 640.0,
    height: float = 300.0,
) -> str:
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    coordinates = []
    for x_value, y_value in points:
        x = 55.0 + (x_value - x_min) / x_span * (width - 75.0)
        y = 20.0 + (y_max - y_value) / y_span * (height - 60.0)
        coordinates.append(f"{x:.2f},{y:.2f}")
    return " ".join(coordinates)


def render_training_curve(
    path: Path, histories: dict[str, list[dict[str, float | int]]]
) -> None:
    all_losses = [
        float(row[key])
        for history in histories.values()
        for row in history
        for key in ("train_loss", "validation_loss")
    ]
    maximum_epoch = max(len(history) for history in histories.values())
    maximum_loss = max(all_losses)
    colors = {
        "primary": ("#145da0", "#33b8c4"),
        "marker_inclusive_control": ("#783f8e", "#d67ab1"),
    }
    lines = []
    for name, history in histories.items():
        train_color, validation_color = colors[name]
        for key, color, dash in (
            ("train_loss", train_color, ""),
            ("validation_loss", validation_color, ' stroke-dasharray="7 5"'),
        ):
            points = [(float(row["epoch"]), float(row[key])) for row in history]
            lines.append(
                f'<polyline points="{_line_path(points, 1, maximum_epoch, 0, maximum_loss)}" '
                f'fill="none" stroke="{color}" stroke-width="2.4"{dash}/>'
            )
    document = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 340" role="img" aria-label="Training and validation loss">
<rect width="640" height="340" fill="white"/>
<text x="55" y="16" font-family="sans-serif" font-size="13" font-weight="700">Training and validation loss</text>
<line x1="55" y1="300" x2="620" y2="300" stroke="#9db0bc"/><line x1="55" y1="20" x2="55" y2="300" stroke="#9db0bc"/>
{''.join(lines)}
<text x="337" y="330" text-anchor="middle" font-family="sans-serif" font-size="11">Epoch</text>
<text x="15" y="160" text-anchor="middle" transform="rotate(-90 15 160)" font-family="sans-serif" font-size="11">Weighted BCE loss</text>
<g font-family="sans-serif" font-size="10"><text x="400" y="18" fill="#145da0">primary train</text><text x="480" y="18" fill="#33b8c4">primary validation</text><text x="400" y="32" fill="#783f8e">control train</text><text x="480" y="32" fill="#d67ab1">control validation</text></g>
</svg>"""
    path.write_text(document, encoding="utf-8")


def render_roc_curves(
    path: Path,
    curves: dict[str, dict[str, Any]],
) -> None:
    colors = {
        "primary_internal_test": "#145da0",
        "primary_external": "#33b8c4",
        "marker_inclusive_control_internal_test": "#783f8e",
        "marker_inclusive_control_external": "#d67ab1",
    }
    lines = []
    legend = []
    for index, (name, curve) in enumerate(curves.items()):
        color = colors[name]
        points = _line_path(curve["points"], 0, 1, 0, 1)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        label = name.replace("marker_inclusive_control", "control").replace("_", " ")
        legend.append(
            f'<text x="365" y="{48 + index * 16}" fill="{color}" font-family="sans-serif" font-size="10">{html.escape(label)} · AUC {curve["auc"]:.3f}</text>'
        )
    diagonal = _line_path([(0, 0), (1, 1)], 0, 1, 0, 1)
    document = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 340" role="img" aria-label="ROC curves">
<rect width="640" height="340" fill="white"/>
<text x="55" y="16" font-family="sans-serif" font-size="13" font-weight="700">Internal and external ROC curves</text>
<line x1="55" y1="300" x2="620" y2="300" stroke="#9db0bc"/><line x1="55" y1="20" x2="55" y2="300" stroke="#9db0bc"/>
<polyline points="{diagonal}" fill="none" stroke="#bcc8cf" stroke-dasharray="6 5"/>
{''.join(lines)}{''.join(legend)}
<text x="337" y="330" text-anchor="middle" font-family="sans-serif" font-size="11">False-positive rate</text>
<text x="15" y="160" text-anchor="middle" transform="rotate(-90 15 160)" font-family="sans-serif" font-size="11">True-positive rate</text>
</svg>"""
    path.write_text(document, encoding="utf-8")


def render_html_report(path: Path, results: dict[str, Any], generated_at: str) -> None:
    cards = []
    for name, experiment in results.items():
        metric_rows = []
        for split_name in ("internal_test", "external"):
            metrics = experiment["metrics"][split_name]
            metric_rows.append(
                "<tr>"
                f"<td>{html.escape(split_name.replace('_', ' ').title())}</td>"
                f"<td>{metrics['n']}</td>"
                f"<td>{metrics['positives']}</td>"
                f"<td>{metrics['negatives']}</td>"
                f"<td>{metrics['roc_auc']:.3f}</td>"
                f"<td>{metrics['balanced_accuracy']:.3f}</td>"
                f"<td>{metrics['sensitivity']:.3f}</td>"
                f"<td>{metrics['specificity']:.3f}</td>"
                f"<td>{metrics['precision']:.3f}</td>"
                f"<td>{metrics['f1']:.3f}</td>"
                f"<td>{metrics['brier_score']:.3f}</td>"
                "</tr>"
            )
        warning = (
            "ERG excluded to reduce direct-marker leakage. This is the primary analysis."
            if name == "primary"
            else "ERG included as an explicitly labelled positive control; do not present this as the primary predictive result."
        )
        cards.append(
            f"""<section><div class="eyebrow">{html.escape(name.replace('_', ' '))}</div>
<h2>{', '.join(map(html.escape, experiment['features']))}</h2><p>{html.escape(warning)}</p>
<table><thead><tr><th>Set</th><th>n</th><th>Positive</th><th>Negative</th><th>ROC-AUC</th><th>Balanced accuracy</th><th>Sensitivity</th><th>Specificity</th><th>Precision</th><th>F1</th><th>Brier score</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody></table></section>"""
        )
    external_threshold_failed = any(
        experiment["metrics"]["external"]["sensitivity"] == 0.0
        or experiment["metrics"]["external"]["specificity"] == 0.0
        for experiment in results.values()
    )
    threshold_note = (
        "<aside><strong>Important:</strong> The fixed 0.5 threshold did not transfer to the external cohort. "
        "The cohort-specific expression scales differ strongly, so ROC-AUC ranking must not be interpreted "
        "as a calibrated clinical decision rule.</aside>"
        if external_threshold_failed
        else ""
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyTorch fusion-status classifier</title><style>
:root{{--ink:#12263a;--muted:#607181;--blue:#145da0;--pale:#eef6f8;--line:#d8e2e8}}*{{box-sizing:border-box}}body{{margin:0;background:#f7fafb;color:var(--ink);font:15px/1.55 system-ui,sans-serif}}header,main,footer{{width:min(1050px,calc(100% - 40px));margin:auto}}header{{padding:60px 0 30px}}h1{{font-size:clamp(34px,6vw,64px);line-height:1.04;margin:.2em 0}}header p,section p,footer{{color:var(--muted)}}aside{{padding:16px 20px;background:#fff4dc;border:1px solid #e7c878;border-radius:12px}}section{{margin:18px 0;padding:24px;background:white;border:1px solid var(--line);border-radius:14px}}.eyebrow{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}}h2{{margin:.2em 0}}table{{width:100%;border-collapse:collapse;margin-top:18px}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}.figures{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:28px 0}}.figures img{{width:100%;background:white;border:1px solid var(--line);border-radius:14px}}footer{{padding:28px 0 60px;border-top:1px solid var(--line)}}@media(max-width:760px){{.figures{{grid-template-columns:1fr}}table{{font-size:12px}}}}
</style></head><body><header><div class="eyebrow">PyTorch extension · public cohorts</div><h1>Predicting TMPRSS2–ERG status from expression</h1><p>Models use the corrected <code>log2_expression</code> values. They are fitted with TCGA training data, selected with a separate TCGA validation subset, tested once on an untouched TCGA subset and evaluated without refitting in the independent SU2C 2019 capture cohort. This learning exercise was implemented after the thesis and is not represented as part of the submitted thesis analysis.</p></header>
<main>{threshold_note}{''.join(cards)}<div class="figures"><img src="training_curves.svg" alt="Training and early-stopping validation loss"><img src="roc_curves.svg" alt="Untouched internal-test and external ROC curves"></div></main>
<footer>Generated {html.escape(generated_at)}. External performance may reflect both biological differences and cross-cohort platform or batch effects.</footer></body></html>"""
    path.write_text(document, encoding="utf-8")


def run_experiments(
    expression_csv: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    internal_test_fraction: float = 0.2,
    hidden_units: int = 8,
    dropout: float = 0.1,
    learning_rate: float = 0.01,
    weight_decay: float = 0.001,
    batch_size: int = 32,
    maximum_epochs: int = 500,
    patience: int = 40,
) -> dict[str, Any]:
    torch = _require_torch()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    results: dict[str, Any] = {}
    histories: dict[str, list[dict[str, float | int]]] = {}
    curves: dict[str, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []

    for experiment_name, features in FEATURE_SETS.items():
        samples = load_samples(expression_csv, features)
        development = [s for s in samples if s.study_id == TRAIN_STUDY_ID]
        external = [s for s in samples if s.study_id == EXTERNAL_STUDY_ID]
        if not development or not external:
            raise ValueError("Both TCGA development and SU2C external samples are required")
        train, validation, internal_test = stratified_development_split(
            development,
            validation_fraction,
            internal_test_fraction,
            seed,
        )
        standardizer = fit_standardizer(train, features)
        model, history, best_epoch = train_model(
            train,
            validation,
            standardizer,
            seed=seed,
            hidden_units=hidden_units,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            maximum_epochs=maximum_epochs,
            patience=patience,
        )
        histories[experiment_name] = history
        split_probabilities = {
            "internal_test": _predict_probabilities(
                torch, model, internal_test, standardizer
            ),
            "external": _predict_probabilities(
                torch, model, external, standardizer
            ),
        }
        split_samples = {"internal_test": internal_test, "external": external}
        metrics = {}
        for split_name, probabilities in split_probabilities.items():
            split = split_samples[split_name]
            labels = [sample.label for sample in split]
            metrics[split_name] = binary_metrics(labels, probabilities)
            curves[f"{experiment_name}_{split_name}"] = {
                "auc": metrics[split_name]["roc_auc"],
                "points": roc_points(labels, probabilities),
            }
            for sample, probability in zip(split, probabilities):
                prediction_rows.append(
                    {
                        "experiment": experiment_name,
                        "split": split_name,
                        "study": sample.study,
                        "study_id": sample.study_id,
                        "sample_id": sample.sample_id,
                        "label": sample.label,
                        "probability": probability,
                        "prediction": int(probability >= 0.5),
                    }
                )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "features": list(features),
            "standardizer": asdict(standardizer),
            "hidden_units": hidden_units,
            "dropout": dropout,
            "best_epoch": best_epoch,
            "seed": seed,
            "input_value_column": ML_VALUE_COLUMN,
            "development_cohort": TRAIN_STUDY_ID,
            "external_cohort": EXTERNAL_STUDY_ID,
        }
        torch.save(checkpoint, output_dir / f"{experiment_name}_model.pt")
        results[experiment_name] = {
            "features": list(features),
            "development_cohort": TRAIN_STUDY_ID,
            "external_cohort": EXTERNAL_STUDY_ID,
            "input_value_column": ML_VALUE_COLUMN,
            "sample_counts": {
                "train": len(train),
                "validation": len(validation),
                "internal_test": len(internal_test),
                "external": len(external),
            },
            "class_counts": {
                "train": class_counts(train),
                "validation": class_counts(validation),
                "internal_test": class_counts(internal_test),
                "external": class_counts(external),
            },
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "standardizer": asdict(standardizer),
            "metrics": metrics,
        }

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "experiment",
            "split",
            "study",
            "study_id",
            "sample_id",
            "label",
            "probability",
            "prediction",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    metadata = {
        "generated_at": generated_at,
        "pytorch_version": torch.__version__,
        "configuration": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "internal_test_fraction": internal_test_fraction,
            "input_value_column": ML_VALUE_COLUMN,
            "hidden_units": hidden_units,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "maximum_epochs": maximum_epochs,
            "patience": patience,
        },
        "experiments": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    render_training_curve(output_dir / "training_curves.svg", histories)
    render_roc_curves(output_dir / "roc_curves.svg", curves)
    render_html_report(output_dir / "report.html", results, generated_at)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expression-csv",
        type=Path,
        default=Path("results/expression_values.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ml_results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--internal-test-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-units", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--maximum-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    metadata = run_experiments(
        arguments.expression_csv,
        arguments.output_dir,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
        internal_test_fraction=arguments.internal_test_fraction,
        hidden_units=arguments.hidden_units,
        dropout=arguments.dropout,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        batch_size=arguments.batch_size,
        maximum_epochs=arguments.maximum_epochs,
        patience=arguments.patience,
    )
    for name, result in metadata["experiments"].items():
        internal_test_auc = result["metrics"]["internal_test"]["roc_auc"]
        external_auc = result["metrics"]["external"]["roc_auc"]
        print(
            f"{name}: internal-test ROC-AUC={internal_test_auc:.3f}, "
            f"external ROC-AUC={external_auc:.3f}"
        )
    print(f"Report: {arguments.output_dir / 'report.html'}")
