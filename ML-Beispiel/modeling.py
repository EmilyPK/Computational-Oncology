#!/usr/bin/env python3
"""Train the baseline and PyTorch models on the prepared TCGA table.

Patients are split before preprocessing. Model settings and the classification
threshold are chosen in the development set; the held-out set is evaluated at
the end.
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SEED = 2026
TEST_FRACTION = 0.20
CV_FOLDS = 5
PROJECT_VERSION = "1.0.0"


@dataclass(frozen=True)
class SplitRecord:
    row_index: int
    patient_id: str
    label: int


@dataclass(frozen=True)
class TorchConfig:
    selected_genes: int
    hidden_units: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int = 32


TORCH_CONFIGS = (
    TorchConfig(32, 16, 0.10, 1e-3, 1e-3),
    TorchConfig(64, 32, 0.20, 1e-3, 1e-3),
    TorchConfig(64, 32, 0.35, 5e-4, 1e-2),
)


def _group_labels(records: Sequence[SplitRecord]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for record in records:
        if record.label not in (0, 1):
            raise ValueError("Labels must be binary")
        previous = labels.setdefault(record.patient_id, record.label)
        if previous != record.label:
            raise ValueError(f"Patient {record.patient_id} has conflicting labels")
    if set(labels.values()) != {0, 1}:
        raise ValueError("Both target classes are required")
    return labels


def stratified_group_holdout(
    records: Sequence[SplitRecord],
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
) -> tuple[list[int], list[int]]:
    """Return deterministic development/test row indices grouped by patient."""
    if not 0.05 <= test_fraction <= 0.50:
        raise ValueError("test_fraction must be between 0.05 and 0.50")
    group_labels = _group_labels(records)
    test_groups: set[str] = set()
    for label in (0, 1):
        groups = sorted(
            patient_id
            for patient_id, group_label in group_labels.items()
            if group_label == label
        )
        random.Random(seed + label).shuffle(groups)
        number_test = max(1, round(len(groups) * test_fraction))
        if number_test >= len(groups):
            raise ValueError("Each class needs patients in both split partitions")
        test_groups.update(groups[:number_test])
    development = sorted(
        record.row_index for record in records if record.patient_id not in test_groups
    )
    test = sorted(
        record.row_index for record in records if record.patient_id in test_groups
    )
    return development, test


def stratified_group_folds(
    records: Sequence[SplitRecord],
    number_of_folds: int = CV_FOLDS,
    seed: int = SEED,
) -> list[tuple[list[int], list[int]]]:
    """Create deterministic stratified patient-level CV train/validation indices."""
    if number_of_folds < 2:
        raise ValueError("number_of_folds must be at least two")
    group_labels = _group_labels(records)
    fold_groups: list[set[str]] = [set() for _ in range(number_of_folds)]
    for label in (0, 1):
        groups = sorted(
            patient_id
            for patient_id, group_label in group_labels.items()
            if group_label == label
        )
        if len(groups) < number_of_folds:
            raise ValueError("Each class needs at least one patient per fold")
        random.Random(seed + 100 + label).shuffle(groups)
        for position, patient_id in enumerate(groups):
            fold_groups[position % number_of_folds].add(patient_id)

    folds = []
    for validation_groups in fold_groups:
        train_indices = sorted(
            record.row_index
            for record in records
            if record.patient_id not in validation_groups
        )
        validation_indices = sorted(
            record.row_index
            for record in records
            if record.patient_id in validation_groups
        )
        folds.append((train_indices, validation_indices))
    return folds


def confusion_counts(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, int]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("Labels and probabilities must have equal non-zero length")
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for label, probability in zip(labels, probabilities):
        if label not in (0, 1):
            raise ValueError("Labels must be binary")
        predicted = int(float(probability) >= threshold)
        key = {(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}[
            (label, predicted)
        ]
        counts[key] += 1
    return counts


def threshold_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    counts = confusion_counts(labels, probabilities, threshold)
    tn, fp, fn, tp = (counts[key] for key in ("tn", "fp", "fn", "tp"))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        **{key: float(value) for key, value in counts.items()},
    }


def choose_balanced_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    """Select a deterministic development-only threshold by balanced accuracy."""
    if len(set(labels)) != 2:
        raise ValueError("Threshold selection requires both classes")
    candidates = sorted({0.0, 0.5, 1.0, *map(float, probabilities)})
    scored = [
        (
            threshold_metrics(labels, probabilities, threshold)["balanced_accuracy"],
            -abs(threshold - 0.5),
            -threshold,
            threshold,
        )
        for threshold in candidates
    ]
    return max(scored)[-1]


def _require_ml_stack() -> dict[str, Any]:
    """Import optional scientific dependencies only for an actual model run."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.base import clone
        from sklearn.calibration import calibration_curve
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "The ML dependencies are missing. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return locals()


def _cv_positions(
    frame: Any,
    folds: Sequence[tuple[list[int], list[int]]],
) -> list[tuple[list[int], list[int]]]:
    """Translate stable frame indices to positional indices required by sklearn."""
    positions = {int(index): position for position, index in enumerate(frame.index)}
    return [
        (
            [positions[index] for index in train if index in positions],
            [positions[index] for index in validation if index in positions],
        )
        for train, validation in folds
    ]


def _fit_sklearn_model(
    stack: dict[str, Any],
    name: str,
    estimator: Any,
    parameter_grid: dict[str, Sequence[Any]],
    development_frame: Any,
    test_frame: Any,
    feature_columns: Sequence[str],
    cv_splits: Sequence[tuple[list[int], list[int]]],
) -> dict[str, Any]:
    np = stack["np"]
    clone = stack["clone"]
    GridSearchCV = stack["GridSearchCV"]
    roc_auc_score = stack["roc_auc_score"]

    x_development = development_frame[list(feature_columns)]
    y_development = development_frame["fusion_status"].astype(int).to_numpy()
    search = GridSearchCV(
        estimator,
        parameter_grid,
        scoring="roc_auc",
        cv=list(cv_splits),
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(x_development, y_development)

    out_of_fold = np.full(len(development_frame), np.nan, dtype=float)
    for train_positions, validation_positions in cv_splits:
        fold_estimator = clone(search.best_estimator_)
        fold_estimator.fit(
            x_development.iloc[train_positions],
            y_development[train_positions],
        )
        out_of_fold[validation_positions] = fold_estimator.predict_proba(
            x_development.iloc[validation_positions]
        )[:, 1]
    if np.isnan(out_of_fold).any():
        raise RuntimeError(f"Incomplete out-of-fold predictions for {name}")
    threshold = choose_balanced_threshold(y_development.tolist(), out_of_fold.tolist())
    test_probabilities = search.best_estimator_.predict_proba(
        test_frame[list(feature_columns)]
    )[:, 1]
    return {
        "name": name,
        "estimator": search.best_estimator_,
        "best_parameters": search.best_params_,
        "cv_roc_auc": float(search.best_score_),
        "oof_probabilities": out_of_fold,
        "test_probabilities": test_probabilities,
        "threshold": threshold,
        "oof_roc_auc": float(roc_auc_score(y_development, out_of_fold)),
    }


def _make_clinical_estimator(stack: dict[str, Any]) -> Any:
    ColumnTransformer = stack["ColumnTransformer"]
    LogisticRegression = stack["LogisticRegression"]
    OneHotEncoder = stack["OneHotEncoder"]
    Pipeline = stack["Pipeline"]
    SimpleImputer = stack["SimpleImputer"]
    StandardScaler = stack["StandardScaler"]

    numeric = ["clinical__age"]
    categorical = ["clinical__path_t_stage", "clinical__path_n_stage"]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _make_expression_estimator(stack: dict[str, Any], gene_count: int) -> Any:
    LogisticRegression = stack["LogisticRegression"]
    Pipeline = stack["Pipeline"]
    SelectKBest = stack["SelectKBest"]
    SimpleImputer = stack["SimpleImputer"]
    StandardScaler = stack["StandardScaler"]
    f_classif = stack["f_classif"]
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=min(64, gene_count))),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=8000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _fit_preprocessors(
    stack: dict[str, Any],
    frame: Any,
    expression_columns: Sequence[str],
    clinical_columns: Sequence[str],
    selected_genes: int,
) -> tuple[Any, Any]:
    ColumnTransformer = stack["ColumnTransformer"]
    OneHotEncoder = stack["OneHotEncoder"]
    Pipeline = stack["Pipeline"]
    SelectKBest = stack["SelectKBest"]
    SimpleImputer = stack["SimpleImputer"]
    StandardScaler = stack["StandardScaler"]
    f_classif = stack["f_classif"]

    expression = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=min(selected_genes, len(expression_columns)),
                ),
            ),
        ]
    )
    clinical = ColumnTransformer(
        [
            (
                "age",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["clinical__age"],
            ),
            (
                "stage",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                ["clinical__path_t_stage", "clinical__path_n_stage"],
            ),
        ],
        verbose_feature_names_out=False,
    )
    labels = frame["fusion_status"].astype(int).to_numpy()
    expression.fit(frame[list(expression_columns)], labels)
    clinical.fit(frame[list(clinical_columns)], labels)
    return expression, clinical


def _transform_modalities(
    stack: dict[str, Any],
    frame: Any,
    expression_columns: Sequence[str],
    clinical_columns: Sequence[str],
    expression_preprocessor: Any,
    clinical_preprocessor: Any,
) -> tuple[Any, Any]:
    np = stack["np"]
    expression = expression_preprocessor.transform(frame[list(expression_columns)])
    clinical = clinical_preprocessor.transform(frame[list(clinical_columns)])
    return np.asarray(expression, dtype="float32"), np.asarray(clinical, dtype="float32")


def _new_network(stack: dict[str, Any], expression_size: int, clinical_size: int, config: TorchConfig) -> Any:
    torch = stack["torch"]
    nn = torch.nn

    class MultimodalNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expression_branch = nn.Sequential(
                nn.Linear(expression_size, config.hidden_units),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.clinical_branch = None
            clinical_hidden = 0
            if clinical_size:
                clinical_hidden = max(4, config.hidden_units // 4)
                self.clinical_branch = nn.Sequential(
                    nn.Linear(clinical_size, clinical_hidden),
                    nn.ReLU(),
                )
            self.head = nn.Sequential(
                nn.Linear(config.hidden_units + clinical_hidden, config.hidden_units),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_units, 1),
            )

        def forward(self, expression: Any, clinical: Any | None = None) -> Any:
            encoded = self.expression_branch(expression)
            if self.clinical_branch is not None:
                if clinical is None:
                    raise ValueError("Clinical input is required")
                encoded = torch.cat([encoded, self.clinical_branch(clinical)], dim=1)
            return self.head(encoded).squeeze(1)

    return MultimodalNetwork()


def _train_network(
    stack: dict[str, Any],
    expression_train: Any,
    clinical_train: Any,
    labels_train: Any,
    expression_validation: Any | None,
    clinical_validation: Any | None,
    labels_validation: Any | None,
    config: TorchConfig,
    seed: int,
    max_epochs: int,
    patience: int,
    fixed_epochs: int | None = None,
) -> tuple[Any, int]:
    np = stack["np"]
    torch = stack["torch"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    clinical_size = int(clinical_train.shape[1])
    model = _new_network(stack, int(expression_train.shape[1]), clinical_size, config)
    positive = float(labels_train.sum())
    negative = float(len(labels_train) - positive)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(max(negative / max(positive, 1.0), 1e-6))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(expression_train),
        torch.tensor(clinical_train),
        torch.tensor(labels_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    number_epochs = fixed_epochs or max_epochs
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(1, number_epochs + 1):
        model.train()
        for batch_expression, batch_clinical, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(
                batch_expression,
                batch_clinical if clinical_size else None,
            )
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
        if fixed_epochs is not None:
            best_epoch = epoch
            continue
        if expression_validation is None or labels_validation is None:
            raise ValueError("Validation data are required for early stopping")
        model.eval()
        with torch.no_grad():
            validation_logits = model(
                torch.tensor(expression_validation),
                torch.tensor(clinical_validation) if clinical_size else None,
            )
            validation_loss = float(
                loss_function(
                    validation_logits,
                    torch.tensor(labels_validation, dtype=torch.float32),
                )
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    return model, best_epoch


def _predict_network(stack: dict[str, Any], model: Any, expression: Any, clinical: Any) -> Any:
    torch = stack["torch"]
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(expression),
            torch.tensor(clinical) if clinical.shape[1] else None,
        )
        return torch.sigmoid(logits).cpu().numpy()


def _fit_torch_model(
    stack: dict[str, Any],
    name: str,
    development_frame: Any,
    test_frame: Any,
    expression_columns: Sequence[str],
    clinical_columns: Sequence[str],
    cv_splits: Sequence[tuple[list[int], list[int]]],
    include_clinical: bool,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    np = stack["np"]
    roc_auc_score = stack["roc_auc_score"]
    labels = development_frame["fusion_status"].astype(int).to_numpy()
    candidate_results = []
    for config_index, config in enumerate(TORCH_CONFIGS):
        out_of_fold = np.full(len(development_frame), np.nan, dtype=float)
        epochs = []
        for fold_index, (train_positions, validation_positions) in enumerate(cv_splits):
            train_frame = development_frame.iloc[train_positions]
            validation_frame = development_frame.iloc[validation_positions]
            expr_pre, clinical_pre = _fit_preprocessors(
                stack,
                train_frame,
                expression_columns,
                clinical_columns,
                config.selected_genes,
            )
            expr_train, clinical_train = _transform_modalities(
                stack,
                train_frame,
                expression_columns,
                clinical_columns,
                expr_pre,
                clinical_pre,
            )
            expr_validation, clinical_validation = _transform_modalities(
                stack,
                validation_frame,
                expression_columns,
                clinical_columns,
                expr_pre,
                clinical_pre,
            )
            if not include_clinical:
                clinical_train = np.empty((len(train_frame), 0), dtype="float32")
                clinical_validation = np.empty((len(validation_frame), 0), dtype="float32")
            model, best_epoch = _train_network(
                stack,
                expr_train,
                clinical_train,
                train_frame["fusion_status"].astype(int).to_numpy(),
                expr_validation,
                clinical_validation,
                validation_frame["fusion_status"].astype(int).to_numpy(),
                config,
                seed=SEED + config_index * 100 + fold_index,
                max_epochs=max_epochs,
                patience=patience,
            )
            out_of_fold[validation_positions] = _predict_network(
                stack,
                model,
                expr_validation,
                clinical_validation,
            )
            epochs.append(best_epoch)
        candidate_results.append(
            {
                "config": config,
                "oof_probabilities": out_of_fold,
                "cv_roc_auc": float(roc_auc_score(labels, out_of_fold)),
                "epochs": epochs,
            }
        )
    selected = max(candidate_results, key=lambda result: result["cv_roc_auc"])
    config = selected["config"]
    final_epochs = max(1, round(statistics.median(selected["epochs"])))
    expr_pre, clinical_pre = _fit_preprocessors(
        stack,
        development_frame,
        expression_columns,
        clinical_columns,
        config.selected_genes,
    )
    expr_development, clinical_development = _transform_modalities(
        stack,
        development_frame,
        expression_columns,
        clinical_columns,
        expr_pre,
        clinical_pre,
    )
    expr_test, clinical_test = _transform_modalities(
        stack,
        test_frame,
        expression_columns,
        clinical_columns,
        expr_pre,
        clinical_pre,
    )
    if not include_clinical:
        clinical_development = np.empty((len(development_frame), 0), dtype="float32")
        clinical_test = np.empty((len(test_frame), 0), dtype="float32")
    final_model, _ = _train_network(
        stack,
        expr_development,
        clinical_development,
        labels,
        None,
        None,
        None,
        config,
        seed=SEED + 999,
        max_epochs=max_epochs,
        patience=patience,
        fixed_epochs=final_epochs,
    )
    threshold = choose_balanced_threshold(labels.tolist(), selected["oof_probabilities"].tolist())
    return {
        "name": name,
        "model": final_model,
        "expression_preprocessor": expr_pre,
        "clinical_preprocessor": clinical_pre,
        "include_clinical": include_clinical,
        "best_parameters": {**asdict(config), "epochs": final_epochs},
        "cv_roc_auc": selected["cv_roc_auc"],
        "oof_probabilities": selected["oof_probabilities"],
        "test_probabilities": _predict_network(
            stack,
            final_model,
            expr_test,
            clinical_test,
        ),
        "threshold": threshold,
        "transformed_test": (expr_test, clinical_test),
    }


def _full_metrics(
    stack: dict[str, Any],
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    average_precision_score = stack["average_precision_score"]
    brier_score_loss = stack["brier_score_loss"]
    roc_auc_score = stack["roc_auc_score"]
    thresholded = threshold_metrics(labels, probabilities, threshold)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        **thresholded,
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _bootstrap_intervals(
    stack: dict[str, Any],
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    np = stack["np"]
    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        indices = rng.integers(0, len(labels_array), len(labels_array))
        sampled_labels = labels_array[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_probabilities = probabilities_array[indices]
        metrics = _full_metrics(
            stack,
            sampled_labels.tolist(),
            sampled_probabilities.tolist(),
            threshold,
        )
        for key, value in metrics.items():
            if key not in {"tn", "fp", "fn", "tp"}:
                collected[key].append(float(value))
    return {
        key: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for key, values in collected.items()
        if values
    }


def _integrated_gradients(
    stack: dict[str, Any],
    result: dict[str, Any],
    expression_columns: Sequence[str],
    clinical_columns: Sequence[str],
    steps: int = 32,
) -> list[tuple[str, float]]:
    torch = stack["torch"]
    np = stack["np"]
    model = result["model"]
    expression, clinical = result["transformed_test"]
    expression = expression[: min(100, len(expression))]
    clinical = clinical[: min(100, len(clinical))]
    expression_tensor = torch.tensor(expression)
    clinical_tensor = torch.tensor(clinical)
    expression_total = torch.zeros_like(expression_tensor)
    clinical_total = torch.zeros_like(clinical_tensor)
    for alpha in torch.linspace(0.0, 1.0, steps):
        scaled_expression = (expression_tensor * alpha).detach().requires_grad_(True)
        scaled_clinical = (clinical_tensor * alpha).detach().requires_grad_(True)
        logits = model(
            scaled_expression,
            scaled_clinical if clinical.shape[1] else None,
        )
        inputs = [scaled_expression]
        if clinical.shape[1]:
            inputs.append(scaled_clinical)
        gradients = torch.autograd.grad(logits.sum(), inputs)
        expression_total += gradients[0].detach()
        if clinical.shape[1]:
            clinical_total += gradients[1].detach()
    expression_scores = (expression_tensor * expression_total / steps).abs().mean(0).numpy()
    support = result["expression_preprocessor"].named_steps["select"].get_support()
    selected_expression_names = [
        name for name, selected in zip(expression_columns, support) if selected
    ]
    scores = list(zip(selected_expression_names, map(float, expression_scores)))
    if clinical.shape[1]:
        clinical_names = result["clinical_preprocessor"].get_feature_names_out(
            list(clinical_columns)
        )
        clinical_scores = (clinical_tensor * clinical_total / steps).abs().mean(0).numpy()
        scores.extend(zip(map(str, clinical_names), map(float, clinical_scores)))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _linear_coefficients(
    result: dict[str, Any],
    expression_columns: Sequence[str],
    clinical_columns: Sequence[str],
) -> list[tuple[str, float]]:
    estimator = result["estimator"]
    if result["name"] == "clinical_logistic":
        names = estimator.named_steps["preprocess"].get_feature_names_out(
            list(clinical_columns)
        )
    else:
        support = estimator.named_steps["select"].get_support()
        names = [name for name, selected in zip(expression_columns, support) if selected]
    coefficients = estimator.named_steps["model"].coef_[0]
    return sorted(
        zip(map(str, names), map(float, coefficients)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )


def _save_plots(
    stack: dict[str, Any],
    results: Sequence[dict[str, Any]],
    labels: Sequence[int],
    output_dir: Path,
) -> None:
    plt = stack["plt"]
    calibration_curve = stack["calibration_curve"]
    roc_curve = stack["roc_curve"]

    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    for result in results:
        false_positive, true_positive, _ = roc_curve(labels, result["test_probabilities"])
        axis.plot(false_positive, true_positive, label=result["name"])
    axis.plot([0, 1], [0, 1], "--", color="0.55", label="chance")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Untouched TCGA test set")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    for result in results:
        observed, predicted = calibration_curve(
            labels,
            result["test_probabilities"],
            n_bins=6,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=result["name"])
    axis.plot([0, 1], [0, 1], "--", color="0.55", label="perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title="Probability calibration")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "calibration.svg")
    plt.close(figure)


def _write_report(
    metric_rows: Sequence[dict[str, Any]],
    split_counts: dict[str, Counter[int]],
    output_dir: Path,
) -> None:
    table_rows = []
    for row in metric_rows:
        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row[key]))}</td>"
                for key in (
                    "model",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "sensitivity",
                    "specificity",
                    "brier_score",
                )
            )
            + "</tr>"
        )
    report = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>TCGA fusion ML report</title>
<style>body{{font:16px/1.5 system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#18212b}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3da;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:48%;vertical-align:top}}</style>
<h1>Leakage-aware TMPRSS2–ERG prediction</h1>
<p>The test partition was assigned once at patient level and was not used for preprocessing, feature selection, hyperparameter selection or threshold selection.</p>
<p><b>Development:</b> {sum(split_counts['development'].values())} samples ({dict(split_counts['development'])}); <b>test:</b> {sum(split_counts['test'].values())} samples ({dict(split_counts['test'])}).</p>
<table><thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Balanced accuracy</th><th>Sensitivity</th><th>Specificity</th><th>Brier score</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Held-out discrimination and calibration</h2>
<img src="roc_curves.svg" alt="ROC curves"><img src="calibration.svg" alt="Calibration curves">
<h2>Interpretation boundary</h2>
<p>This is an internal TCGA validation study, not a clinical diagnostic model. A random split estimates performance for future samples processed like TCGA; it does not establish transportability to another laboratory or disease stage.</p>
</html>"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def run_experiment(
    data_path: Path,
    output_dir: Path,
    test_fraction: float = TEST_FRACTION,
    cv_folds: int = CV_FOLDS,
    seed: int = SEED,
    bootstrap_repeats: int = 1000,
    max_epochs: int = 300,
    patience: int = 25,
) -> None:
    stack = _require_ml_stack()
    pd = stack["pd"]
    frame = pd.read_csv(data_path)
    required = {"sample_id", "patient_id", "fusion_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing data columns: {', '.join(sorted(missing))}")
    frame["fusion_status"] = frame["fusion_status"].astype(int)
    expression_columns = sorted(column for column in frame if column.startswith("expr__"))
    clinical_columns = [
        "clinical__age",
        "clinical__path_t_stage",
        "clinical__path_n_stage",
    ]
    if not expression_columns or any(column not in frame for column in clinical_columns):
        raise ValueError("Expression and configured clinical predictors are required")
    if {"expr__ERG", "expr__TMPRSS2"} & set(expression_columns):
        raise ValueError("Direct fusion markers must not be present among predictors")

    records = [
        SplitRecord(int(index), str(row.patient_id), int(row.fusion_status))
        for index, row in frame.iterrows()
    ]
    development_indices, test_indices = stratified_group_holdout(
        records,
        test_fraction=test_fraction,
        seed=seed,
    )
    development = frame.loc[development_indices].copy()
    test = frame.loc[test_indices].copy()
    development_records = [record for record in records if record.row_index in set(development_indices)]
    folds = stratified_group_folds(development_records, cv_folds, seed)
    cv_splits = _cv_positions(development, folds)

    clinical_result = _fit_sklearn_model(
        stack,
        "clinical_logistic",
        _make_clinical_estimator(stack),
        {"model__C": (0.01, 0.1, 1.0, 10.0)},
        development,
        test,
        clinical_columns,
        cv_splits,
    )
    expression_result = _fit_sklearn_model(
        stack,
        "expression_elastic_net",
        _make_expression_estimator(stack, len(expression_columns)),
        {
            "select__k": tuple(
                sorted({min(value, len(expression_columns)) for value in (16, 32, 64)})
            ),
            "model__C": (0.03, 0.1, 0.3, 1.0),
            "model__l1_ratio": (0.0, 0.5, 1.0),
        },
        development,
        test,
        expression_columns,
        cv_splits,
    )
    expression_mlp = _fit_torch_model(
        stack,
        "expression_mlp",
        development,
        test,
        expression_columns,
        clinical_columns,
        cv_splits,
        include_clinical=False,
        max_epochs=max_epochs,
        patience=patience,
    )
    multimodal_mlp = _fit_torch_model(
        stack,
        "multimodal_mlp",
        development,
        test,
        expression_columns,
        clinical_columns,
        cv_splits,
        include_clinical=True,
        max_epochs=max_epochs,
        patience=patience,
    )
    results = [clinical_result, expression_result, expression_mlp, multimodal_mlp]
    labels_test = test["fusion_status"].astype(int).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    prediction_rows = []
    selection_manifest = {}
    for model_index, result in enumerate(results):
        probabilities = list(map(float, result["test_probabilities"]))
        metrics = _full_metrics(stack, labels_test, probabilities, result["threshold"])
        intervals = _bootstrap_intervals(
            stack,
            labels_test,
            probabilities,
            result["threshold"],
            repeats=bootstrap_repeats,
            seed=seed + model_index,
        )
        row: dict[str, Any] = {
            "model": result["name"],
            "n_test": len(test),
            "threshold": round(float(result["threshold"]), 6),
        }
        for key, value in metrics.items():
            row[key] = round(value, 6)
            if key in intervals:
                row[f"{key}_ci_low"] = round(intervals[key][0], 6)
                row[f"{key}_ci_high"] = round(intervals[key][1], 6)
        metric_rows.append(row)
        for (_, sample), probability in zip(test.iterrows(), probabilities):
            prediction_rows.append(
                {
                    "model": result["name"],
                    "sample_id": sample["sample_id"],
                    "patient_id": sample["patient_id"],
                    "fusion_status": int(sample["fusion_status"]),
                    "probability": probability,
                    "threshold": result["threshold"],
                    "prediction": int(probability >= result["threshold"]),
                }
            )
        selection_manifest[result["name"]] = {
            "best_parameters": result["best_parameters"],
            "development_cv_roc_auc": result["cv_roc_auc"],
            "development_threshold": result["threshold"],
        }

    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    split_rows = []
    development_set = set(development_indices)
    for record in records:
        split_rows.append(
            {
                "sample_id": frame.loc[record.row_index, "sample_id"],
                "patient_id": record.patient_id,
                "fusion_status": record.label,
                "partition": "development" if record.row_index in development_set else "test",
            }
        )
    pd.DataFrame(split_rows).to_csv(output_dir / "split_manifest.csv", index=False)

    feature_rows = []
    for result in (clinical_result, expression_result):
        for rank, (feature, importance) in enumerate(
            _linear_coefficients(result, expression_columns, clinical_columns)[:20],
            start=1,
        ):
            feature_rows.append(
                {"model": result["name"], "rank": rank, "feature": feature, "importance": importance, "method": "standardized_logistic_coefficient"}
            )
    for result in (expression_mlp, multimodal_mlp):
        for rank, (feature, importance) in enumerate(
            _integrated_gradients(stack, result, expression_columns, clinical_columns)[:20],
            start=1,
        ):
            feature_rows.append(
                {"model": result["name"], "rank": rank, "feature": feature, "importance": importance, "method": "mean_absolute_integrated_gradient"}
            )
    pd.DataFrame(feature_rows).to_csv(output_dir / "top_features.csv", index=False)

    run_manifest = {
        "project_version": PROJECT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "seed": seed,
        "test_fraction": test_fraction,
        "cv_folds": cv_folds,
        "bootstrap_repeats": bootstrap_repeats,
        "split_counts": {
            "development": dict(Counter(map(int, development["fusion_status"]))),
            "test": dict(Counter(map(int, test["fusion_status"]))),
        },
        "model_selection": selection_manifest,
        "interpretation": "Internal TCGA validation only; no claim of cross-cohort transportability.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    _save_plots(stack, results, labels_test, output_dir)
    split_counts = {
        "development": Counter(map(int, development["fusion_status"])),
        "test": Counter(map(int, test["fusion_status"])),
    }
    _write_report(metric_rows, split_counts, output_dir)
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"\nResults written to {output_dir}")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project_dir / "data" / "model_input.csv")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short smoke run (50 epochs and 100 bootstrap resamples).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        test_fraction=args.test_fraction,
        cv_folds=args.cv_folds,
        seed=args.seed,
        bootstrap_repeats=100 if args.quick else args.bootstrap_repeats,
        max_epochs=50 if args.quick else args.max_epochs,
        patience=10 if args.quick else args.patience,
    )


if __name__ == "__main__":
    main()
