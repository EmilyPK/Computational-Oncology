#!/usr/bin/env python3
"""Reproduce the thesis comparison in two public prostate-cancer cohorts.

Samples are grouped by TMPRSS2--ERG status and compared for ERG, RAI2, CTBP1
and CTBP2 expression. The script writes the tables and report used in the
walkthrough notebook.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


API_BASE = "https://www.cbioportal.org/api"
GENES = ("ERG", "RAI2", "CTBP1", "CTBP2")


@dataclass(frozen=True)
class Study:
    label: str
    study_id: str
    expression_profile_id: str
    structural_variant_profile_id: str
    grouping_source: str
    analysis_transform: str


STUDIES = (
    Study(
        label="TCGA PanCancer Atlas",
        study_id="prad_tcga_pan_can_atlas_2018",
        expression_profile_id="prad_tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
        structural_variant_profile_id="prad_tcga_pan_can_atlas_2018_structural_variants",
        grouping_source="structural_variants",
        analysis_transform="legacy_r_decimal_stripping",
    ),
    Study(
        label="SU2C/PCF Dream Team (2019 capture)",
        study_id="prad_su2c_2019",
        expression_profile_id="prad_su2c_2019_mrna_seq_fpkm_capture",
        structural_variant_profile_id="prad_su2c_2019_structural_variants",
        grouping_source="clinical_ets",
        analysis_transform="log2",
    ),
)


class CbioPortalClient:
    """Small cBioPortal client with a local JSON cache."""

    def __init__(self, base_url: str, cache_dir: Path, retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries

    def get(self, path: str, query: dict[str, str] | None = None) -> Any:
        suffix = "?" + urllib.parse.urlencode(query) if query else ""
        return self._request("GET", path + suffix, None)

    def post(self, path: str, payload: Any, query: dict[str, str] | None = None) -> Any:
        suffix = "?" + urllib.parse.urlencode(query) if query else ""
        return self._request("POST", path + suffix, payload)

    def _request(self, method: str, path: str, payload: Any) -> Any:
        body = None if payload is None else json.dumps(payload, sort_keys=True).encode("utf-8")
        key_material = method.encode() + b"\0" + path.encode() + b"\0" + (body or b"")
        cache_path = self.cache_dir / f"{hashlib.sha256(key_material).hexdigest()}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "prostate-fusion-cohort-analysis/1.0",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = json.load(response)
                cache_path.write_text(json.dumps(result), encoding="utf-8")
                return result
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"cBioPortal request failed after {self.retries} attempts") from last_error


def fetch_gene_ids(client: CbioPortalClient) -> dict[int, str]:
    genes = client.post(
        "/genes/fetch",
        list(GENES),
        query={"geneIdType": "HUGO_GENE_SYMBOL", "projection": "SUMMARY"},
    )
    mapping = {int(gene["entrezGeneId"]): str(gene["hugoGeneSymbol"]) for gene in genes}
    missing = set(GENES) - set(mapping.values())
    if missing:
        raise RuntimeError(f"Genes missing from cBioPortal response: {sorted(missing)}")
    return mapping


def is_tmprss2_erg_fusion(variant: dict[str, Any]) -> bool:
    partners = {
        str(variant.get("site1HugoSymbol", "")).upper(),
        str(variant.get("site2HugoSymbol", "")).upper(),
    }
    return partners == {"TMPRSS2", "ERG"}


def legacy_r_tcga_log2(value: Any) -> float:
    """Reproduce the numeric conversion used by the original TCGA R script.

    The source script removed decimal points from character values before
    converting them to numbers.  This is retained only for exact thesis
    reproduction; ``log2_expression`` in the tidy output stores the corrected
    direct log2(value + 1) transformation for downstream machine learning.
    """
    digits = str(value).replace(".", "")
    if not digits.isdigit():
        raise ValueError(f"Value cannot be converted by the legacy R rule: {value!r}")
    return math.log2(float(digits) + 1.0)


def classify_su2c_ets(fusion_sequence: str | None, fusion_detail: str | None) -> str | None:
    """Reproduce the SU2C inclusion and grouping used by the supplied R script.

    Samples without an ETS sequencing result are omitted.  Named non-TMPRSS2–ERG
    fusions are excluded by the input selection used for the thesis figure.
    A missing fusion detail is assigned to the comparison group, matching the
    original 85/96 sample split.
    """
    if fusion_sequence is None:
        return None
    if fusion_detail == "TMPRSS2-ERG":
        return "TMPRSS2-ERG detected"
    if fusion_detail in (None, ""):
        return "not detected"
    return None


def fetch_su2c_clinical_groups(
    client: CbioPortalClient,
    study: Study,
    sample_ids: Sequence[str],
) -> dict[str, str]:
    clinical = client.post(
        f"/studies/{study.study_id}/clinical-data/fetch",
        {
            "ids": sorted(sample_ids),
            "attributeIds": ["ETS_FUSION_SEQ", "ETS_FUSION_DETAILS"],
        },
        query={"clinicalDataType": "SAMPLE", "projection": "SUMMARY"},
    )
    values: dict[str, dict[str, str]] = {}
    for record in clinical:
        values.setdefault(str(record["sampleId"]), {})[
            str(record["clinicalAttributeId"])
        ] = str(record["value"])

    groups: dict[str, str] = {}
    for sample_id in sample_ids:
        attributes = values.get(sample_id, {})
        group = classify_su2c_ets(
            attributes.get("ETS_FUSION_SEQ"),
            attributes.get("ETS_FUSION_DETAILS"),
        )
        if group is not None:
            groups[sample_id] = group
    return groups


def fetch_study_rows(
    client: CbioPortalClient,
    study: Study,
    gene_names: dict[int, str],
) -> list[dict[str, Any]]:
    expression = client.post(
        "/molecular-data/fetch",
        {
            "molecularProfileIds": [study.expression_profile_id],
            "entrezGeneIds": sorted(gene_names),
        },
        query={"projection": "SUMMARY"},
    )
    expression_by_sample: dict[str, dict[str, Any]] = {}
    for record in expression:
        raw_value = record["value"]
        value = float(raw_value)
        if value < 0:
            raise ValueError(
                f"Unexpected negative expression value for {record['sampleId']}: {value}"
            )
        expression_by_sample.setdefault(str(record["sampleId"]), {})[
            gene_names[int(record["entrezGeneId"])]
        ] = raw_value

    complete_samples = {
        sample_id
        for sample_id, values in expression_by_sample.items()
        if all(gene in values for gene in GENES)
    }
    if study.grouping_source == "clinical_ets":
        sample_groups = fetch_su2c_clinical_groups(client, study, sorted(complete_samples))
    else:
        variants = client.post(
            "/structural-variant/fetch",
            {"molecularProfileIds": [study.structural_variant_profile_id]},
        )
        fusion_samples = {
            str(variant["sampleId"])
            for variant in variants
            if is_tmprss2_erg_fusion(variant)
        }
        sample_groups = {
            sample_id: (
                "TMPRSS2-ERG detected" if sample_id in fusion_samples else "not detected"
            )
            for sample_id in complete_samples
        }

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(complete_samples & sample_groups.keys()):
        values = expression_by_sample[sample_id]
        try:
            thesis_values = {
                gene: (
                    legacy_r_tcga_log2(values[gene])
                    if study.analysis_transform == "legacy_r_decimal_stripping"
                    else math.log2(float(values[gene]) + 1.0)
                )
                for gene in GENES
            }
        except ValueError:
            continue
        for gene in GENES:
            value = float(values[gene])
            rows.append(
                {
                    "study": study.label,
                    "study_id": study.study_id,
                    "sample_id": sample_id,
                    "gene": gene,
                    "expression": value,
                    "log2_expression": math.log2(value + 1.0),
                    "thesis_analysis_value": thesis_values[gene],
                    "fusion_status": sample_groups[sample_id],
                }
            )
    return rows


def average_ranks(values: Sequence[float]) -> tuple[list[float], list[int]]:
    """Return average ranks and tie-group sizes for a numeric sequence."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    tie_sizes: list[int] = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        if end - start > 1:
            tie_sizes.append(end - start)
        start = end
    return ranks, tie_sizes


def mann_whitney_u(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    """Two-sided Mann--Whitney U test with tie and continuity correction.

    Returns U for ``x``, an asymptotic two-sided p-value, and rank-biserial
    correlation. The implementation follows the standard large-sample normal
    approximation and is suitable for the cohort sizes used here.
    """
    if not x or not y:
        raise ValueError("Both Mann--Whitney groups must contain observations")
    n_x, n_y = len(x), len(y)
    combined = [float(v) for v in x] + [float(v) for v in y]
    ranks, tie_sizes = average_ranks(combined)
    u_x = sum(ranks[:n_x]) - n_x * (n_x + 1) / 2.0
    mean_u = n_x * n_y / 2.0
    total = n_x + n_y
    tie_term = sum(size**3 - size for size in tie_sizes)
    variance = n_x * n_y / 12.0 * (
        (total + 1) - tie_term / (total * (total - 1))
    )
    if variance <= 0:
        p_value = 1.0
    else:
        z = max(0.0, abs(u_x - mean_u) - 0.5) / math.sqrt(variance)
        p_value = math.erfc(z / math.sqrt(2.0))
    rank_biserial = 2.0 * u_x / (n_x * n_y) - 1.0
    return u_x, p_value, rank_biserial


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini--Hochberg false-discovery-rate adjustment."""
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running_minimum = 1.0
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running_minimum = min(running_minimum, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running_minimum)
    return adjusted


def row_analysis_value(row: dict[str, Any]) -> float:
    """Return the thesis-reproduction value, with a test-data fallback."""
    return float(row.get("thesis_analysis_value", row["log2_expression"]))


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for study in STUDIES:
        study_results: list[dict[str, Any]] = []
        for gene in GENES:
            gene_rows = [r for r in rows if r["study_id"] == study.study_id and r["gene"] == gene]
            detected = [row_analysis_value(r) for r in gene_rows if r["fusion_status"].startswith("TMPRSS2")]
            not_detected = [row_analysis_value(r) for r in gene_rows if r["fusion_status"] == "not detected"]
            u_statistic, p_value, effect = mann_whitney_u(detected, not_detected)
            study_results.append(
                {
                    "study": study.label,
                    "study_id": study.study_id,
                    "gene": gene,
                    "n_fusion": len(detected),
                    "n_not_detected": len(not_detected),
                    "median_fusion": quantile(detected, 0.5),
                    "median_not_detected": quantile(not_detected, 0.5),
                    "median_difference": quantile(detected, 0.5) - quantile(not_detected, 0.5),
                    "u_statistic": u_statistic,
                    "p_value": p_value,
                    "rank_biserial": effect,
                }
            )
        adjusted = benjamini_hochberg([r["p_value"] for r in study_results])
        for result, q_value in zip(study_results, adjusted):
            result["q_value_bh"] = q_value
        summaries.extend(study_results)
    return summaries


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def svg_distribution(rows: Sequence[dict[str, Any]], study_id: str, gene: str) -> str:
    selected = [r for r in rows if r["study_id"] == study_id and r["gene"] == gene]
    groups = (
        (
            "No detected fusion",
            [r for r in selected if r["fusion_status"] == "not detected"],
            215,
            "#64748b",
            "#e2e8f0",
        ),
        (
            "TMPRSS2–ERG detected",
            [r for r in selected if r["fusion_status"].startswith("TMPRSS2")],
            430,
            "#087f8c",
            "#cceff0",
        ),
    )
    all_values = [row_analysis_value(r) for r in selected]
    detected_values = [row_analysis_value(r) for r in groups[1][1]]
    not_detected_values = [row_analysis_value(r) for r in groups[0][1]]
    _, p_value, _ = mann_whitney_u(detected_values, not_detected_values)

    def format_p_value(value: float) -> str:
        if value < 0.001:
            mantissa, exponent = f"{value:.2e}".split("e")
            superscript = str(int(exponent)).translate(
                str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
            )
            return f"{mantissa} × 10{superscript}"
        return f"{value:.3g}"

    def format_p_stars(value: float) -> str:
        if value < 0.001:
            return "***"
        if value < 0.01:
            return "**"
        if value < 0.05:
            return "*"
        return "ns"

    fixed_axes = {
        "prad_tcga_pan_can_atlas_2018": (10.0, 25.0, 5.0, 23.75, 23.4, 24.5),
        "prad_su2c_2019": (0.0, 15.0, 5.0, 13.75, 13.4, 14.5),
    }
    if study_id in fixed_axes:
        low, high, tick_step, bracket_line, bracket_bottom, p_text_position = (
            fixed_axes[study_id]
        )
    else:
        raw_low, raw_high = min(all_values), max(all_values)
        raw_span = max(raw_high - raw_low, 0.5)
        tick_step = 10 ** math.floor(math.log10(raw_span / 5))
        for multiplier in (1, 2, 2.5, 5, 10):
            candidate = tick_step * multiplier
            if raw_span / candidate <= 6:
                tick_step = candidate
                break
        low = math.floor((raw_low - 0.03 * raw_span) / tick_step) * tick_step
        high = math.ceil((raw_high + 0.03 * raw_span) / tick_step) * tick_step
        if high <= low:
            high = low + tick_step
        bracket_line = high - 0.08 * (high - low)
        bracket_bottom = high - 0.11 * (high - low)
        p_text_position = high - 0.025 * (high - low)

    def y_position(value: float) -> float:
        return 344 - (value - low) / (high - low) * 264

    figure_id = "".join(character if character.isalnum() else "-" for character in f"{study_id}-{gene}")

    elements = [
        '<svg viewBox="0 0 600 440" width="100%" style="display:block;height:auto" '
        f'role="img" aria-labelledby="{figure_id}-title {figure_id}-description" xmlns="http://www.w3.org/2000/svg">',
        f'<title id="{figure_id}-title">{html.escape(gene)} expression by TMPRSS2–ERG status</title>',
        f'<desc id="{figure_id}-description">Vertical distribution of log2 transformed '
        f'{html.escape(gene)} expression with individual samples, interquartile ranges, '
        'Tukey whiskers, medians and means.</desc>',
        """<defs><style>
            .plot-text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
            .grid{stroke:#e8eef2;stroke-width:1}
            .axis{stroke:#94a3b8;stroke-width:1.2}
            .tick{fill:#64748b;font-size:11px}
            .axis-label{fill:#475569;font-size:12px;font-weight:600}
            .group-label{fill:#1e293b;font-size:12px;font-weight:650}
            .group-count{fill:#64748b;font-size:10.5px}
            .median-label{fill:#334155;font-size:10.5px;font-weight:650}
            .mean-label{fill:#334155;font-size:10.5px;font-weight:650}
            .test-label{fill:#334155;font-size:11px;font-weight:650}
            .test-name{fill:#64748b;font-size:10.5px}
            .significance-bracket{stroke:#334155;stroke-width:1.2;stroke-linecap:round}
            .whisker{stroke-width:1.5;stroke-linecap:round}
            .whisker-cap{stroke-width:1.5}
            .median{stroke-width:2.5}
            .point{stroke:#fff;stroke-width:.35}
        </style></defs>""",
        f'<defs><clipPath id="{figure_id}-plot-clip"><rect x="70" y="80" '
        'width="490" height="264"/></clipPath></defs>',
        '<rect x="0.5" y="0.5" width="599" height="439" rx="12" fill="#ffffff" stroke="#e2e8f0"/>',
        '<g class="plot-text">',
        '<text x="315" y="29" text-anchor="middle" class="test-name">Two-sided Mann–Whitney U test</text>',
    ]
    tick_count = int(round((high - low) / tick_step))
    for tick_index in range(tick_count + 1):
        tick_value = low + tick_index * tick_step
        tick_y = y_position(tick_value)
        tick_label = f"{tick_value:.1f}" if tick_step < 1 else f"{tick_value:g}"
        elements.extend(
            [
                f'<line x1="70" y1="{tick_y:.2f}" x2="560" y2="{tick_y:.2f}" class="grid"/>',
                f'<text x="60" y="{tick_y + 4:.2f}" text-anchor="end" class="tick">{tick_label}</text>',
            ]
        )
    elements.extend(
        [
            '<line x1="70" y1="80" x2="70" y2="344" class="axis"/>',
            '<line x1="70" y1="344" x2="560" y2="344" class="axis"/>',
            '<text x="18" y="212" text-anchor="middle" class="axis-label" '
            'transform="rotate(-90 18 212)">log₂(mRNA expression + 1)</text>',
            f'<line x1="215" y1="{y_position(bracket_line):.2f}" x2="430" '
            f'y2="{y_position(bracket_line):.2f}" class="significance-bracket"/>',
            f'<line x1="215" y1="{y_position(bracket_line):.2f}" x2="215" '
            f'y2="{y_position(bracket_bottom):.2f}" class="significance-bracket"/>',
            f'<line x1="430" y1="{y_position(bracket_line):.2f}" x2="430" '
            f'y2="{y_position(bracket_bottom):.2f}" class="significance-bracket"/>',
            f'<text x="322.5" y="{y_position(p_text_position) + 4:.2f}" '
            f'text-anchor="middle" class="test-label">p = {format_p_value(p_value)} '
            f'· {format_p_stars(p_value)}</text>',
        ]
    )
    for label, group, x, color, pale_color in groups:
        values = [row_analysis_value(r) for r in group]
        q1, median, q3 = (quantile(values, p) for p in (0.25, 0.5, 0.75))
        mean = sum(values) / len(values)
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lower_whisker = min(value for value in values if value >= lower_fence)
        upper_whisker = max(value for value in values if value <= upper_fence)
        label_color = html.escape(color)
        if label.startswith("No detected"):
            first_line, second_line = "No detected", "fusion"
        else:
            first_line, second_line = "TMPRSS2–ERG", "detected"
        elements.extend(
            [
                f'<line x1="{x}" y1="{y_position(lower_whisker):.2f}" '
                f'x2="{x}" y2="{y_position(upper_whisker):.2f}" class="whisker" '
                f'stroke="{label_color}" clip-path="url(#{figure_id}-plot-clip)"/>',
                f'<line x1="{x - 8}" y1="{y_position(lower_whisker):.2f}" '
                f'x2="{x + 8}" y2="{y_position(lower_whisker):.2f}" class="whisker-cap" '
                f'stroke="{label_color}" clip-path="url(#{figure_id}-plot-clip)"/>',
                f'<line x1="{x - 8}" y1="{y_position(upper_whisker):.2f}" '
                f'x2="{x + 8}" y2="{y_position(upper_whisker):.2f}" class="whisker-cap" '
                f'stroke="{label_color}" clip-path="url(#{figure_id}-plot-clip)"/>',
                f'<rect x="{x - 20}" y="{y_position(q3):.2f}" width="40" '
                f'height="{max(y_position(q1) - y_position(q3), 1):.2f}" rx="5" '
                f'fill="{html.escape(pale_color)}" stroke="{label_color}" stroke-width="1.5" '
                f'clip-path="url(#{figure_id}-plot-clip)"/>',
                f'<line x1="{x - 24}" y1="{y_position(median):.2f}" '
                f'x2="{x + 24}" y2="{y_position(median):.2f}" class="median" '
                f'stroke="{label_color}" clip-path="url(#{figure_id}-plot-clip)"/>',
                f'<text x="{x - 31}" y="{y_position(median) - 5:.2f}" text-anchor="end" '
                f'class="median-label">median {median:.2f}</text>',
                f'<path d="M {x} {y_position(mean) - 6:.2f} L {x + 6} {y_position(mean):.2f} '
                f'L {x} {y_position(mean) + 6:.2f} L {x - 6} {y_position(mean):.2f} Z" '
                f'fill="#ffffff" stroke="{label_color}" stroke-width="2" '
                f'clip-path="url(#{figure_id}-plot-clip)"/>',
                f'<text x="{x + 31}" y="{y_position(mean) + 4:.2f}" '
                f'class="mean-label">mean {mean:.2f}</text>',
                f'<circle cx="{x}" cy="365" r="4" fill="{label_color}"/>',
                f'<text x="{x}" y="386" text-anchor="middle" class="group-label">{first_line}</text>',
                f'<text x="{x}" y="402" text-anchor="middle" class="group-label">{second_line}</text>',
                f'<text x="{x}" y="420" text-anchor="middle" class="group-count">n = {len(group)}</text>',
            ]
        )
        for record in group:
            digest = hashlib.sha256(str(record["sample_id"]).encode()).digest()
            jitter = digest[0] / 255.0 * 56.0 - 28.0
            elements.append(
                f'<circle cx="{x + jitter:.2f}" cy="{y_position(row_analysis_value(record)):.2f}" '
                f'r="2.35" class="point" fill="{label_color}" fill-opacity=".30" '
                f'clip-path="url(#{figure_id}-plot-clip)"/>'
            )
    elements.extend(["</g>", "</svg>"])
    return "".join(elements)


def render_report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    generated_at: str,
) -> None:
    cards: list[str] = []
    for study in STUDIES:
        study_summaries = [r for r in summaries if r["study_id"] == study.study_id]
        plots = []
        for result in study_summaries:
            gene = str(result["gene"])
            plots.append(
                f"""
                <article class="plot-card">
                  <div class="plot-heading">
                    <h3>{html.escape(gene)}</h3>
                    <p>Δ median = {result['median_difference']:.2f} · p = {result['p_value']:.3g} · r<sub>rb</sub> = {result['rank_biserial']:.2f}</p>
                  </div>
                  {svg_distribution(rows, study.study_id, gene)}
                </article>
                """
            )
        sample_count = len({r["sample_id"] for r in rows if r["study_id"] == study.study_id})
        fusion_count = len(
            {
                r["sample_id"]
                for r in rows
                if r["study_id"] == study.study_id and r["fusion_status"].startswith("TMPRSS2")
            }
        )
        cards.append(
            f"""
            <section class="cohort">
              <div class="cohort-heading">
                <div><span>Public cohort</span><h2>{html.escape(study.label)}</h2></div>
                <p><strong>{sample_count}</strong> samples · <strong>{fusion_count}</strong> with detected TMPRSS2–ERG</p>
              </div>
              <div class="plot-grid">{''.join(plots)}</div>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TMPRSS2–ERG cohort analysis</title>
  <style>
    :root {{ --ink:#12263a; --muted:#607181; --blue:#145da0; --cyan:#33b8c4; --pale:#eef6f8; --line:#d8e2e8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); font:15px/1.55 system-ui,sans-serif; background:#f7fafb; }}
    header, main {{ width:min(1160px,calc(100% - 40px)); margin:auto; }}
    header {{ padding:64px 0 38px; }}
    .eyebrow, .cohort-heading span {{ color:var(--blue); font-size:12px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }}
    h1 {{ max-width:850px; margin:.25em 0; font-size:clamp(36px,6vw,72px); line-height:1.02; letter-spacing:-.045em; }}
    header p {{ max-width:780px; color:var(--muted); font-size:18px; }}
    .cohort {{ margin:0 0 56px; }}
    .cohort-heading {{ display:flex; align-items:end; justify-content:space-between; gap:24px; margin-bottom:18px; }}
    h2 {{ margin:.15em 0 0; font-size:30px; }}
    .cohort-heading p {{ margin:0; color:var(--muted); }}
    .plot-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
    .plot-card {{ padding:22px; background:#fff; border:1px solid var(--line); border-radius:14px; box-shadow:0 8px 30px rgba(18,38,58,.05); }}
    .plot-heading {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; }}
    h3 {{ margin:0; font-size:22px; }}
    .plot-heading p {{ margin:0; color:var(--muted); font-size:12px; }}
    svg {{ width:100%; height:auto; overflow:visible; margin-top:8px; }}
    .axis,.whisker {{ stroke:#9db0bc; stroke-width:1.2; }}
    .box {{ fill:var(--pale); stroke:var(--blue); stroke-width:1.3; }}
    .median {{ stroke:var(--blue); stroke-width:2.2; }}
    .point {{ fill:var(--cyan); fill-opacity:.34; }}
    .tick,.group-label {{ fill:var(--muted); font-size:11px; }}
    footer {{ margin-top:18px; padding:28px 0 70px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }}
    code {{ color:var(--blue); }}
    @media(max-width:760px) {{ .plot-grid {{ grid-template-columns:1fr; }} .cohort-heading {{ align-items:start; flex-direction:column; }} }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Reproducible Python analysis · public data</div>
    <h1>TMPRSS2–ERG and the RAI2–CtBP axis</h1>
    <p>Expression distributions are shown after log₂(value + 1) transformation. Groups are compared with a two-sided Mann–Whitney U test; q-values use Benjamini–Hochberg correction within each cohort.</p>
  </header>
  <main>
    {''.join(cards)}
    <footer>Generated {html.escape(generated_at)} from the public cBioPortal REST API. The statistical figures reproduce the supplied R workflow, including its TCGA decimal-stripping conversion and SU2C 2019 capture-profile grouping. Correct direct log₂ values are retained separately for downstream machine learning.</footer>
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def run(output_dir: Path, cache_dir: Path, api_base: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = CbioPortalClient(api_base, cache_dir)
    gene_names = fetch_gene_ids(client)
    rows = [
        row
        for study in STUDIES
        for row in fetch_study_rows(client, study, gene_names)
    ]
    summaries = summarize(rows)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    write_csv(
        output_dir / "expression_values.csv",
        rows,
        (
            "study",
            "study_id",
            "sample_id",
            "gene",
            "expression",
            "log2_expression",
            "thesis_analysis_value",
            "fusion_status",
        ),
    )
    write_csv(
        output_dir / "summary.csv",
        summaries,
        (
            "study",
            "study_id",
            "gene",
            "n_fusion",
            "n_not_detected",
            "median_fusion",
            "median_not_detected",
            "median_difference",
            "u_statistic",
            "p_value",
            "q_value_bh",
            "rank_biserial",
        ),
    )
    render_report(output_dir / "report.html", rows, summaries, generated_at)
    metadata = {
        "generated_at": generated_at,
        "api_base": api_base,
        "genes": list(GENES),
        "studies": [study.__dict__ for study in STUDIES],
        "reproduction_notes": {
            "tcga": "The supplied R script removes decimal points before numeric conversion; thesis_analysis_value reproduces that rule.",
            "su2c": "Uses prad_su2c_2019 capture FPKM data and ETS clinical grouping; named non-TMPRSS2-ERG fusions are excluded.",
            "machine_learning": "log2_expression retains the corrected direct log2(value + 1) transformation.",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows):,} expression records and {len(summaries)} comparisons")
    print(f"Report: {output_dir / 'report.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--api-base", default=API_BASE)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output_dir, arguments.cache_dir, arguments.api_base)
