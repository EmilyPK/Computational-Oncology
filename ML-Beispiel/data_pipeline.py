#!/usr/bin/env python3
"""Prepare the TCGA table used by the fusion-prediction example.

The target is TMPRSS2--ERG status. ERG and TMPRSS2 expression are excluded from
the predictors. Source data come from cBioPortal and are cached locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


API_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "prad_tcga_pan_can_atlas_2018"
EXPRESSION_PROFILE_ID = f"{STUDY_ID}_rna_seq_v2_mrna"
EXPRESSION_SAMPLE_LIST_ID = EXPRESSION_PROFILE_ID
STRUCTURAL_VARIANT_PROFILE_ID = f"{STUDY_ID}_structural_variants"
STRUCTURAL_VARIANT_SAMPLE_LIST_ID = f"{STUDY_ID}_sv"
CLINICAL_ATTRIBUTES = ("AGE", "PATH_T_STAGE", "PATH_N_STAGE")
FORBIDDEN_EXPRESSION_GENES = frozenset({"ERG", "TMPRSS2"})
FORBIDDEN_CLINICAL_ATTRIBUTES = frozenset({"SUBTYPE"})
PROJECT_VERSION = "1.0.0"


class CbioPortalClient:
    """Small cBioPortal client with a local JSON cache."""

    def __init__(self, base_url: str, cache_dir: Path, retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.ssl_context = _certificate_context()

    def get(self, path: str, query: dict[str, Any] | None = None) -> Any:
        suffix = "?" + urllib.parse.urlencode(query) if query else ""
        return self._request("GET", path + suffix, None)

    def post(
        self,
        path: str,
        payload: Any,
        query: dict[str, Any] | None = None,
    ) -> Any:
        suffix = "?" + urllib.parse.urlencode(query) if query else ""
        return self._request("POST", path + suffix, payload)

    def _request(self, method: str, path: str, payload: Any) -> Any:
        body = None if payload is None else json.dumps(payload, sort_keys=True).encode()
        key = hashlib.sha256(
            method.encode() + b"\0" + path.encode() + b"\0" + (body or b"")
        ).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": f"tcga-fusion-ml-example/{PROJECT_VERSION}",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=90,
                    context=self.ssl_context,
                ) as response:
                    result = json.load(response)
                cache_path.write_text(json.dumps(result), encoding="utf-8")
                return result
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"cBioPortal request failed after {self.retries} attempts: {path}"
        ) from last_error


def _certificate_context() -> ssl.SSLContext:
    """Prefer certifi when available, fixing common macOS Python CA issues."""
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def batched(values: Sequence[int], size: int) -> Iterable[list[int]]:
    if size < 1:
        raise ValueError("Batch size must be positive")
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def read_gene_panel(path: Path) -> list[str]:
    """Read unique HUGO symbols while allowing comments in the panel file."""
    symbols: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        symbol = raw_line.split("#", 1)[0].strip().upper()
        if not symbol:
            continue
        if not symbol.replace("-", "").isalnum():
            raise ValueError(f"Invalid gene symbol at {path}:{line_number}: {symbol}")
        if symbol in FORBIDDEN_EXPRESSION_GENES:
            raise ValueError(
                f"{symbol} is a direct fusion marker and must not be a predictor"
            )
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    if len(symbols) < 10:
        raise ValueError("The expression panel must contain at least ten genes")
    return symbols


def is_tmprss2_erg_fusion(record: dict[str, Any]) -> bool:
    partners = {
        str(record.get("site1HugoSymbol", "")).upper(),
        str(record.get("site2HugoSymbol", "")).upper(),
    }
    return partners == {"TMPRSS2", "ERG"}


def resolve_gene_ids(
    client: CbioPortalClient,
    symbols: Sequence[str],
) -> dict[int, str]:
    records = client.post(
        "/genes/fetch",
        list(symbols),
        query={"geneIdType": "HUGO_GENE_SYMBOL", "projection": "SUMMARY"},
    )
    mapping = {
        int(record["entrezGeneId"]): str(record["hugoGeneSymbol"]).upper()
        for record in records
        if str(record.get("hugoGeneSymbol", "")).upper() in set(symbols)
    }
    missing = sorted(set(symbols) - set(mapping.values()))
    if missing:
        raise RuntimeError(f"Genes not resolved by cBioPortal: {', '.join(missing)}")
    return mapping


def fetch_samples(client: CbioPortalClient) -> list[dict[str, str]]:
    expression_samples = client.post(
        "/samples/fetch",
        {"sampleListIds": [EXPRESSION_SAMPLE_LIST_ID]},
        query={"projection": "SUMMARY"},
    )
    sv_samples = client.post(
        "/samples/fetch",
        {"sampleListIds": [STRUCTURAL_VARIANT_SAMPLE_LIST_ID]},
        query={"projection": "SUMMARY"},
    )
    sv_ids = {str(record["sampleId"]) for record in sv_samples}
    samples = [
        {
            "sample_id": str(record["sampleId"]),
            "patient_id": str(record["patientId"]),
        }
        for record in expression_samples
        if str(record["sampleId"]) in sv_ids
    ]
    if not samples:
        raise RuntimeError("No samples have both expression and structural-variant data")
    return sorted(samples, key=lambda record: record["sample_id"])


def fetch_fusion_labels(client: CbioPortalClient) -> set[str]:
    records = client.post(
        "/structural-variant/fetch",
        {"molecularProfileIds": [STRUCTURAL_VARIANT_PROFILE_ID]},
    )
    return {
        str(record["sampleId"])
        for record in records
        if is_tmprss2_erg_fusion(record)
    }


def fetch_expression(
    client: CbioPortalClient,
    gene_ids: dict[int, str],
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Fetch batch-normalized RSEM values and apply log2(value + 1).

    cBioPortal contains a handful of slightly negative batch-corrected values;
    they remain mathematically valid as long as they are greater than -1.
    """
    by_sample: dict[str, dict[str, float]] = {}
    ids = sorted(gene_ids)
    for gene_batch in batched(ids, batch_size):
        records = client.post(
            "/molecular-data/fetch",
            {
                "molecularProfileIds": [EXPRESSION_PROFILE_ID],
                "entrezGeneIds": gene_batch,
            },
            query={"projection": "SUMMARY"},
        )
        for record in records:
            value = float(record["value"])
            if value <= -1:
                raise ValueError("Expression values must be greater than -1 for log2(value + 1)")
            symbol = gene_ids[int(record["entrezGeneId"])]
            by_sample.setdefault(str(record["sampleId"]), {})[symbol] = math.log2(
                value + 1.0
            )
    return by_sample


def fetch_patient_clinical_data(
    client: CbioPortalClient,
    patient_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    if set(CLINICAL_ATTRIBUTES) & FORBIDDEN_CLINICAL_ATTRIBUTES:
        raise RuntimeError("A target-derived clinical attribute was configured")
    records = client.post(
        f"/studies/{STUDY_ID}/clinical-data/fetch",
        {"ids": sorted(set(patient_ids)), "attributeIds": list(CLINICAL_ATTRIBUTES)},
        query={"clinicalDataType": "PATIENT", "projection": "SUMMARY"},
    )
    by_patient: dict[str, dict[str, str]] = {}
    for record in records:
        by_patient.setdefault(str(record["patientId"]), {})[
            str(record["clinicalAttributeId"])
        ] = str(record["value"])
    return by_patient


def assemble_rows(
    samples: Sequence[dict[str, str]],
    fusion_positive_samples: set[str],
    expression: dict[str, dict[str, float]],
    clinical: dict[str, dict[str, str]],
    gene_symbols: Sequence[str],
    minimum_expression_fraction: float = 0.90,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    minimum_count = math.ceil(minimum_expression_fraction * len(gene_symbols))
    for sample in samples:
        sample_id = sample["sample_id"]
        patient_id = sample["patient_id"]
        values = expression.get(sample_id, {})
        if len(values) < minimum_count:
            continue
        patient_values = clinical.get(patient_id, {})
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "fusion_status": int(sample_id in fusion_positive_samples),
            "clinical__age": patient_values.get("AGE", ""),
            "clinical__path_t_stage": patient_values.get("PATH_T_STAGE", ""),
            "clinical__path_n_stage": patient_values.get("PATH_N_STAGE", ""),
        }
        for symbol in gene_symbols:
            row[f"expr__{symbol}"] = values.get(symbol, "")
        rows.append(row)
    if not rows:
        raise RuntimeError("No complete modelling rows could be assembled")
    if len({row["fusion_status"] for row in rows}) != 2:
        raise RuntimeError("The modelling table must contain both target classes")
    return rows


def write_dataset(
    rows: Sequence[dict[str, Any]],
    gene_symbols: Sequence[str],
    output_dir: Path,
    gene_panel_path: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_input.csv"
    fields = [
        "sample_id",
        "patient_id",
        "fusion_status",
        "clinical__age",
        "clinical__path_t_stage",
        "clinical__path_n_stage",
        *[f"expr__{symbol}" for symbol in gene_symbols],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    class_counts = Counter(int(row["fusion_status"]) for row in rows)
    missing_counts = {
        field: sum(row.get(field, "") == "" for row in rows)
        for field in fields
        if field.startswith(("clinical__", "expr__"))
    }
    manifest = {
        "project_version": PROJECT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "api": API_BASE,
            "study_id": STUDY_ID,
            "expression_profile_id": EXPRESSION_PROFILE_ID,
            "structural_variant_profile_id": STRUCTURAL_VARIANT_PROFILE_ID,
        },
        "target": {
            "column": "fusion_status",
            "definition": "TMPRSS2--ERG partner pair in structural-variant data",
            "negative": 0,
            "positive": 1,
        },
        "leakage_controls": {
            "excluded_expression_genes": sorted(FORBIDDEN_EXPRESSION_GENES),
            "excluded_clinical_attributes": sorted(FORBIDDEN_CLINICAL_ATTRIBUTES),
            "feature_selection": "performed within development folds by modelling.py",
            "test_set": "assigned at patient level and untouched until final evaluation",
        },
        "expression_transform": "log2(batch-normalized RSEM + 1)",
        "gene_panel_file": gene_panel_path.name,
        "gene_count": len(gene_symbols),
        "genes": list(gene_symbols),
        "sample_count": len(rows),
        "patient_count": len({str(row["patient_id"]) for row in rows}),
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "missing_counts": missing_counts,
    }
    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return csv_path, manifest_path


def prepare_dataset(
    output_dir: Path,
    cache_dir: Path,
    gene_panel_path: Path,
    batch_size: int = 40,
) -> tuple[Path, Path]:
    genes = read_gene_panel(gene_panel_path)
    client = CbioPortalClient(API_BASE, cache_dir)
    gene_ids = resolve_gene_ids(client, genes)
    samples = fetch_samples(client)
    positives = fetch_fusion_labels(client)
    expression = fetch_expression(client, gene_ids, batch_size=batch_size)
    clinical = fetch_patient_clinical_data(
        client,
        [sample["patient_id"] for sample in samples],
    )
    rows = assemble_rows(samples, positives, expression, clinical, genes)
    return write_dataset(rows, genes, output_dir, gene_panel_path)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--cache-dir", type=Path, default=project_dir / ".cache")
    parser.add_argument(
        "--gene-panel",
        type=Path,
        default=project_dir / "gene_panel.txt",
    )
    parser.add_argument("--batch-size", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, manifest_path = prepare_dataset(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        gene_panel_path=args.gene_panel,
        batch_size=args.batch_size,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
