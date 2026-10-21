from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from data_pipeline import (
    FORBIDDEN_EXPRESSION_GENES,
    assemble_rows,
    is_tmprss2_erg_fusion,
    read_gene_panel,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class GenePanelTests(unittest.TestCase):
    def test_checked_in_panel_is_large_and_excludes_direct_markers(self) -> None:
        genes = read_gene_panel(PROJECT_DIR / "gene_panel.txt")
        self.assertEqual(len(genes), 92)
        self.assertFalse(set(genes) & FORBIDDEN_EXPRESSION_GENES)
        self.assertEqual(len(genes), len(set(genes)))

    def test_direct_marker_in_panel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory) / "panel.txt"
            panel.write_text("ERG\n" + "\n".join(f"GENE{x}" for x in range(10)))
            with self.assertRaisesRegex(ValueError, "direct fusion marker"):
                read_gene_panel(panel)


class TargetDefinitionTests(unittest.TestCase):
    def test_fusion_partner_order_does_not_matter(self) -> None:
        self.assertTrue(
            is_tmprss2_erg_fusion(
                {"site1HugoSymbol": "ERG", "site2HugoSymbol": "TMPRSS2"}
            )
        )
        self.assertTrue(
            is_tmprss2_erg_fusion(
                {"site1HugoSymbol": "TMPRSS2", "site2HugoSymbol": "ERG"}
            )
        )
        self.assertFalse(
            is_tmprss2_erg_fusion(
                {"site1HugoSymbol": "ETV1", "site2HugoSymbol": "TMPRSS2"}
            )
        )

    def test_assembly_applies_labels_and_missing_clinical_values(self) -> None:
        genes = [f"G{x}" for x in range(10)]
        samples = [
            {"sample_id": "S1", "patient_id": "P1"},
            {"sample_id": "S2", "patient_id": "P2"},
        ]
        expression = {
            "S1": {gene: float(index) for index, gene in enumerate(genes)},
            "S2": {gene: float(index) for index, gene in enumerate(genes)},
        }
        rows = assemble_rows(
            samples,
            {"S1"},
            expression,
            {"P1": {"AGE": "65", "PATH_T_STAGE": "T2A"}},
            genes,
        )
        self.assertEqual([row["fusion_status"] for row in rows], [1, 0])
        self.assertEqual(rows[0]["clinical__age"], "65")
        self.assertEqual(rows[1]["clinical__path_n_stage"], "")


class PreparedDatasetTests(unittest.TestCase):
    def test_checked_in_dataset_counts_and_columns(self) -> None:
        with (PROJECT_DIR / "data" / "model_input.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 493)
        self.assertEqual(len({row["patient_id"] for row in rows}), 493)
        self.assertEqual(Counter(row["fusion_status"] for row in rows), {"0": 304, "1": 189})
        self.assertNotIn("expr__ERG", rows[0])
        self.assertNotIn("expr__TMPRSS2", rows[0])
        self.assertNotIn("clinical__subtype", rows[0])


if __name__ == "__main__":
    unittest.main()
