import sys
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import (  # noqa: E402
    average_ranks,
    benjamini_hochberg,
    classify_su2c_ets,
    is_tmprss2_erg_fusion,
    legacy_r_tcga_log2,
    mann_whitney_u,
    quantile,
    svg_distribution,
)


class StatisticsTests(unittest.TestCase):
    def test_average_ranks_handles_ties(self):
        ranks, tie_sizes = average_ranks([2.0, 1.0, 2.0, 4.0])
        self.assertEqual(ranks, [2.5, 1.0, 2.5, 4.0])
        self.assertEqual(tie_sizes, [2])

    def test_mann_whitney_detects_separated_groups(self):
        u_statistic, p_value, effect = mann_whitney_u(
            [8.0, 9.0, 10.0, 11.0, 12.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(u_statistic, 25.0)
        self.assertLess(p_value, 0.02)
        self.assertEqual(effect, 1.0)

    def test_quantile_uses_linear_interpolation(self):
        self.assertEqual(quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_benjamini_hochberg_is_monotone_in_rank(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
        self.assertAlmostEqual(adjusted[0], 0.04)
        self.assertAlmostEqual(adjusted[1], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[2], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[3], 0.20)


class VariantTests(unittest.TestCase):
    def test_tmprss2_erg_is_orientation_independent(self):
        forward = {"site1HugoSymbol": "TMPRSS2", "site2HugoSymbol": "ERG"}
        reverse = {"site1HugoSymbol": "ERG", "site2HugoSymbol": "TMPRSS2"}
        other = {"site1HugoSymbol": "TMPRSS2", "site2HugoSymbol": "ETV1"}
        self.assertTrue(is_tmprss2_erg_fusion(forward))
        self.assertTrue(is_tmprss2_erg_fusion(reverse))
        self.assertFalse(is_tmprss2_erg_fusion(other))

    def test_su2c_clinical_grouping_matches_supplied_r_workflow(self):
        self.assertEqual(
            classify_su2c_ets("Positive", "TMPRSS2-ERG"),
            "TMPRSS2-ERG detected",
        )
        self.assertEqual(classify_su2c_ets("Negative", None), "not detected")
        self.assertEqual(classify_su2c_ets("Positive", None), "not detected")
        self.assertIsNone(classify_su2c_ets("Positive", "SLC45A3-ERG"))
        self.assertIsNone(classify_su2c_ets(None, None))

    def test_legacy_tcga_conversion_removes_decimal_point(self):
        self.assertAlmostEqual(legacy_r_tcga_log2(12.34), math.log2(1234 + 1))


class FigureTests(unittest.TestCase):
    def test_distribution_figure_contains_mean_median_and_test_result(self):
        rows = []
        for status, values in (
            ("not detected", [1.0, 2.0, 3.0, 4.0]),
            ("TMPRSS2-ERG detected", [4.0, 5.0, 6.0, 7.0]),
        ):
            for index, value in enumerate(values):
                rows.append(
                    {
                        "study_id": "prad_tcga_pan_can_atlas_2018",
                        "sample_id": f"{status}-{index}",
                        "gene": "RAI2",
                        "log2_expression": value,
                        "fusion_status": status,
                    }
                )

        figure = svg_distribution(rows, "prad_tcga_pan_can_atlas_2018", "RAI2")

        ET.fromstring(figure)
        self.assertIn("median 2.50", figure)
        self.assertIn("mean 2.50", figure)
        self.assertIn("Mann–Whitney U test", figure)
        self.assertIn("p = ", figure)
        self.assertIn("· *", figure)
        self.assertIn('class="significance-bracket"', figure)
        self.assertIn('transform="rotate(-90', figure)
        self.assertIn(">10<", figure)
        self.assertIn(">25<", figure)
        self.assertIn("plot-clip", figure)


if __name__ == "__main__":
    unittest.main()
