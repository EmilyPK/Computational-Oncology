import csv
import tempfile
import unittest
from pathlib import Path

from ml_extension import (
    EXTERNAL_STUDY_ID,
    ML_VALUE_COLUMN,
    PRIMARY_FEATURES,
    Sample,
    TRAIN_STUDY_ID,
    binary_metrics,
    fit_standardizer,
    load_samples,
    roc_auc,
    stratified_development_split,
    stratified_split,
)


class DataPreparationTests(unittest.TestCase):
    def test_primary_features_exclude_direct_erg_marker(self):
        self.assertNotIn("ERG", PRIMARY_FEATURES)
        self.assertEqual(PRIMARY_FEATURES, ("RAI2", "CTBP1", "CTBP2"))
        self.assertEqual(ML_VALUE_COLUMN, "log2_expression")
        self.assertEqual(TRAIN_STUDY_ID, "prad_tcga_pan_can_atlas_2018")
        self.assertEqual(EXTERNAL_STUDY_ID, "prad_su2c_2019")

    def test_load_samples_pivots_tidy_rows_in_feature_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expression.csv"
            fields = (
                "study",
                "study_id",
                "sample_id",
                "gene",
                "expression",
                "log2_expression",
                "thesis_analysis_value",
                "fusion_status",
            )
            rows = [
                ("Study", "study", "sample-1", "CTBP1", "3", "2", "92", "not detected"),
                ("Study", "study", "sample-1", "RAI2", "1", "1", "91", "not detected"),
                ("Study", "study", "sample-1", "CTBP2", "7", "3", "93", "not detected"),
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            samples = load_samples(path, PRIMARY_FEATURES)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].values, (1.0, 2.0, 3.0))
        self.assertEqual(samples[0].label, 0)

    def test_standardizer_uses_supplied_training_samples(self):
        samples = [
            Sample("S", "s", "a", 0, (1.0, 10.0, 3.0)),
            Sample("S", "s", "b", 1, (3.0, 14.0, 3.0)),
        ]
        scaler = fit_standardizer(samples, PRIMARY_FEATURES)
        self.assertEqual(scaler.means, (2.0, 12.0, 3.0))
        self.assertEqual(scaler.standard_deviations, (1.0, 2.0, 1e-12))
        self.assertEqual(scaler.transform((2.0, 10.0, 3.0)), [0.0, -1.0, 0.0])

    def test_stratified_split_is_reproducible_and_keeps_both_classes(self):
        samples = [
            Sample("S", "s", f"n-{index}", 0, (float(index),) * 3)
            for index in range(10)
        ] + [
            Sample("S", "s", f"p-{index}", 1, (float(index),) * 3)
            for index in range(8)
        ]
        first = stratified_split(samples, 0.25, 17)
        second = stratified_split(samples, 0.25, 17)
        self.assertEqual(first, second)
        self.assertEqual({sample.label for sample in first[0]}, {0, 1})
        self.assertEqual({sample.label for sample in first[1]}, {0, 1})
        self.assertEqual(len(first[0]) + len(first[1]), len(samples))

    def test_development_split_has_three_disjoint_stratified_sets(self):
        samples = [
            Sample("S", "s", f"n-{index}", 0, (float(index),) * 3)
            for index in range(20)
        ] + [
            Sample("S", "s", f"p-{index}", 1, (float(index),) * 3)
            for index in range(20)
        ]
        train, validation, internal_test = stratified_development_split(
            samples, 0.2, 0.2, 42
        )
        identifiers = [
            {sample.sample_id for sample in split}
            for split in (train, validation, internal_test)
        ]
        self.assertFalse(identifiers[0] & identifiers[1])
        self.assertFalse(identifiers[0] & identifiers[2])
        self.assertFalse(identifiers[1] & identifiers[2])
        self.assertEqual(sum(map(len, identifiers)), len(samples))
        self.assertTrue(
            all(
                {sample.label for sample in split} == {0, 1}
                for split in (train, validation, internal_test)
            )
        )


class MetricTests(unittest.TestCase):
    def test_roc_auc_is_one_for_perfect_ranking(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_binary_metrics_report_expected_confusion_matrix(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9])
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
