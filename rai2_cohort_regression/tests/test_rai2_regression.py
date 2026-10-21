import csv
import tempfile
import unittest
from pathlib import Path

from rai2_regression import (
    FUSION_FEATURE,
    ML_VALUE_COLUMN,
    MODEL_SPECS,
    TARGET_GENE,
    RegressionSample,
    fit_feature_standardizer,
    fit_target_standardizer,
    load_regression_samples,
    regression_metrics,
    stratified_kfold,
)


def make_sample(sample_id, fusion_status, erg, rai2, ctbp1, ctbp2):
    return RegressionSample(
        study="Study",
        study_id="study",
        sample_id=sample_id,
        fusion_status=fusion_status,
        expression_values=(erg, rai2, ctbp1, ctbp2),
    )


class DataPreparationTests(unittest.TestCase):
    def test_primary_model_predicts_rai2_without_using_rai2_as_a_feature(self):
        self.assertEqual(TARGET_GENE, "RAI2")
        self.assertEqual(
            MODEL_SPECS["fusion_ctbp_primary"],
            (FUSION_FEATURE, "CTBP1", "CTBP2"),
        )
        self.assertNotIn(TARGET_GENE, MODEL_SPECS["fusion_ctbp_primary"])
        self.assertEqual(ML_VALUE_COLUMN, "log2_expression")

    def test_load_regression_samples_pivots_complete_expression_rows(self):
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
                ("Study", "study", "sample-1", "ERG", "1", "1", "91", "not detected"),
                ("Study", "study", "sample-1", "CTBP2", "7", "3", "93", "not detected"),
                ("Study", "study", "sample-1", "RAI2", "15", "4", "94", "not detected"),
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            samples = load_regression_samples(path)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].expression_values, (1.0, 4.0, 2.0, 3.0))
        self.assertEqual(samples[0].target, 4.0)
        self.assertEqual(samples[0].fusion_status, 0)

    def test_feature_scaler_keeps_fusion_binary_and_scales_expression(self):
        samples = [
            make_sample("a", 0, 1.0, 4.0, 10.0, 3.0),
            make_sample("b", 1, 2.0, 6.0, 14.0, 3.0),
        ]
        scaler = fit_feature_standardizer(
            samples, (FUSION_FEATURE, "CTBP1", "CTBP2")
        )
        self.assertEqual(scaler.means, (0.0, 12.0, 3.0))
        self.assertEqual(scaler.standard_deviations, (1.0, 2.0, 1e-12))
        self.assertEqual(scaler.transform(samples[0]), [0.0, -1.0, 0.0])
        self.assertEqual(scaler.transform(samples[1]), [1.0, 1.0, 0.0])

    def test_target_scaler_is_fitted_from_training_samples(self):
        samples = [
            make_sample("a", 0, 1.0, 4.0, 10.0, 3.0),
            make_sample("b", 1, 2.0, 8.0, 14.0, 3.0),
        ]
        scaler = fit_target_standardizer(samples)
        self.assertEqual(scaler.mean, 6.0)
        self.assertEqual(scaler.standard_deviation, 2.0)
        self.assertEqual(scaler.inverse_transform(scaler.transform(8.0)), 8.0)

    def test_target_cannot_be_used_as_a_predictor(self):
        sample = make_sample("a", 0, 1.0, 4.0, 10.0, 3.0)
        with self.assertRaises(ValueError):
            fit_feature_standardizer([sample], (TARGET_GENE,))


class CrossValidationTests(unittest.TestCase):
    def test_stratified_folds_are_reproducible_disjoint_and_complete(self):
        samples = [
            make_sample(f"n-{index}", 0, 1.0, float(index), 2.0, 3.0)
            for index in range(10)
        ] + [
            make_sample(f"p-{index}", 1, 2.0, float(index), 3.0, 4.0)
            for index in range(10)
        ]
        first = stratified_kfold(samples, 5, 17)
        second = stratified_kfold(samples, 5, 17)
        self.assertEqual(first, second)
        identifiers = [
            {sample.sample_id for sample in fold} for fold in first
        ]
        self.assertEqual(sum(map(len, identifiers)), len(samples))
        self.assertEqual(len(set().union(*identifiers)), len(samples))
        self.assertTrue(
            all({sample.fusion_status for sample in fold} == {0, 1} for fold in first)
        )


class MetricTests(unittest.TestCase):
    def test_regression_metrics_are_perfect_for_exact_predictions(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(metrics["r_squared"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["pearson_r"], 1.0)

    def test_r_squared_is_zero_for_mean_only_predictions(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(metrics["r_squared"], 0.0)
        self.assertAlmostEqual(metrics["pearson_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
