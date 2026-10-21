from __future__ import annotations

import unittest
from collections import Counter

from modeling import (
    SplitRecord,
    choose_balanced_threshold,
    stratified_group_folds,
    stratified_group_holdout,
    threshold_metrics,
)


def example_records() -> list[SplitRecord]:
    records = []
    index = 0
    for label, patients in ((0, 30), (1, 20)):
        for patient in range(patients):
            patient_id = f"P{label}-{patient:02d}"
            records.append(SplitRecord(index, patient_id, label))
            index += 1
    return records


class SplitTests(unittest.TestCase):
    def test_holdout_is_reproducible_stratified_and_disjoint(self) -> None:
        records = example_records()
        development, test = stratified_group_holdout(records, 0.20, seed=7)
        self.assertEqual((development, test), stratified_group_holdout(records, 0.20, seed=7))
        development_records = [record for record in records if record.row_index in development]
        test_records = [record for record in records if record.row_index in test]
        self.assertFalse(
            {record.patient_id for record in development_records}
            & {record.patient_id for record in test_records}
        )
        self.assertEqual(Counter(record.label for record in test_records), {0: 6, 1: 4})

    def test_each_record_is_validation_once(self) -> None:
        records = example_records()
        folds = stratified_group_folds(records, number_of_folds=5, seed=7)
        validation = [index for _, fold in folds for index in fold]
        self.assertEqual(sorted(validation), list(range(len(records))))
        for train, fold in folds:
            self.assertFalse(set(train) & set(fold))
            labels = Counter(records[index].label for index in fold)
            self.assertEqual(labels, {0: 6, 1: 4})

    def test_groups_with_repeated_rows_never_cross_a_fold(self) -> None:
        records = example_records()
        records.append(SplitRecord(len(records), records[0].patient_id, records[0].label))
        for train, validation in stratified_group_folds(records, 5, seed=3):
            train_patients = {records[index].patient_id for index in train}
            validation_patients = {records[index].patient_id for index in validation}
            self.assertFalse(train_patients & validation_patients)


class MetricTests(unittest.TestCase):
    def test_threshold_metrics(self) -> None:
        metrics = threshold_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], 0.5)
        self.assertAlmostEqual(metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)

    def test_threshold_is_selected_without_test_data(self) -> None:
        labels = [0, 0, 0, 1, 1, 1]
        probabilities = [0.05, 0.20, 0.40, 0.55, 0.70, 0.90]
        threshold = choose_balanced_threshold(labels, probabilities)
        # Both 0.50 and 0.55 separate these examples perfectly; the deterministic
        # tie-break deliberately prefers the threshold closest to 0.50.
        self.assertEqual(threshold, 0.5)
        self.assertEqual(
            threshold_metrics(labels, probabilities, threshold)["balanced_accuracy"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
