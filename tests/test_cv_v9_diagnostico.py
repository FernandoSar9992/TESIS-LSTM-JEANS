"""Pruebas aisladas del diseño temporal CV-V9; no requieren TensorFlow."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cv_v9_diagnostico import (
    AuditConfig, expanding_folds, evaluate_cross_validation,
)


class TestCrossValidationV9(unittest.TestCase):
    def test_folds_are_disjoint_and_before_test(self):
        n = 156
        folds = expanding_folds(n, max_folds=3)
        test_start = int(n * (.70 + .15))
        self.assertEqual(len(folds), 3)
        for i, (start, stop) in enumerate(folds):
            self.assertEqual(stop - start, 12)
            self.assertLessEqual(stop, test_start)
            self.assertGreaterEqual(start, 84)
            if i:
                self.assertEqual(folds[i-1][1], start)

    def test_short_series_does_not_fabricate_more_folds(self):
        folds = expanding_folds(113, max_folds=3)
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0], (84, 96))
        with self.assertRaises(ValueError):
            expanding_folds(105, max_folds=3)

    def test_test_data_cannot_change_cv_metrics(self):
        n = 156
        y = 300 + .3 * np.arange(n) + 24 * np.sin(np.arange(n) / 6.)
        series = pd.DataFrame({
            "fecha_inicio": pd.date_range("2023-01-02", periods=n, freq="7D"),
            "valor": y,
        })
        test_start = int(n * .85)

        def fake_fit(values, fit_end, lookback, units, seed, config):
            self.assertLess(fit_end, test_start)
            member = {
                "lookback": lookback, "units": units, "seed": seed,
                "epochs": 8, "smear": 1.0,
            }
            curve = pd.DataFrame({
                "Época": [1, 2],
                "loss_train": [1.0, .7],
                "loss_validacion_interna": [.9, .8],
                "Época_óptima": [2, 2],
            })
            return member, curve

        def fake_refit(values, fit_end, lookback, units, seed, epochs, **kwargs):
            self.assertLess(fit_end, test_start)
            return {
                "lookback": lookback, "units": units, "seed": seed,
                "epochs": epochs, "smear": kwargs["smear"],
            }

        def fake_predict(members, values, origins, steps, bias=0.0):
            values = np.asarray(values, dtype=float)
            return np.asarray([
                values[int(origin)-1] + bias
                for origin in np.asarray(origins)
            ])

        mocks = (
            patch("cv_v9_diagnostico._train_member_with_curve",
                  side_effect=fake_fit),
            patch("cv_v9_diagnostico._fit_fixed_member",
                  side_effect=fake_refit),
            patch("cv_v9_diagnostico._predict_members",
                  side_effect=fake_predict),
        )
        with mocks[0], mocks[1], mocks[2]:
            first = evaluate_cross_validation(series)
            changed = series.copy()
            changed.loc[test_start:, "valor"] = 1000000.
            second = evaluate_cross_validation(changed)

        pd.testing.assert_frame_equal(first["folds"], second["folds"])
        pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
        pd.testing.assert_frame_equal(first["candidates"], second["candidates"])
        self.assertFalse(bool(first["summary"].iloc[0]["TEST_consultado"]))
        self.assertEqual(len(first["folds"]), 3)
        self.assertEqual(len(first["predictions"]), 36)


if __name__ == "__main__":
    unittest.main()
