"""Pruebas del contrato temporal V9. No requieren TensorFlow para validar causalidad."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from lstm_residual_estacional import (
    HORIZON,
    _build_features,
    _training_pairs,
    _baseline_predictions,
    _candidate_grid,
    select_seasonal_residual,
    evaluate_seasonal_residual,
    ResidualCandidate,
)


class DirectTemporalTests(unittest.TestCase):
    def test_input_has_only_pre_origin_information(self):
        y = 220 + np.arange(190, dtype=float) * 0.8 + 20 * np.sin(np.arange(190) / 7.0)
        a = _build_features(y, 120, 7, 8, 5.0, 0.9)
        modified = y.copy()
        modified[120:] = 999999.0
        b = _build_features(modified, 120, 7, 8, 5.0, 0.9)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(a[0].shape, (8, 6))
        self.assertEqual(a[1].shape, (15,))

    def test_training_pairs_keep_full_target_in_training(self):
        y = np.arange(1, 151, dtype=float)
        seq, ctx, base, targets, metadata = _training_pairs(y, 120, 8, 3.1, 0.5)
        self.assertEqual(seq.shape[1:], (8, 6))
        self.assertEqual(ctx.shape[1], 15)
        self.assertEqual(len(seq), len(ctx))
        self.assertEqual(len(seq), len(base))
        self.assertEqual(len(seq), len(targets))
        self.assertEqual(max(x[2] for x in metadata), 119)
        self.assertEqual(min(x[1] for x in metadata), 1)
        self.assertEqual(max(x[1] for x in metadata), 12)

    def test_seasonal_reference_does_not_use_hidden_targets(self):
        y = np.arange(1, 130, dtype=float)
        origins = np.full(12, 100)
        steps = np.arange(1, 13)
        expected = y[48:60]
        np.testing.assert_array_equal(_baseline_predictions(y, origins, steps, 52), expected)
        modified = y.copy()
        modified[100:] = 1000000
        np.testing.assert_array_equal(
            _baseline_predictions(modified, origins, steps, 52), expected)

    def test_test_is_never_used_to_select_or_calibrate(self):
        class Settings:
            train_ratio = .70
            val_ratio = .15
            lookback = 12
            units = 16
            epochs = 20
            patience = 4
            batch_size = 16
            dropout = .15
            seed = 42

        y = (300 + np.arange(200) * 0.2).astype(float)
        series = pd.DataFrame({
            "fecha_inicio": pd.date_range("2023-01-02", periods=200, freq="7D"),
            "valor": y,
        })

        def fake_fit(values, end, lb, units, seeds, *_args, **_kwargs):
            return [{"lookback":lb, "units":units, "epochs":7, "seed":seed,
                     "smear":1.0} for seed in seeds]

        def fake_predict(_members, values, origins, steps, bias=0.0):
            return _baseline_predictions(values, origins, steps, 52) + bias

        with patch("lstm_residual_estacional._fit_ensemble", side_effect=fake_fit), \
             patch("lstm_residual_estacional._predict_members", side_effect=fake_predict):
            a = select_seasonal_residual(series, Settings())
            changed = series.copy()
            changed.loc[170:, "valor"] += 1000.0
            b = select_seasonal_residual(changed, Settings())
        self.assertEqual(a["selected"], b["selected"])
        self.assertEqual(a["bias_correction"], b["bias_correction"])
        pd.testing.assert_frame_equal(a["validation"], b["validation"])
        self.assertIsNone(a["test"])
        self.assertEqual(len(a["candidate_grid"]), 2)
        self.assertEqual(set(a["candidates"]), {a["selected"]})

    def test_test_calculates_fixed_and_rolling_with_frozen_candidate(self):
        class Settings:
            train_ratio = .70
            val_ratio = .15
            lookback = 8
            units = 8
            epochs = 20
            patience = 4
            batch_size = 16
            dropout = .15
            seed = 42

        y = 300 + np.arange(200) * 0.3
        series = pd.DataFrame({
            "fecha_inicio": pd.date_range("2023-01-02", periods=200, freq="7D"),
            "valor": y,
        })
        candidate = ResidualCandidate(
            model=[{"smear":1.0}], lookback=8, period=52,
            units=8, epochs=3, seeds=(42,), member_epochs=(3,)
        )
        selection = {
            "selected": "LSTM candidata",
            "series": series,
            "train_end": 140,
            "validation_end": 170,
            "candidates": {"LSTM candidata": candidate},
            "settings": Settings(),
            "test": None,
        }
        with patch("lstm_residual_estacional._fit_fixed_ensemble",
                   return_value=[{"smear":1.0}]), \
             patch("lstm_residual_estacional._predict_members",
                   side_effect=lambda members, y, origins, steps, bias=0.0:
                   _baseline_predictions(y, origins, steps, 52) + bias):
            table = evaluate_seasonal_residual(selection)
        self.assertEqual(
            int(table.loc[table.Modelo == "LSTM candidata", "Orígenes"].iloc[0]), 19
        )
        self.assertEqual(
            len(selection["test_predictions"]), 19 * HORIZON
        )
        self.assertEqual(len(selection["fixed_test_predictions"]), HORIZON)
        self.assertEqual(
            selection["fixed_test_origin_date"], series.fecha_inicio.iloc[170].date()
        )
        self.assertEqual(
            set(table.Modelo),
            {"LSTM candidata", "Naive estacional 48", "Naive estacional 52"},
        )


if __name__ == "__main__":
    unittest.main()
