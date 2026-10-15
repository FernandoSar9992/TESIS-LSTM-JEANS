"""Regresiones de fuga temporal, selección y calidad de datos; unittest estándar."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast_engine import (ForecastConfig, _features, _future_lstm_input,
                             _lstm_arrays, clean_series, future_labels,
                             metrics, run_experiment, suggested_seasonal_period)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.y = 100 + np.random.default_rng(42).normal(0, 10, 156)
        self.data = pd.DataFrame({"periodo": [str(i) for i in range(156)], "demanda": self.y})
        self.config = ForecastConfig()
        self.models = ["Ridge directa", "Último valor", "Media móvil", "Media histórica", "Estacional ingenuo"]

    def test_future_extreme_cannot_change_training_scale_or_windows(self):
        original = _lstm_arrays(self.y, self.config, 109)
        changed_y = self.y.copy()
        changed_y[109:] = 100000
        changed = _lstm_arrays(changed_y, self.config, 109)
        self.assertEqual(original[3:], changed[3:])
        train = original[2] + self.config.horizon <= 109
        np.testing.assert_array_equal(original[0][train], changed[0][train])
        np.testing.assert_array_equal(original[1][train], changed[1][train])
        self.assertGreater(changed[1].max(), 1)  # no reescalado/recorte usando el futuro

    def test_lstm_refits_use_correct_scalers_windows_and_exact_best_epoch(self):
        fits = []

        class FakeModel:
            def fit(self, x, y, **kwargs):
                fits.append((len(x), kwargs))
                return SimpleNamespace(history={"loss": [0.2, 0.1], "val_loss": [0.1, 0.2]})

            def predict(self, x, **kwargs):
                return np.zeros((len(x), 8))

            def count_params(self):
                return 10

        fake_tf = SimpleNamespace(keras=SimpleNamespace(
            backend=SimpleNamespace(clear_session=lambda: None),
            callbacks=SimpleNamespace(EarlyStopping=lambda **kw: kw)))
        with patch.dict(sys.modules, {"tensorflow": fake_tf}), \
             patch("forecast_engine.tensorflow_status", return_value=(True, "test")), \
             patch("forecast_engine._build_lstm", side_effect=lambda *args: FakeModel()), \
             patch("forecast_engine._lstm_arrays", wraps=_lstm_arrays) as arrays:
            result = run_experiment(self.data, self.config, ["LSTM"])
        self.assertEqual([call.args[2] for call in arrays.call_args_list], [109, 132, 156])
        self.assertEqual([count for count, _ in fits], [90, 113, 137])
        self.assertEqual([kw["epochs"] for _, kw in fits], [150, 1, 1])
        self.assertTrue(all(kw["shuffle"] is False for _, kw in fits))
        self.assertEqual(result["history"]["LSTM"]["best_epochs"], 1)

    def test_final_input_matches_historical_feature_contract(self):
        x, _, _, minimum, span = _lstm_arrays(self.y, self.config, 109)
        origin = 100
        expected = _future_lstm_input(self.y[:origin], self.config, minimum, span)
        np.testing.assert_allclose(x[origin-self.config.lookback], expected[0])

    def test_test_values_cannot_select_winner_or_calibrate_band(self):
        original = run_experiment(self.data, self.config, self.models)
        mutated = self.data.copy()
        mutated.loc[132:, "demanda"] = 100000
        changed = run_experiment(mutated, self.config, self.models)
        pd.testing.assert_frame_equal(original["validation_metrics"], changed["validation_metrics"])
        self.assertEqual(original["winner"], changed["winner"])
        self.assertEqual(original["winner"], original["validation_metrics"].iloc[0]["Modelo"])
        # Anchura superior equivale al error de validación aunque cambie el ajuste futuro.
        for result in (original, changed):
            f = result["forecast"]
            result["radius"] = f["Límite superior aprox."] - f["Pronóstico"]
        np.testing.assert_allclose(original["radius"], changed["radius"])
        # Primera predicción no ve aún ninguno de los valores de prueba cambiados.
        for name in self.models:
            np.testing.assert_allclose(original["detail"].query("Origen == 132")[name], changed["detail"].query("Origen == 132")[name])

    def test_split_targets_are_disjoint_and_wholly_contained(self):
        _, _, origins = _features(self.y, 12, 52, 8)
        blocks = [(0, 109), (109, 132), (132, 156)]
        targets = []
        for lower, upper in blocks:
            accepted = origins[(origins >= lower) & (origins+8 <= upper)]
            targets.append({int(t+h) for t in accepted for h in range(8)})
        self.assertFalse(targets[0] & targets[1])
        self.assertFalse(targets[1] & targets[2])

    def test_incomplete_validation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ventanas"):
            run_experiment(self.data, ForecastConfig(horizon=26), ["LSTM"])

    def test_seasonal_baseline_cannot_wrap_negative_indices(self):
        result = run_experiment(self.data, ForecastConfig(seasonal_period=120), self.models)
        self.assertNotIn("Estacional ingenuo", result["metrics"]["Modelo"].tolist())
        self.assertIn("Estacional_error", result["history"])

    def test_zero_metrics_and_shape_validation(self):
        self.assertEqual(metrics(np.zeros(3), np.zeros(3))["sMAPE_%"], 0)
        self.assertEqual(metrics(np.array([0, 1]), np.array([0, 0]))["sMAPE_%"], 100)
        with self.assertRaises(ValueError):
            metrics(np.ones(3), np.ones(1))

    def test_calendar_and_invalid_rows(self):
        frame = pd.DataFrame({"p": ["Diciembre 2025-Se4", "Enero 2026-Se1"], "y": [1, 2]})
        data, _ = clean_series(frame, "p", "y")
        self.assertEqual(suggested_seasonal_period(data), 48)
        self.assertEqual(future_labels(data, 1), ["Enero 2026-Se2"])
        with self.assertRaisesRegex(ValueError, "ordenados"):
            clean_series(frame.iloc[::-1], "p", "y")
        for invalid in (np.nan, np.inf, -1, "mal"):
            with self.assertRaises(ValueError):
                clean_series(pd.DataFrame({"p": ["a", "b"], "y": [1, invalid]}), "p", "y")
        with self.assertRaises(ValueError):
            clean_series(pd.DataFrame({"p": ["a", "a"], "y": [1, 2]}), "p", "y")

    def test_reported_metrics_match_manual_errors(self):
        result = run_experiment(self.data, self.config, self.models)
        for name in self.models:
            detail = result["detail"]
            expected = np.sqrt(np.mean((detail["Real"]-detail[name])**2))
            actual = result["metrics"].set_index("Modelo").loc[name, "RMSE"]
            self.assertAlmostEqual(expected, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
