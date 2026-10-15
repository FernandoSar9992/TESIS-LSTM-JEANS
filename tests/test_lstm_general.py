from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand_data import prepare_series, read_upload
from lstm_engine import (LSTMConfig, candidates_for, future_input, run_lstm,
                         scaled_windows, split_points, suggest_cycle)


class DataTests(unittest.TestCase):
    def test_arbitrary_labels_no_calendar_assumption(self):
        data = prepare_series(pd.DataFrame({"ciclo": ["Q-A", "Q-B", "Q-C"], "pedidos": [2, 4, 3]}), "ciclo", "pedidos")
        self.assertIsNone(data.frequency)
        self.assertEqual(data.future_labels(2), ["Período +1", "Período +2"])

    def test_dates_sort_infer_and_project(self):
        frame = pd.DataFrame({"fecha": pd.to_datetime(["2025-03-01", "2025-01-01", "2025-02-01"]), "y": [3, 1, 2]})
        data = prepare_series(frame, "fecha", "y", date_mode=True)
        self.assertEqual(data.data.demanda.tolist(), [1, 2, 3])
        self.assertEqual(data.future_labels(2), ["2025-04-01", "2025-05-01"])

    def test_transactions_are_aggregated_only_explicitly(self):
        frame = pd.DataFrame({"fecha": ["2025-01-01", "2025-01-01", "2025-01-03"], "y": [2, 3, 8]})
        with self.assertRaisesRegex(ValueError, "repetidas"):
            prepare_series(frame, "fecha", "y", date_mode=True)
        with self.assertRaisesRegex(ValueError, "Faltan"):
            prepare_series(frame, "fecha", "y", date_mode=True, frequency="D")
        data = prepare_series(frame, "fecha", "y", date_mode=True, frequency="D", fill_missing_zero=True)
        self.assertEqual(data.data.demanda.tolist(), [5, 0, 8])
        self.assertEqual(data.future_labels(1), ["2025-01-04"])

    def test_decimal_comma_csv_and_invalid_values(self):
        frame = read_upload(b"fecha;venta\na;1,5\nb;2,5\n", "datos.csv", decimal=",")
        self.assertEqual(prepare_series(frame, "fecha", "venta").data.demanda.tolist(), [1.5, 2.5])
        frame["venta"] = frame["venta"].astype(object)
        for invalid in (np.nan, np.inf, -2, "error"):
            frame.loc[0, "venta"] = invalid
            with self.assertRaises(ValueError):
                prepare_series(frame, "fecha", "venta")

    def test_hourly_dates_keep_the_time(self):
        frame = pd.DataFrame({"fecha": pd.date_range("2025-01-01", periods=48, freq="h"), "y": np.ones(48)})
        series = prepare_series(frame, "fecha", "y", date_mode=True)
        self.assertFalse(series.data.periodo.duplicated().any())
        self.assertEqual(series.future_labels(2), ["2025-01-03 00:00:00", "2025-01-03 01:00:00"])


class LSTMTests(unittest.TestCase):
    def setUp(self):
        self.y = 100 + np.sin(np.arange(160)/2)*10
        self.data = pd.DataFrame({"periodo": np.arange(160).astype(str), "demanda": self.y})

    def test_scaler_and_training_windows_do_not_see_future(self):
        altered = self.y.copy()
        altered[100:] += 100000
        for mode in ("nivel", "cambio"):
            original = scaled_windows(self.y, 100, 12, 8, mode)
            changed = scaled_windows(altered, 100, 12, 8, mode)
            self.assertEqual(original[-2:], changed[-2:])
            mask = original[2]+8 <= 100
            np.testing.assert_array_equal(original[0][mask], changed[0][mask])
            np.testing.assert_array_equal(original[1][mask], changed[1][mask])

    def test_future_input_and_historical_input_are_identical(self):
        for mode in ("nivel", "cambio"):
            x, _, origins, anchors, center, scale = scaled_windows(self.y, 100, 12, 8, mode)
            future, anchor = future_input(self.y[:120], 12, mode, center, scale)
            np.testing.assert_allclose(future[0], x[origins == 120][0])
            self.assertEqual(anchor, anchors[origins == 120][0])

    def test_scaling_adapts_to_different_demand_units(self):
        original = scaled_windows(self.y, 100, 12, 8, "nivel")
        larger = scaled_windows(self.y*1000, 100, 12, 8, "nivel")
        np.testing.assert_allclose(original[0], larger[0], atol=1e-6)
        np.testing.assert_allclose(original[1], larger[1], atol=1e-6)

    def test_cycle_is_learned_not_hardcoded(self):
        self.assertEqual(suggest_cycle(100+20*np.sin(2*np.pi*np.arange(120)/7)), 7)
        self.assertIsNone(suggest_cycle(np.full(120, 42.)))
        self.assertIsNone(suggest_cycle(np.arange(120, dtype=float)))
        choices = candidates_for(self.y[:110], LSTMConfig())
        self.assertTrue(all(c["lookback"]+8+19 <= 110 for c in choices))
        self.assertEqual({c["representation"] for c in choices}, {"nivel", "cambio", "local"})

    def test_insufficient_history_and_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "insuficiente"):
            split_points(20, 8)
        for config in (LSTMConfig(horizon=0), LSTMConfig(members=2), LSTMConfig(learning_rate=float("nan"))):
            with self.assertRaises(ValueError):
                config.validate()

    def test_holdout_cannot_select_configuration_or_epoch(self):
        class FakeModel:
            def fit(self, x, target, **kw):
                return SimpleNamespace(history={"loss": [.2, .1], "val_loss": [.1, .2]})

            def predict(self, x, **kw):
                return np.zeros((len(x), 8))

            def count_params(self):
                return 10

        fake_tf = SimpleNamespace(keras=SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None),
                                  callbacks=SimpleNamespace(EarlyStopping=lambda **kw: kw, TerminateOnNaN=lambda: None)))
        config = LSTMConfig(members=1)
        with patch.dict(sys.modules, {"tensorflow": fake_tf}), \
             patch("lstm_engine.tensorflow_status", return_value=(True, "test")), \
             patch("lstm_engine.build_network", side_effect=lambda *args: FakeModel()):
            result = run_lstm(self.data, config)
            changed = self.data.copy()
            changed.loc[result["val_end"]:, "demanda"] = 100000
            other = run_lstm(changed, config)
        pd.testing.assert_frame_equal(result["search"], other["search"])
        self.assertEqual(result["selected"], other["selected"])
        self.assertEqual(result["selected"]["epochs"], 1)
        first = result["val_end"]
        np.testing.assert_allclose(result["detail"].query("Origen == @first")["LSTM"], other["detail"].query("Origen == @first")["LSTM"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

