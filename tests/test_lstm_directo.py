import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from motor_lstm import Settings
from protocolo_final import Candidate, recursive
from lstm_directo import _windows, select_direct


class DirectHorizonTests(unittest.TestCase):
    def test_training_windows_keep_all_targets_inside_train(self):
        z = np.arange(40, dtype=float)
        x, y = _windows(z, end=30, lookback=8, horizon=12)
        self.assertEqual(x.shape, (11, 8, 1))
        self.assertEqual(y.shape, (11, 12))
        # Last target of last training window is index 29, never 30 or later.
        self.assertEqual(y[-1, -1], 29)
        self.assertEqual(x[-1, -1, 0], 21)

    def test_direct_prediction_emits_all_horizons_at_once(self):
        class Net:
            def __init__(self): self.calls = 0
            def __call__(self, x, training=False):
                self.calls += 1
                return np.zeros((len(x), 12), dtype=float)
        net = Net()
        c = Candidate("direct", net, 0., 1., 8, 10,
                      strategy="direct", output_horizon=12)
        out = recursive(c, np.arange(1, 20, dtype=float), 12)
        self.assertEqual(len(out), 12)
        self.assertEqual(net.calls, 1)
        np.testing.assert_allclose(out, 0.)

    def test_longer_horizon_advances_in_direct_blocks(self):
        class Net:
            def __init__(self): self.calls = 0
            def __call__(self, x, training=False):
                self.calls += 1
                return np.full((len(x), 12), self.calls, dtype=float)
        net = Net()
        c = Candidate("direct", net, 0., 1., 8, 10,
                      strategy="direct", output_horizon=12)
        out = recursive(c, np.arange(1, 20, dtype=float), 26)
        self.assertEqual(len(out), 26)
        self.assertEqual(net.calls, 3)
        np.testing.assert_allclose(out[:12], np.expm1(1.))
        np.testing.assert_allclose(out[12:24], np.expm1(2.))
        np.testing.assert_allclose(out[24:], np.expm1(3.))

    def test_validation_selection_never_trains_on_validation_or_test(self):
        s = pd.DataFrame({
            "fecha_inicio": pd.date_range("2023-01-02", periods=160, freq="7D"),
            "valor": np.tile([80., 100., 120., 90.], 40)
        })
        settings = Settings(lookback=12, units=16, train_ratio=.70, val_ratio=.15)
        tr = int(len(s)*settings.train_ratio)
        seen = []
        class Net:
            def __call__(self, x, training=False):
                return np.zeros((len(x), 12), dtype=float)
        def fake_train(y, cfg, lookback, epochs, horizon):
            seen.append(np.asarray(y).copy())
            return Candidate("direct", Net(), 0., 1., lookback, epochs,
                             settings=cfg, strategy="direct", output_horizon=horizon)
        recursive_net = type("RNet", (), {
            "__call__": lambda self, x, training=False: np.zeros((len(x), 1), dtype=float)
        })()
        control_candidate = Candidate("log", recursive_net, 0., 1., 12, 10)
        control = {
            "candidates": {"LSTM control": control_candidate,
                           "Persistencia": Candidate("Persistencia", None, 0., 1., 0, 0)},
            "train_end": tr,
            "validation_end": int(len(s)*.85),
            "series": s.copy(),
            "settings": settings,
            "test": None
        }
        with patch("lstm_directo.train_direct", side_effect=fake_train):
            first = select_direct(s, settings, control=control)
            changed = s.copy()
            changed.loc[control["validation_end"]:, "valor"] = 999999.
            second = select_direct(changed, settings, control=control)
        self.assertEqual(len(seen), 2*len((8, 12, 16))*len((10, 30, 60, 100)))
        for values in seen:
            np.testing.assert_array_equal(values, s.valor.iloc[:tr].to_numpy())
        self.assertEqual(first["selected"], second["selected"])
        pd.testing.assert_frame_equal(first["validation"], second["validation"])
        self.assertIsNone(first["test"])

    def test_direct_candidate_requires_exact_output_dimension(self):
        class BadNet:
            def __call__(self, x, training=False):
                return np.zeros((len(x), 11))
        c = Candidate("direct", BadNet(), 0., 1., 8, 10,
                      strategy="direct", output_horizon=12)
        with self.assertRaisesRegex(ValueError, "debe devolver 12"):
            recursive(c, np.arange(1, 20, dtype=float), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
