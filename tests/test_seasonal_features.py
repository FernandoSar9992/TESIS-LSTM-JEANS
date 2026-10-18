"""Pruebas de codificacion estacional causal, sin requerir TensorFlow."""
import unittest

import numpy as np
import pandas as pd

from lstm_engine import seasonal_encoding, scaled_windows, future_input
from thesis_engine import calendar_start_offset, candidates, ThesisConfig


class SeasonalFeatureTests(unittest.TestCase):
    def test_calendar_origin_for_different_histories(self):
        maquila = pd.DataFrame({"periodo": ["Enero 2023-Se1"]})
        corte = pd.DataFrame({"periodo": ["Mayo 2024-Se4"]})
        self.assertEqual(calendar_start_offset(maquila), 0)
        self.assertEqual(calendar_start_offset(corte), 19)

    def test_december_fair_indicator_matches_calendar(self):
        z = np.arange(120, dtype=float)
        # diciembre-Se4 == period index 47; last feature pair: fair, low-season
        dec = seasonal_encoding(z, 47, 1, "nivel", 0., 48, 0, True)
        self.assertEqual(dec.shape, (1, 8))
        self.assertEqual(dec[0, -2], 1.)
        jan = seasonal_encoding(z, 48, 1, "nivel", 0., 48, 0, True)
        self.assertEqual(jan[0, -2], 0.)
        self.assertEqual(jan[0, -1], 1.)

    def test_future_values_cannot_change_historic_features(self):
        y = np.arange(1, 131, dtype=float)
        a = scaled_windows(y, 80, 8, 1, "local", 48, 19, True)[0]
        y2 = y.copy()
        y2[80:] = 100000.
        b = scaled_windows(y2, 80, 8, 1, "local", 48, 19, True)[0]
        # Se evalúan solo secuencias cuyos insumos terminan antes de TRAIN.
        np.testing.assert_allclose(a[:80-8], b[:80-8], rtol=0, atol=1e-6)

    def test_features_match_future_interface(self):
        y = np.arange(1, 101, dtype=float)
        x, target, origins, anchor, center, scale = scaled_windows(
            y, 80, 8, 1, "local", 48, 19, True)
        xf, af = future_input(y, 8, "local", center, scale, 48, 19, True)
        self.assertEqual(x.shape[-1], 10)
        self.assertEqual(xf.shape, (1, 8, 10))
        self.assertEqual(x.shape[-1], xf.shape[-1])

    def test_annual_residual_alignment_and_causal_inputs(self):
        values = np.tile(np.arange(48, dtype=float) + 150, 4)
        x, target, origins, anchors, center, scale = scaled_windows(
            values, 130, 8, 1, "estacional", 48)
        self.assertEqual(origins[0], 48)
        expected = (values[origins] - values[origins - 48]) / scale
        np.testing.assert_allclose(target.ravel(), expected, atol=1e-5)
        future, anchor = future_input(values[:150], 8, "estacional",
                                      center, scale, 48)
        np.testing.assert_allclose(future[0], x[origins == 150][0])
        self.assertAlmostEqual(float(anchor), float(anchors[origins == 150][0]))

    def test_short_series_gets_annual_candidates_with_later_folds(self):
        config = ThesisConfig(calendar_experiment=True)
        variants = candidates(config, 45, 19, latest_training=79)
        self.assertTrue(any(v["representation"] == "estacional" for v in variants))

    def test_candidates_include_nonseasonal_and_seasonal(self):
        config = ThesisConfig(calendar_experiment=True)
        opts = candidates(config, 85, 19)
        self.assertTrue(any(not c.get("periodicity") for c in opts))
        self.assertTrue(any(c.get("calendar_extra") for c in opts))
        self.assertTrue(all(c.get("calendar_offset") == 19 for c in opts if c.get("periodicity")))


if __name__ == "__main__":
    unittest.main()
