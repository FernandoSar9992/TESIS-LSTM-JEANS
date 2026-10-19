"""Regresiones de la auditoría temporal sin TensorFlow."""
import unittest
import numpy as np
import pandas as pd
from seasonal_diagnostics import profile_temporal


class ProfileTests(unittest.TestCase):
    def test_exact_annual_recurrence_and_no_test_leakage(self):
        year = np.array([150 + 10*np.cos(i/4) for i in range(48)])
        y = np.tile(year, 4)
        data = pd.DataFrame({"demanda": y, "origen": ["IMPUTADO"]*len(y)})
        a = profile_temporal(data)
        self.assertAlmostEqual(a["ESTACIONAL_TODOS_DESCRIPTIVO"]["R2"], 1.0, places=8)
        self.assertEqual(a["ESTACIONAL_SOLO_REAL"]["N"], 0)
        y[180:] = 1e6
        b = profile_temporal(pd.DataFrame({"demanda": y, "origen": ["IMPUTADO"]*len(y)}))
        self.assertAlmostEqual(a["ESTACIONAL_TODOS_DESCRIPTIVO"]["RMSE"],
                               b["ESTACIONAL_TODOS_DESCRIPTIVO"]["RMSE"], places=8)

    def test_partial_origins_are_not_real(self):
        y = np.linspace(100, 200, 180)
        origin = ["IMPUTADO"] * 180
        origin[130] = "MIXTO"
        origin[131] = "REAL"
        data = pd.DataFrame({"demanda": y, "origen": origin})
        result = profile_temporal(data)
        self.assertEqual(result["ESTACIONAL_SOLO_REAL"]["N"], 1)
        self.assertIsNone(result["ESTACIONAL_SOLO_REAL"]["R2"])


if __name__ == "__main__":
    unittest.main()
