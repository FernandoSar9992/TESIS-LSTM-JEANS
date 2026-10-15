"""Pruebas de preparación y conciliación del modelo dual (sin entrenar TensorFlow)."""
import unittest

import numpy as np
import pandas as pd

from dual_forecast import prepare_dual, combined_forecast, combined_test
from forecast_engine import future_labels


def sample(n=112):
    months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
              "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    labels = [f"{months[(5 + i // 4) % 12]} {2024 + (5 + i // 4) // 12}-Se{i % 4 + 1}"
              for i in range(n)]
    return pd.DataFrame({
        "Periodo (4/mes)": labels,
        "Bloque temporal": ["TRAIN"] * 78 + ["VALIDACIÓN"] * 17 + ["PRUEBA"] * (n-95),
        "DEMANDA MAQUILA LSTM": [300.0] * n,
        "Origen maquila": ["SIMULADA"] * 78 + ["ORIGINAL"] * (n-78),
        "Maquila: registrado base": [None] * 78 + [300] * (n-78),
        "DEMANDA CORTE LSTM": [20.0] * n,
        "Origen corte": ["SOLO ESCENARIO"] * 78 + ["PARCIAL REGISTRADO"] * (n-78),
        "Corte: documentado neto": [None] * 78 + [20] * (n-78),
        "DEMANDA TOTAL LSTM": [320.0] * n,
    })


class DualModelTests(unittest.TestCase):
    def test_prepares_112_aligned_periods_and_provenance(self):
        result = prepare_dual(sample())
        self.assertEqual(len(result["Maquila"]), 112)
        self.assertEqual(result["Maquila"].origen.iloc[0], "IMPUTADO")
        self.assertEqual(result["Maquila"].origen.iloc[-1], "REAL")
        self.assertEqual(result["Corte propio"].origen.iloc[-1], "MIXTO")

    def test_future_is_48_period_calendar(self):
        labels = future_labels(prepare_dual(sample())["Maquila"], 3)
        self.assertEqual(labels, ["Octubre 2026-Se1", "Octubre 2026-Se2", "Octubre 2026-Se3"])

    def test_rejects_inconsistent_totals(self):
        raw = sample()
        raw.loc[0, "DEMANDA TOTAL LSTM"] = 700
        with self.assertRaisesRegex(ValueError, "no coincide"):
            prepare_dual(raw)

    def test_rejects_missing_periods(self):
        raw = sample().drop(index=10)
        with self.assertRaisesRegex(ValueError, "consecutivos"):
            prepare_dual(raw)

    def test_adds_forecasts_and_keeps_precision(self):
        a = pd.DataFrame({"Periodo": ["A", "B"], "Horizonte": [1, 2], "LSTM": [300.2, 320.4]})
        b = pd.DataFrame({"Periodo": ["A", "B"], "Horizonte": [1, 2], "LSTM": [22.5, 24.1]})
        future = combined_forecast(a, b)
        self.assertTrue(np.allclose(future.Total, [322.7, 344.5]))

    def test_rejects_misaligned_future(self):
        a = pd.DataFrame({"Periodo": ["A"], "Horizonte": [1], "LSTM": [300.]})
        b = pd.DataFrame({"Periodo": ["B"], "Horizonte": [1], "LSTM": [20.]})
        with self.assertRaisesRegex(ValueError, "no coinciden"):
            combined_forecast(a, b)

    def test_total_r2_is_recomputed_and_not_sum_of_r2(self):
        a = pd.DataFrame({"Periodo": ["A", "B", "C"], "Origen": ["REAL"]*3,
                          "Demanda_real": [3, 5, 7], "Prediccion_LSTM": [2, 5, 8]})
        b = pd.DataFrame({"Periodo": ["A", "B", "C"], "Origen": ["MIXTO"]*3,
                          "Demanda_real": [10, 10, 10], "Prediccion_LSTM": [11, 9, 9]})
        detail, all_scores, observed, n = combined_test(a, b)
        self.assertEqual(n, 0)
        self.assertIsNone(observed)
        self.assertTrue(np.allclose(detail.Pred_total, [13, 14, 17]))
        self.assertTrue(np.isfinite(all_scores["MAE"]))


if __name__ == "__main__":
    unittest.main()
