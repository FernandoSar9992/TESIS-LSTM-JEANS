"""Pruebas de auditoría cronológica y trazabilidad sin uso de TEST."""
import unittest
import numpy as np
import pandas as pd

from validation_audit import (
    validation_error_audit, validation_error_summary, validation_error_priority
)


class ValidationAuditTests(unittest.TestCase):
    def make_result(self):
        n = 100
        y = np.arange(n, dtype=float) + 30
        frame = pd.DataFrame({
            "periodo": [f"P{i}" for i in range(n)],
            "demanda": y.copy(),
            "origen": ["IMPUTADO"] * n,
            "tipo_estimacion": ["SIMULADA"] * n,
            "demanda_original": [np.nan] * n,
        })
        frame.loc[70, "origen"] = "REAL"
        frame.loc[70, "demanda_original"] = y[70]
        val = np.arange(70, 85)
        return {"train_end": 70, "val_end": 85, "config": {"periodicity": 48},
                "_artifacts": {"data": frame},
                "reference_model": "Naive",
                "validation_detail": pd.DataFrame({
                    "Periodo": frame.periodo.iloc[val].to_numpy(),
                    "Real": y[val], "LSTM": y[val] + 2,
                    "Naive": y[val]-1, "Naive estacional": y[val]-4,
                    "Ridge": y[val]+5})}

    def test_only_validation_and_original_preserved(self):
        r = self.make_result()
        audit = validation_error_audit(r)
        self.assertEqual(len(audit), 15)
        self.assertEqual(audit.Periodo.iloc[0], "P70")
        self.assertEqual(audit.Periodo.iloc[-1], "P84")
        self.assertEqual(audit.Procedencia.iloc[0], "REAL")
        self.assertEqual(int(audit.LSTM_mejor_que_referencia.sum()), 0)
        self.assertAlmostEqual(audit.Error_absoluto_LSTM.mean(), 2.0)
        self.assertTrue(np.isnan(audit.Demanda_documentada_original.iloc[-1]))
        self.assertTrue((audit.Indice_temporal <= 85).all())

    def test_test_targets_do_not_change_audit(self):
        r = self.make_result()
        first = validation_error_audit(r)
        r["_artifacts"]["data"].loc[85:, "demanda"] = 99999
        other = validation_error_audit(r)
        pd.testing.assert_frame_equal(first, other)

    def test_group_and_top_errors(self):
        r = self.make_result()
        detailed = validation_error_audit(r)
        summarized = validation_error_summary(detailed).set_index("Grupo")
        self.assertEqual(summarized.loc["SOLO_REAL", "N"], 1)
        self.assertTrue(np.isnan(summarized.loc["SOLO_REAL", "R2"]))
        self.assertEqual(summarized.loc["MIXTO", "N"], 0)
        self.assertEqual(len(validation_error_priority(detailed, n=3)), 3)

    def test_misaligned_periods_rejected(self):
        r = self.make_result()
        r["validation_detail"].loc[0, "Periodo"] = "incorrecto"
        with self.assertRaisesRegex(ValueError, "no coinciden"):
            validation_error_audit(r)


if __name__ == "__main__":
    unittest.main()
