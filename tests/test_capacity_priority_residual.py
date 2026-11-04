"""Regresión: CP debe aprovechar capacidad residual tras asignar maquila."""
import unittest
import pandas as pd
from planificacion_reposicion import calcular_plan_reposicion
from capacidad_recursos import loads

class CapacityResidualTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "fecha_inicio": pd.date_range("2026-11-09", periods=8, freq="7D"),
            "demanda_mq": [10]*8,
            "demanda_cp": [100]*8,
        })

    def resources(self):
        return pd.DataFrame({
            "recurso": ["compartido"],
            "te_mq_s": [2.0],
            "te_cp_s": [3.0],
            "segundos_disponibles": [100.0],
        })

    def test_cp_uses_residual_after_priority_quota(self):
        resources = self.resources()
        plan = calcular_plan_reposicion(
            self.frame(), 0, capacidad_equiv_mq=420,
            semanas_cobertura=4, reserva_capacidad_cp_pct=35,
            recursos=resources,
        )
        self.assertEqual(int(plan.loc[0, "Producción MQ propuesta"]), 10)
        self.assertEqual(int(plan.loc[0, "Producción CP propuesta"]), 26)
        audit = loads(plan, resources)
        self.assertTrue((audit.carga_segundos <= audit.segundos_disponibles).all())
        self.assertAlmostEqual(float(plan.loc[0, "Utilización máxima de recurso (%)"]), 98.0)

    def test_zero_cp_priority_still_uses_residual(self):
        resources = self.resources()
        plan = calcular_plan_reposicion(
            self.frame(), 0, semanas_cobertura=4,
            reserva_capacidad_cp_pct=0, recursos=resources,
        )
        self.assertEqual(int(plan.loc[0, "Producción MQ propuesta"]), 10)
        self.assertEqual(int(plan.loc[0, "Producción CP propuesta"]), 26)

if __name__ == "__main__":
    unittest.main()
