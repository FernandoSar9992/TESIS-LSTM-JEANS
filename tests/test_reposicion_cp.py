"""Pruebas de capacidad compartida y prioridad de corte propio."""
import unittest
import pandas as pd
from planificacion_reposicion import calcular_plan_reposicion

class ReposicionTests(unittest.TestCase):
    def test_cp_utiliza_holgura_sin_exceder_capacidad(self):
        dates = pd.date_range("2026-11-02", periods=8, freq="7D")
        frame = pd.DataFrame({"fecha_inicio": dates, "demanda_mq": [0]*8,
                              "demanda_cp": [200]*8})
        plan = calcular_plan_reposicion(frame, inventario_inicial=0,
                   capacidad_equiv_mq=420, te_mq=22.04, te_cp=35.16,
                   semanas_cobertura=4, reserva_capacidad_cp_pct=35)
        cap_cp = int(420 // (35.16/22.04))
        self.assertEqual(int(plan.loc[0,"Producción CP propuesta"]), cap_cp)
        self.assertTrue((plan["Carga utilizada (equiv. MQ)"] <= 420.01).all())

    def test_prioridad_cp_con_demanda_mq_alta(self):
        dates = pd.date_range("2026-11-02", periods=8, freq="7D")
        frame = pd.DataFrame({"fecha_inicio": dates, "demanda_mq": [400]*8,
                              "demanda_cp": [200]*8})
        plan = calcular_plan_reposicion(frame, 0, capacidad_equiv_mq=420,
                  te_mq=22.04, te_cp=35.16, semanas_cobertura=4,
                  reserva_capacidad_cp_pct=35)
        self.assertGreater(int(plan.loc[0,"Producción CP propuesta"]),0)
        self.assertTrue((plan["Carga utilizada (equiv. MQ)"] <= 420.01).all())
        self.assertGreater(int(plan.loc[0,"Maquila pendiente al cierre"]),0)

    def test_without_reservation_mq_first(self):
        dates = pd.date_range("2026-11-02", periods=8, freq="7D")
        frame = pd.DataFrame({"fecha_inicio":dates,"demanda_mq":[420]*8,
                              "demanda_cp":[200]*8})
        plan = calcular_plan_reposicion(frame,0,reserva_capacidad_cp_pct=0)
        self.assertEqual(int(plan.loc[0,"Producción CP propuesta"]),0)
        self.assertEqual(int(plan.loc[0,"Producción MQ propuesta"]),420)

if __name__ == "__main__":
    unittest.main()
