import unittest
import pandas as pd
from planificacion_reposicion import calcular_plan_reposicion
from capacidad_recursos import loads

class ResourceCapacityTests(unittest.TestCase):
    def test_shared_resource_constrains_both_modalities(self):
        resources=pd.DataFrame({'recurso':['compartido','solo_cp'],
            'te_mq_s':[2.,0.],'te_cp_s':[3.,5.],'segundos_disponibles':[100.,50.]})
        f=pd.DataFrame({'fecha_inicio':pd.date_range('2026-11-02',periods=12,freq='7D'),
                        'demanda_mq':[40.]*12,'demanda_cp':[20.]*12,'produccion_cp':[99999.]*12})
        p=calcular_plan_reposicion(f,0,reserva_capacidad_cp_pct=35,recursos=resources)
        audit=loads(p,resources)
        self.assertTrue((audit.carga_segundos<=audit.segundos_disponibles+1e-8).all())
        self.assertTrue((p['Producción CP propuesta']<=10).all())
        self.assertNotIn('Carga utilizada (equiv. MQ)',p)
        pd.testing.assert_frame_equal(p,calcular_plan_reposicion(f.assign(produccion_cp=0),0,
                                       reserva_capacidad_cp_pct=35,recursos=resources))

    def test_stock_covers_sales_without_manufacturing(self):
        resources=pd.DataFrame({'recurso':['compartido'],
            'te_mq_s':[2.],'te_cp_s':[3.],'segundos_disponibles':[100.]})
        f=pd.DataFrame({'fecha_inicio':pd.date_range('2026-11-02',periods=12,freq='7D'),
                        'demanda_mq':[40.]*12,'demanda_cp':[20.]*12})
        p=calcular_plan_reposicion(f,698,recursos=resources)
        self.assertEqual(p['Producción CP propuesta'].sum(),0)
        self.assertEqual(p['Inventario CP final'].iloc[-1],458)

if __name__=='__main__': unittest.main()
