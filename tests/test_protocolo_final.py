import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from motor_lstm import extract_series, make_forecast_table
from protocolo_final import confirm_pre_sales, origins, baseline, recursive, Candidate, select, evaluate, future, choose_lstm, compare_fixed_origin


class ProtocolTests(unittest.TestCase):
    def test_original_demand_future_refit_is_explicit_and_preserves_test(self):
        from types import SimpleNamespace
        s=pd.DataFrame({'fecha_inicio':pd.date_range('2024-01-01',periods=100,freq='7D'),
                        'valor':np.ones(100)})
        fit=SimpleNamespace(target='demanda_cp')
        c=Candidate('log',object(),0.,1.,12,51)
        p={'series':s,'test':pd.DataFrame({'RMSE':[8.]}),'selected':'original',
           'candidates':{'original':c},'original_demand_fit':fit}
        first=s.fecha_inicio.iloc[-1]+pd.Timedelta(days=7)
        original=pd.DataFrame({'fecha_inicio':[first.date()],'demanda_cp':[84.]})
        refitted=original.assign(demanda_cp=90.)
        snapshot=p['test'].copy(deep=True)
        with patch('pronostico_horizonte.forecast_lstm_verified',return_value=original) as old:
            with patch('motor_lstm.full_refit_forecast',return_value=refitted) as refit:
                result=future(p,first,refit=False)
                self.assertEqual(result.pronostico.iloc[0],84.)
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.fecha_inicio))
                refit.assert_not_called()
                self.assertEqual(future(p,first,refit=True).pronostico.iloc[0],90.)
                self.assertEqual(old.call_count,1)
                self.assertEqual(refit.call_count,1)
        pd.testing.assert_frame_equal(p['test'],snapshot)

    def test_demand_keeps_original_weights_scale_and_epochs(self):
        from types import SimpleNamespace
        from protocolo_final import demand_from_fit
        from motor_lstm import Settings
        class Net:
            def __call__(self,x,training=False):
                return np.zeros((len(x),1))
        s=pd.DataFrame({'fecha_inicio':pd.date_range('2023-01-02',periods=200,freq='7D'),
                        'valor':np.arange(200,dtype=float)+100})
        fit=SimpleNamespace(name='Demanda de maquila',series=s,model=Net(),mean=5.,std=.2,
                            best_epoch=51,settings=Settings(),idx_val=np.arange(140,170),
                            idx_test=np.arange(170,200))
        p=demand_from_fit(fit)
        c=p['candidates'][p['selected']]
        self.assertIs(c.model,fit.model)
        self.assertEqual((c.mean,c.std,c.epochs),(5.,.2,51))
        self.assertIsNone(p['test'])
        changed=s.copy(); changed.loc[170:,'valor']=99999
        fit.series=changed
        pd.testing.assert_frame_equal(p['validation'],demand_from_fit(fit)['validation'])

    def test_principal_is_lstm_even_when_reference_wins(self):
        candidates={'Persistencia':Candidate('Persistencia',None,0,1,0,0),
                    'LSTM A':Candidate('log',object(),0,1,12,3),
                    'LSTM B':Candidate('original',object(),0,1,12,4)}
        validation=pd.DataFrame({'Modelo':['Persistencia','LSTM A','LSTM B'],
                                 'RMSE':[1.,5.,3.],'MAE':[1.,4.,2.]})
        self.assertEqual(choose_lstm(validation,candidates),'LSTM B')
        with self.assertRaises(ValueError):
            choose_lstm(validation,{'Persistencia':candidates['Persistencia']})

    def test_final_forecast_rejects_baseline_as_principal(self):
        s=pd.DataFrame({'fecha_inicio':pd.date_range('2024-01-01',periods=100,freq='7D'),
                        'valor':np.ones(100)})
        p={'series':s,'test':pd.DataFrame({'RMSE':[1.]}),'selected':'Persistencia',
           'candidates':{'Persistencia':Candidate('Persistencia',None,0,1,0,0)}}
        with self.assertRaisesRegex(ValueError,'LSTM'):
            future(p,s.fecha_inicio.iloc[-1]+pd.Timedelta(days=7))

    def test_confirm_only_active_presales_and_preserve_source(self):
        frame = pd.DataFrame({'fecha_inicio': pd.date_range('2024-01-01', periods=80, freq='7D'),
                              'produccion_cp': [0]*10+[100]+[0]*69,
                              'demanda_cp': [np.nan]*15+[10]*65})
        result, trace = confirm_pre_sales(frame, True)
        self.assertTrue(frame.demanda_cp.iloc[:15].isna().all())
        self.assertTrue(result.demanda_cp.iloc[:10].isna().all())
        self.assertEqual(len(trace), 5)
        self.assertEqual(extract_series(result, 'produccion_cp').valor.eq(0).sum(), 69)
        self.assertEqual(len(extract_series(result, 'demanda_cp')), 70)
        unchanged, trace = confirm_pre_sales(frame, False)
        self.assertEqual(len(trace), 0)
        pd.testing.assert_frame_equal(unchanged, frame)

    def test_origins_stay_inside_block(self):
        oo = origins(70, 90, 12)
        self.assertEqual(oo, list(range(70, 79)))
        with self.assertRaises(ValueError):
            origins(70, 78, 12)

    def test_fixed_origin_uses_one_uninterrupted_12_week_block(self):
        class Net:
            def __call__(self, x, training=False):
                return np.array([[x[0, -1, 0]]])
        y=np.arange(1,41,dtype=float)
        c=Candidate("original",Net(),0.,1.,2,1)
        table,detail=compare_fixed_origin({"LSTM":c},y,20,12,y[:20])
        self.assertEqual(int(table.iloc[0]["Orígenes"]),1)
        self.assertEqual(int(table.iloc[0]["Horizonte"]),12)
        self.assertEqual(len(detail),12)
        self.assertEqual(detail["origen"].nunique(),1)
        self.assertEqual(detail["indice"].tolist(),list(range(20,32)))
        with self.assertRaises(ValueError):
            compare_fixed_origin({"LSTM":c},y,35,12,y[:20])

    def test_two_stage_expectation_and_sba(self):
        y = np.array([100., 0, 0, 100, 0, 0])
        self.assertAlmostEqual(baseline(y, 'Dos etapas', 3)[0], 100/3)
        self.assertTrue(np.isfinite(baseline(y, 'SBA', 12, .1)).all())

    def test_recursion_uses_predictions(self):
        class Network:
            def __call__(self, x, training=False):
                return np.array([[x[0, -1, 0]+1]])
        c = Candidate('original', Network(), 0., 1., 2, 1)
        np.testing.assert_allclose(recursive(c, np.array([2., 3.]), 3), [4, 5, 6])

    def test_total_is_demand_only(self):
        dates = pd.date_range('2026-11-02', periods=2, freq='7D')
        fs = {name: pd.DataFrame({'fecha_inicio': dates, col: [value]*2})
              for name, col, value in [('Demanda de maquila','demanda_mq',10),
                                       ('Demanda de corte propio','demanda_cp',20),
                                       ('Producción de corte propio','produccion_cp',999)]}
        self.assertTrue(make_forecast_table(fs).demanda_total.eq(30).all())

    def test_evaluate_reports_fixed_and_rolling_test_separately(self):
        class Net:
            def __call__(self, x, training=False):
                return np.array([[float(x[0, -1, 0])]])
        y=np.arange(1,101,dtype=float)
        series=pd.DataFrame({"fecha_inicio":pd.date_range("2023-01-02",periods=100,freq="7D"),
                             "valor":y})
        c=Candidate("original",Net(),0.,1.,2,1)
        selection={"series":series,"candidates":{"LSTM":c},
                   "train_end":70,"validation_end":85,"test":None}
        rolling=evaluate(selection)
        self.assertEqual(len(rolling),1)
        self.assertEqual(int(selection["fixed_test"].iloc[0]["Orígenes"]),1)
        self.assertEqual(int(selection["fixed_test"].iloc[0]["Horizonte"]),12)
        self.assertEqual(len(selection["fixed_test_predictions"]),12)
        self.assertEqual(len(selection["test_predictions"]),4*12)

    def test_test_values_cannot_change_selection(self):
        class Net:
            def __call__(self, x, training=False):
                return np.zeros((len(x), 1))
        series = pd.DataFrame({'fecha_inicio':pd.date_range('2023-01-02',periods=140,freq='7D'),
                               'valor':np.tile([0.,100.,0.,0.,50.],28)})
        c = Candidate('log',Net(),1.,1.,12,1)
        with patch('protocolo_final.train_candidate',return_value=(c,2.)):
            a = select(series,True)
            changed = series.copy()
            changed.loc[a['validation_end']:, 'valor'] = 999999.
            b = select(changed,True)
        self.assertEqual(a['selected'],b['selected'])
        pd.testing.assert_frame_equal(a['validation'],b['validation'])

    def test_full_refit_preserves_test_and_uses_frozen_epochs(self):
        from motor_lstm import Settings
        class Net:
            def __call__(self,x,training=False):
                return np.ones((len(x),1))
        s = pd.DataFrame({'fecha_inicio':pd.date_range('2024-01-01',periods=100,freq='7D'),
                          'valor':np.arange(100,dtype=float)})
        c = Candidate('original',Net(),0.,10.,12,7,settings=Settings())
        p = {'series':s,'test':pd.DataFrame({'RMSE':[12.]}),'selected':'fixed','candidates':{'fixed':c}}
        original_test = p['test'].copy()
        with patch('protocolo_final.train_candidate',return_value=Candidate('original',Net(),0.,20.,12,7)) as fit:
            result = future(p,s.fecha_inicio.iloc[-1]+pd.Timedelta(days=7))
        self.assertEqual(len(fit.call_args.args[0]),100)
        self.assertEqual(fit.call_args.args[3],7)
        self.assertEqual(c.std,10.)
        pd.testing.assert_frame_equal(p['test'],original_test)
        self.assertTrue(result.pronostico.eq(20.).all())


    def test_generic_refit_preserves_cp_bias_correction(self):
        from motor_lstm import Settings
        class Net:
            def __call__(self,x,training=False):
                return np.zeros((len(x),1))
        s=pd.DataFrame({'fecha_inicio':pd.date_range('2024-01-01',periods=100,freq='7D'),
                        'valor':np.arange(100,dtype=float)+20})
        c=Candidate('log',Net(),0.,1.,12,7,settings=Settings(),bias_correction=6.5)
        p={'series':s,'test':pd.DataFrame({'RMSE':[12.]}),'selected':'fixed',
           'candidates':{'fixed':c}}
        with patch('protocolo_final.train_candidate',
                   return_value=Candidate('log',Net(),0.,1.,12,7,settings=Settings())):
            result=future(p,s.fecha_inicio.iloc[-1]+pd.Timedelta(days=7))
        self.assertTrue((result.pronostico>=6.5).all())

    def test_planning_ignores_statistical_production_and_respects_capacity(self):
        from planificacion_reposicion import calcular_plan_reposicion
        data = pd.DataFrame({'fecha_inicio':pd.date_range('2026-11-02',periods=12,freq='7D'),
                             'demanda_mq':[350.]*12,'demanda_cp':[80.]*12,'produccion_cp':[0.]*12})
        a = calcular_plan_reposicion(data,100,0,420,22.04,35.16,4,35,80)
        b = calcular_plan_reposicion(data.assign(produccion_cp=1000000.),100,0,420,22.04,35.16,4,35,80)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue((a['Carga utilizada (equiv. MQ)']<=420).all())
        self.assertTrue((a['Inventario CP al inicio']+a['Ingreso desde lavado']-
                         a['Ventas CP cubiertas']).equals(a['Inventario CP final']))

if __name__ == '__main__':
    unittest.main()
