import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from motor_lstm import Settings
from protocolo_final import Candidate
from lstm_validacion_horizonte import search, future_search, GRID_EPOCHS

class HorizonTests(unittest.TestCase):
    def test_frozen_configuration_is_fitted_only_on_train(self):
        from lstm_validacion_horizonte import select_frozen
        class Net:
            def __call__(self,x,training=False): return np.ones((len(x),1))
        s=pd.DataFrame({'fecha_inicio':pd.date_range('2023-01-02',periods=140,freq='7D'),
                        'valor':np.arange(140,dtype=float)+1})
        settings=Settings(lookback=8,units=16)
        c=Candidate('original',Net(),0.,2.,8,30,settings=settings)
        spec={'selected':'H12 LSTM original w8 u16 e30','kind':'original',
              'epochs':30,'settings':vars(settings)}
        control={'candidates':{'Persistencia':Candidate('Persistencia',None,0,1,0,0)},
                 'train_end':98,'validation_end':119,'series':s,'settings':Settings(),
                 'selected':'Persistencia','original_demand_fit':object(),'test':pd.DataFrame({'old':[1.]})}
        with patch('lstm_validacion_horizonte.train_fixed',return_value=c) as fit:
            p=select_frozen(s,spec,control)
        np.testing.assert_array_equal(fit.call_args.args[0],s.valor.iloc[:98])
        self.assertIsNone(p['test'])
        self.assertNotIn('original_demand_fit',p)
        self.assertEqual(p['selected'],spec['selected'])
        self.assertIsNotNone(control['test'])

    def test_final_refit_uses_frozen_kind_epochs_and_keeps_test(self):
        class Net:
            def __call__(self,x,training=False): return np.ones((len(x),1))
        s=pd.DataFrame({'fecha_inicio':pd.date_range('2023-01-02',periods=140,freq='7D'),
                        'valor':np.arange(140,dtype=float)})
        c=Candidate('hurdle',Net(),0.,2.,8,30,settings=Settings(lookback=8,units=16))
        p={'selected':'H12 LSTM hurdle w8 u16 e30','candidates':{'H12 LSTM hurdle w8 u16 e30':c},
           'series':s,'test':pd.DataFrame({'RMSE':[12.]})}
        before=p['test'].copy()
        with patch('lstm_validacion_horizonte.train_fixed',return_value=c) as fit:
            out=future_search(p,s.fecha_inicio.iloc[-1]+pd.Timedelta(days=7),refit=True)
        self.assertEqual(len(fit.call_args.args[0]),140)
        self.assertEqual(fit.call_args.args[1],'hurdle')
        self.assertEqual(fit.call_args.args[3],30)
        self.assertTrue(out.pronostico.eq(2.).all())
        pd.testing.assert_frame_equal(before,p['test'])

    def test_search_reads_train_and_validation_only(self):
        class Net:
            def __call__(self,x,training=False):
                return np.ones((len(x),1))
        s=pd.DataFrame({'fecha_inicio':pd.date_range('2023-01-02',periods=140,freq='7D'),
                        'valor':np.arange(140,dtype=float)+1})
        calls=[]
        def train(y,kind,settings,epochs):
            calls.append((y.copy(),epochs))
            return Candidate('original',Net(),0.,1.,settings.lookback,epochs,settings=settings)
        with patch('lstm_validacion_horizonte.train_fixed',side_effect=train):
            a=search(s,False,Settings())
            changed=s.copy(); changed.loc[a['validation_end']:,'valor']=99999.
            b=search(changed,False,Settings())
        self.assertEqual(a['selected'],b['selected'])
        pd.testing.assert_frame_equal(a['validation'],b['validation'])
        self.assertIsNone(a['test'])
        for y,epoch in calls:
            np.testing.assert_array_equal(y,s.valor.iloc[:98])
            self.assertIn(epoch,GRID_EPOCHS)


    def test_cp_bias_calibration_uses_validation_only_and_survives_refit(self):
        from lstm_validacion_horizonte import select_frozen
        class Net:
            def __call__(self,x,training=False): return np.zeros((len(x),1))
        s=pd.DataFrame({'fecha_inicio':pd.date_range('2023-01-02',periods=140,freq='7D'),
                        'valor':np.arange(140,dtype=float)+10})
        settings=Settings(lookback=8,units=16)
        c=Candidate('original',Net(),0.,1.,8,30,settings=settings)
        spec={'selected':'H12 LSTM original w8 u16 e30','kind':'original',
              'epochs':30,'settings':vars(settings)}
        control={'candidates':{'Persistencia':Candidate('Persistencia',None,0,1,0,0)},
                 'train_end':98,'validation_end':119,'series':s,'settings':Settings(),
                 'selected':'Persistencia','test':None}
        with patch('lstm_validacion_horizonte.train_fixed',return_value=c):
            raw=select_frozen(s,spec,control,calibrate_bias=False)
            calibrated=select_frozen(s,spec,control,calibrate_bias=True)
        self.assertEqual(raw['bias_correction'],0.0)
        self.assertGreater(calibrated['bias_correction'],0.0)
        raw_rmse=raw['validation'].loc[raw['validation'].Modelo==spec['selected'],'RMSE'].iloc[0]
        calibrated_rmse=calibrated['validation'].loc[calibrated['validation'].Modelo==spec['selected'],'RMSE'].iloc[0]
        self.assertEqual(raw_rmse,calibrated_rmse)
        with patch('lstm_validacion_horizonte.train_fixed',return_value=c):
            out=future_search({**calibrated,'test':pd.DataFrame({'ok':[1]})},
                s.fecha_inicio.iloc[-1]+pd.Timedelta(days=7),weeks=12,refit=True)
        self.assertTrue((out.pronostico>=0).all())
        self.assertEqual(c.bias_correction,calibrated['bias_correction'])

if __name__=='__main__': unittest.main()
