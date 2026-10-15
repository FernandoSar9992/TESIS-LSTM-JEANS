import unittest
import numpy as np
import pandas as pd
import workshop_engine as engine


class NestedWorkshopTests(unittest.TestCase):
    def test_selection_rejects_one_lucky_fold_and_worse_worst_fold(self):
        self.assertTrue(hasattr(engine,'select_neural_variant'))
        frame=pd.DataFrame([
            {'Variante':name,'Corte':i,'RMSE':error,'MAE':error/2}
            for name,errors in [('control',[5,5,10]),('inestable',[.1,.1,11]),('estable',[4,4,9])]
            for i,error in enumerate(errors,1)])
        self.assertEqual(engine.select_neural_variant(frame,'control'),'estable')
        self.assertEqual(engine.select_neural_variant(frame[~frame.Variante.eq('estable')],'control'),'control')

    def data(self):
        t=np.arange(110)
        return pd.DataFrame({'periodo':[f'p{i}' for i in t], 'demanda':100+t+8*np.sin(t),
                             'cliente':'A','precio_unitario':1.5,'costo_unitario':1.,'origen':'REAL'})

    def test_rmsse_uses_only_training_seasonal_differences(self):
        data=self.data()
        targets=np.arange(80,90)
        pred=data.demanda.iloc[targets].to_numpy()+3
        actual=engine.score(data,targets,pred,fit_end=70,period=4)
        self.assertIn('RMSSE',actual)
        denom=np.mean((data.demanda.iloc[4:70].to_numpy()-data.demanda.iloc[:66].to_numpy())**2)
        self.assertAlmostEqual(actual['RMSSE'],3/np.sqrt(denom))
        data.loc[70:79,'demanda']=1e10
        self.assertEqual(actual['RMSSE'],engine.score(data,targets,pred,70,4)['RMSSE'])

    def test_zero_scale_is_undefined_not_perfect(self):
        data=self.data();data['demanda']=5.
        actual=engine.score(data,np.arange(80,90),np.full(10,5.),70,4)
        self.assertIn('RMSSE',actual)
        self.assertTrue(np.isnan(actual['RMSSE']))

    def test_outer_validation_cannot_choose_epochs_or_weights(self):
        self.assertTrue(hasattr(engine,'fit_temporal_neural'))
        data=self.data(); changed=data.copy()
        changed.loc[70:,'demanda']=1e8
        config=engine.WorkshopConfig(epochs=3,patience=1,members=1)
        candidate={'family':'LSTM','units':4,'dropout':.1,'lookback':12,'residual':'seasonal','loss':'huber'}
        a=engine.fit_temporal_neural(data,70,100,config,candidate,None)
        b=engine.fit_temporal_neural(changed,70,100,config,candidate,None)
        self.assertEqual(a['epochs'],b['epochs'])
        self.assertEqual(a['encoder'].fit_end,70)
        for x,y in zip(a['network'].get_weights(),b['network'].get_weights()):
            np.testing.assert_array_equal(x,y)

    def test_short_internal_training_is_rejected(self):
        self.assertTrue(hasattr(engine,'fit_temporal_neural'))
        with self.assertRaisesRegex(ValueError,'intern'):
            engine.fit_temporal_neural(self.data(),35,60,engine.WorkshopConfig(),
                                      {'lookback':24,'family':'LSTM','units':8,'dropout':.1},None)

    def test_application_training_uses_internal_epoch_selection(self):
        # El entrenamiento real de redes es caro; el doble registra fronteras
        # y devuelve una corrección nula. Se verifica la decisión observable.
        from unittest.mock import patch
        data=self.data()
        calls=[]
        class Zero:
            def __call__(self,inputs,**kwargs): return np.zeros((len(inputs[0]),1))
        def training(frame,fit_end,stop,config,candidate,ridge_candidate,fixed_epochs=None):
            calls.append((fit_end,stop,len(frame),fixed_epochs))
            return {'encoder':engine.Encoder(frame,fit_end,config),'network':Zero(),'ridge':None,
                    'ridge_candidate':ridge_candidate,'candidate':candidate,'epochs':2,'history':{}}
        with patch('workshop_engine.fit_neural',side_effect=training):
            engine.train_workshop(data,engine.WorkshopConfig(members=1))
        for fit,stop,size,fixed in calls:
            if fixed is None:
                self.assertEqual(stop,size)
                self.assertLess(fit,stop)
            else:
                self.assertEqual(fixed,2)
                self.assertEqual(fit,stop)
        self.assertTrue(any(fixed is not None for _,_,_,fixed in calls),
                        'Falta reajuste de TRAIN con las épocas seleccionadas internamente.')


if __name__=='__main__': unittest.main()
