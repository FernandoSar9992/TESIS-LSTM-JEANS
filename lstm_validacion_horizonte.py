"""Búsqueda predefinida de LSTM por validación recursiva; nunca selecciona con TEST."""
from dataclasses import replace
import numpy as np
import pandas as pd
from motor_lstm import Settings, _xy
from protocolo_final import (Candidate, encode, _network, _dataset, compare,
                              choose_lstm, origins, recursive, future)

GRID_EPOCHS=(10,30,60,100)

def train_fixed(y, kind, settings, epochs):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(settings.seed)
    tf.config.experimental.enable_op_determinism()
    y=np.asarray(y,float)
    a=np.log1p(y) if kind=='log' else y
    mean=float(a.mean()) if kind=='log' else 0.
    std=(float(a.std()) if kind=='log' else float(np.sqrt(np.mean(a*a)))) or 1.
    if kind=='hurdle':
        inputs=tf.keras.Input((settings.lookback,1))
        h=tf.keras.layers.LSTM(settings.units)(inputs)
        h=tf.keras.layers.Dropout(settings.dropout)(h)
        h=tf.keras.layers.Dense(max(8,settings.units//2),activation='relu')(h)
        probability=tf.keras.layers.Dense(1,activation='sigmoid',name='probabilidad')(h)
        quantity=tf.keras.layers.Dense(1,activation='softplus',name='cantidad_positiva')(h)
        outputs=tf.keras.layers.Concatenate()([probability,quantity])
        training=tf.keras.Model(inputs,outputs)
        positive_rate=max(float((y[settings.lookback:]>0).mean()),1e-6)
        def hurdle_loss(actual,pred):
            actual=tf.reshape(actual,[-1])
            event=tf.cast(actual>0,tf.float32)
            prob=tf.clip_by_value(pred[:,0],1e-7,1-1e-7)
            bce=-(event*tf.math.log(prob)+(1-event)*tf.math.log(1-prob))
            return bce+event*tf.square(actual-pred[:,1])/positive_rate
        training.compile(optimizer=tf.keras.optimizers.Adam(.001,clipnorm=1.),loss=hurdle_loss)
        model=tf.keras.Model(inputs,tf.keras.layers.Multiply()([probability,quantity]))
    else:
        training=model=_network(kind,settings)
    c=Candidate(kind,model,mean,std,settings.lookback,epochs,settings=settings)
    z=encode(y,c)
    x,yy=_xy(z,np.arange(settings.lookback,len(y)),settings.lookback)
    training.fit(_dataset(x,yy,settings.batch_size),epochs=epochs,verbose=0,shuffle=False)
    return c

def search(series,production=False,settings=None,control=None,progress=None):
    settings=settings or Settings()
    y=series.valor.to_numpy(float)
    tr=int(len(y)*settings.train_ratio)
    ve=int(len(y)*(settings.train_ratio+settings.val_ratio))
    origins(tr,ve); origins(ve,len(y))
    candidates=dict(control['candidates']) if control else {}
    for method in ('Persistencia','Media móvil 4','Media móvil 8','Media histórica'):
        candidates.setdefault(method,Candidate(method,None,0,1,0,0))
    tables=[]; details=[]
    table,detail=compare(candidates,y,tr,ve,y[:tr])
    tables.append(table); details.append(detail)
    kinds=('log','original','hurdle') if production else ('log','original')
    for kind in kinds:
        for lookback in (8,12):
            config=replace(settings,lookback=lookback,units=16)
            for epoch in GRID_EPOCHS:
                name=f'H12 LSTM {kind} w{lookback} u16 e{epoch}'
                if progress: progress(name)
                c=train_fixed(y[:tr],kind,config,epoch)
                candidates[name]=c
                table,detail=compare({name:c},y,tr,ve,y[:tr])
                tables.append(table); details.append(detail)
    table=pd.concat(tables,ignore_index=True).sort_values(['RMSE','MAE','Modelo'],kind='stable').reset_index(drop=True)
    selected=choose_lstm(table,candidates)
    # TEST only needs the winner, controls and references, not every discarded grid point.
    kept={k:v for k,v in candidates.items() if k==selected or not k.startswith('H12 ')}
    result={'selected':selected,'best_comparison':str(table.iloc[0].Modelo),
            'selection_policy':'Parrilla predefinida, RMSE recursivo de validación a 12 semanas',
            'candidates':kept,'validation':table,'validation_predictions':pd.concat(details,ignore_index=True),
            'train_end':tr,'validation_end':ve,'series':series.copy(),'settings':settings,
            'test':None,'horizon_search':True}
    if control and 'original_demand_fit' in control and selected==control['selected']:
        result['original_demand_fit']=control['original_demand_fit']
    return result

def select_frozen(series,spec,control,calibrate_bias=False):
    """Reproduce la decisión de desarrollo sin repetir la parrilla en Streamlit.

    El llamador debe verificar hash del archivo, ceros confirmados y configuración.
    Se entrena exclusivamente TRAIN; ninguna cifra de TEST entra en la elección.
    """
    if not spec['selected'].startswith('H12 '):
        if spec['selected']!=control['selected']:
            raise ValueError('El control no coincide con la selección congelada')
        return control
    settings=Settings(**spec['settings'])
    y=series.valor.to_numpy(float)
    tr,ve=control['train_end'],control['validation_end']
    c=train_fixed(y[:tr],spec['kind'],settings,int(spec['epochs']))
    candidates={**control['candidates'],spec['selected']:c}
    table,detail=compare(candidates,y,tr,ve,y[:tr])
    table=table.sort_values(['RMSE','MAE','Modelo'],kind='stable').reset_index(drop=True)
    # CP bias calibration estimated only from validation residuals; TEST remains untouched.
    bias_correction=0.0
    if calibrate_bias:
        selected_detail=detail.loc[detail['Modelo']==spec['selected']]
        if selected_detail.empty:
            raise ValueError('No hay predicciones de validación para calibrar el sesgo CP.')
        residual=(selected_detail['real']-selected_detail['prediccion']).to_numpy(float)
        if not np.isfinite(residual).all():
            raise ValueError('Los residuos de validación CP no son finitos.')
        bias_correction=float(np.mean(residual))
        c.bias_correction=bias_correction
    result={**control,'series':series.copy(),'selected':spec['selected'],'candidates':candidates,
            'best_comparison':str(table.iloc[0].Modelo),'validation':table,
            'validation_predictions':detail,'test':None,'horizon_search':True,
            'bias_correction':bias_correction,
            'selection_policy':'Configuración congelada por validación recursiva a 12 semanas'}
    result.pop('original_demand_fit',None)
    result.pop('test_predictions',None)
    return result


def future_search(selection,first_plan,weeks=12,refit=False):
    if not selection['selected'].startswith('H12 '):
        return future(selection,first_plan,weeks,refit=refit)
    if selection['test'] is None:
        raise ValueError('Evaluar prueba antes de generar futuro')
    s=selection['series']; first_plan=pd.Timestamp(first_plan)
    days=(first_plan-(pd.Timestamp(s.fecha_inicio.iloc[-1])+pd.Timedelta(days=7))).days
    if first_plan.weekday()!=0 or days<0 or days%7 or days>112 or not 1<=weeks<=26:
        raise ValueError('Fecha u horizonte inválido')
    c=selection['candidates'][selection['selected']]
    y=s.valor.to_numpy(float)
    if refit:
        c=train_fixed(y,c.kind,c.settings,c.epochs)
        c.bias_correction=float(selection.get('bias_correction',0.0))
    pred=recursive(c,y,weeks+days//7)[days//7:]
    return pd.DataFrame({'fecha_inicio':pd.date_range(first_plan,periods=weeks,freq='7D'),'pronostico':pred})
