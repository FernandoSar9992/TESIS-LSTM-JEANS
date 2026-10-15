"""Auditoría reproducible de artefactos existentes; no entrena ni cambia el modelo."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
from verify_workshop_artifacts import load_artifact
from workshop_engine import predict, pairs
from workshop_data import prepare_workshop


def main():
    old = ROOT/'results/experiments/lstm_v2_2026-10-06/modelo'
    out = ROOT/'results/experiments/deep_audit_2026-10-07'
    out.mkdir(exist_ok=False)
    meta = json.loads((old/'metadata.json').read_text(encoding='utf-8'))
    manifest = json.loads((old.parent/'manifest.json').read_text(encoding='utf-8'))
    d = pd.read_csv(old/'processed.csv')
    source = Path(manifest['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['source_hash']
    fresh = prepare_workshop(pd.read_excel(source, sheet_name='PARA_MODELO'), declared_real=True).data
    pd.testing.assert_frame_equal(d.drop(columns='tipo_estimacion'), fresh.drop(columns='tipo_estimacion'), check_dtype=False)
    snapshots = out/'original_code'
    snapshots.mkdir()
    files = ['workshop_engine.py', 'workshop_app.py', 'workshop_data.py', 'demand_data.py',
             'forecast_engine.py', 'thesis_engine.py', 'workshop_planning.py', 'verify_workshop_artifacts.py']
    hashes = {}
    for file in files:
        shutil.copy2(ROOT/file, snapshots/file)
        hashes[file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    (out/'source_snapshot.json').write_text(json.dumps({'source_sha256': meta['source_hash'], 'code': hashes}, indent=2), encoding='utf-8')
    tr, va = meta['train_end'], meta['val_end']
    d['Etapa'] = np.where(d.index < tr, 'TRAIN', np.where(d.index < va, 'VALIDATION', 'TEST'))
    d['Indice'] = d.index
    d['Mes'] = (d.index % 48)//4+1
    d['Periodo_mes'] = d.index%4+1
    stats = []
    for stage, group in d.groupby('Etapa', sort=False):
        y = group.demanda
        q = y.quantile([.25, .5, .75, .9, .95, .99])
        stats.append({'Etapa': stage, 'N': len(y), 'Media': y.mean(), 'Mediana': y.median(), 'SD': y.std(),
                      'CV': y.std()/y.mean(), 'Min': y.min(), 'Max': y.max(), 'Q1': q[.25], 'Q3': q[.75],
                      'IQR': q[.75]-q[.25], 'P90': q[.9], 'P95': q[.95], 'P99': q[.99],
                      'Tendencia_OLS_por_periodo': np.polyfit(group.Indice, y, 1)[0]})
    pd.DataFrame(stats).to_csv(out/'distribution.csv', index=False)
    acf = pd.DataFrame({'Lag': range(1, 97), 'ACF': [d.demanda.autocorr(k) for k in range(1, 97)],
                        'ACF_TRAIN': [d.demanda.iloc[:tr].autocorr(k) for k in range(1, 97)]})
    pacf = []
    y = d.demanda.to_numpy()
    for k in range(1, 13):
        x = np.column_stack([np.ones(len(y)-k)]+[y[k-j:len(y)-j] for j in range(1, k+1)])
        pacf.append({'Lag': k, 'PACF_OLS_descriptiva': np.linalg.lstsq(x, y[k:], rcond=None)[0][-1]})
    acf.to_csv(out/'acf.csv', index=False)
    pd.DataFrame(pacf).to_csv(out/'pacf.csv', index=False)
    per_split=[]
    for stage,group in d.groupby('Etapa',sort=False):
        values=group.demanda.to_numpy()
        for k in range(1,min(12,len(values)//4)+1):
            x=np.column_stack([np.ones(len(values)-k)]+[values[k-j:len(values)-j] for j in range(1,k+1)])
            per_split.append({'Etapa':stage,'N':len(values),'Lag':k,
                              'ACF':pd.Series(values).autocorr(k),
                              'PACF_OLS':np.linalg.lstsq(x,values[k:],rcond=None)[0][-1]})
    pd.DataFrame(per_split).to_csv(out/'acf_pacf_by_split.csv',index=False)
    d['delta_anual']=d.demanda-d.demanda.shift(48)
    d.groupby('Etapa').delta_anual.agg(['count','mean','median','std','min','max']).to_csv(out/'seasonal_differences.csv')
    d.groupby(['Etapa', 'Mes']).demanda.agg(['count','mean','median','std']).to_csv(out/'month_patterns.csv')
    d.groupby(['Etapa', 'Periodo_mes']).demanda.agg(['count','mean','median','std']).to_csv(out/'within_month_patterns.csv')
    summary = pd.read_csv(old/'split_metrics.csv')
    allpred, independent, sizes = [], [], []
    for path in sorted(old.glob('models_*.json')):
        name, model = load_artifact(path, d)
        if name == meta['thesis_selected']:
            member = model['members'][0]
            origins, steps = pairs(24, tr, 8)
            params = int(sum(np.prod(w.shape) for w in member['network'].trainable_weights))
            sizes.append({'Parametros_por_red': params, 'Redes': len(model['members']), 'Filas_TRAIN': tr,
                          'Pares_entrenamiento': len(origins), 'Origenes_distintos': len(np.unique(origins)),
                          'Objetivos_distintos': len(np.unique(origins+steps-1)),
                          'Parametros_por_par': params/len(origins),
                          'Parametros_por_origen': params/len(np.unique(origins)),
                          'Parametros_por_objetivo': params/len(np.unique(origins+steps-1)),
                          'Epocas': model['epochs'], 'Activacion_salida': member['network'].layers[-1].activation.__name__,
                          'Media_scaler': member['encoder'].center, 'SD_scaler': member['encoder'].scale,
                          '735_escalado': (735-member['encoder'].center)/member['encoder'].scale})
        for stage, start, stop in [('TRAIN',24,tr),('VALIDATION',tr,va),('TEST',va,len(d))]:
            idx = np.arange(start, stop)
            pred = predict(model, d.iloc[:stop], idx, np.ones(len(idx),int))
            real = y[idx]
            error = pred-real
            # Segunda implementación, independiente de forecast_engine.metrics.
            mse = sum(float(e)**2 for e in error)/len(error)
            mae = sum(abs(float(e)) for e in error)/len(error)
            r2 = 1-sum(float(e)**2 for e in error)/sum((float(v)-float(real.mean()))**2 for v in real)
            mape = 100*sum(abs(float(e)/float(v)) for e,v in zip(error,real))/len(real)
            denom = y[48:tr]-y[:tr-48]
            values = {'MAE':mae, 'RMSE':mse**.5, 'MSE':mse, 'MAPE_%':mape, 'R2':r2,
                      'MASE':mae/np.mean(abs(denom)), 'RMSSE':np.sqrt(mse/np.mean(denom**2)), 'Sesgo':error.mean()}
            saved = summary[(summary.Modelo==name)&(summary.Grupo==stage)].iloc[0]
            for key in ('MAE','RMSE','MSE','MAPE_%','R2','MASE'):
                np.testing.assert_allclose(values[key], saved[key], atol=3e-4, rtol=1e-5)
            independent.append({'Modelo': name, 'Grupo':stage, 'N':len(idx), **values})
            for i,r,p in zip(idx,real,pred):
                allpred.append({'Modelo':name,'Etapa':stage,'Indice':i,'Periodo':d.periodo.iloc[i],
                                'Real':r,'Prediccion':p,'Error':p-r,'Absoluto':abs(p-r)})
    pd.DataFrame(sizes).to_json(out/'network_size.json', orient='records', indent=2)
    pd.DataFrame(independent).to_csv(out/'metrics_independent.csv', index=False)
    pred = pd.DataFrame(allpred)
    pred.to_csv(out/'one_step_predictions.csv', index=False)
    principal = pred[pred.Modelo.eq(meta['thesis_selected'])].copy()
    test = principal[principal.Etapa.eq('TEST')].copy()
    test['Porcentaje_SSE'] = 100*test.Error**2/sum(test.Error**2)
    test.sort_values('Absoluto', ascending=False).to_csv(out/'rank_errors.csv',index=False)
    trainq = np.quantile(y[:tr],[.25,.75,.9])
    spikes = []
    for index in (160,171):
        row = test[test.Indice.eq(index)].iloc[0]
        spikes.append({**row.to_dict(), 'Cliente':d.cliente.iloc[index], 'Precio':d.precio_unitario.iloc[index],
                       'Rezago48':y[index-48], 'Rezago96':y[index-96], 'Rezago144':y[index-144],
                       'Correccion_LSTM':row.Prediccion-y[index-48], 'Media_previa8':y[index-8:index].mean(),
                       'Umbral_Tukey_TRAIN':trainq[1]+1.5*(trainq[1]-trainq[0]),
                       'Outlier_Tukey_TRAIN':bool(row.Real>trainq[1]+1.5*(trainq[1]-trainq[0]))})
    pd.DataFrame(spikes).to_csv(out/'spikes.csv',index=False)
    controls = []
    for (model_name,stage),group in pred.groupby(['Modelo','Etapa']):
        threshold = group.Real.quantile(.9)
        for label,mask in [('90_inferior',group.Real<=threshold),('10_superior',group.Real>threshold),
                           ('superior_P90_TRAIN',group.Real>trainq[2])]:
            e=group.loc[mask,'Error']
            controls.append({'Modelo':model_name,'Etapa':stage,'Segmento':label,'N':len(e),
                             'Umbral': trainq[2] if label=='superior_P90_TRAIN' else threshold,
                             'MAE':e.abs().mean(),'RMSE':np.sqrt((e**2).mean()),'Sesgo':e.mean(),'Max_abs':e.abs().max()})
    pd.DataFrame(controls).to_csv(out/'peak_metrics.csv',index=False)
    sensitivity=[]
    for excluded in ([],[171],[160,171]):
        subset=test[~test.Indice.isin(excluded)]
        e=subset.Error.to_numpy(); r=subset.Real.to_numpy()
        sensitivity.append({'Excluidos_solo_diagnostico':str(excluded),'N':len(r),
                            'R2':1-sum(e**2)/sum((r-r.mean())**2), 'MAE':abs(e).mean(), 'RMSE':np.sqrt((e**2).mean())})
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity_NOT_OFFICIAL.csv', index=False)
    folds=pd.read_csv(old/'validation_folds.csv')
    folds['RMSSE']=[np.sqrt(row.MSE/np.mean((y[48:int(row.Train_fin)]-y[:int(row.Train_fin)-48])**2)) for _,row in folds.iterrows()]
    folds.to_csv(out/'historical_folds.csv',index=False)
    folds.groupby('Variante')[['MAE','RMSE','R2','MAPE_%','MASE','RMSSE']].agg(['mean','median','std','min','max']).to_csv(out/'historical_cv_summary.csv')
    figs=[px.line(d,x='Indice',y='demanda',hover_data=['periodo'],title='1. Serie completa'),
          px.line(d,x='Indice',y='demanda',color='Etapa',title='2. División temporal'),
          px.line(principal,x='Indice',y=['Real','Prediccion'],title='3. Real frente a predicho — LSTM'),
          px.line(principal,x='Indice',y='Error',color='Etapa',title='4. Residuos (predicción − real)'),
          px.histogram(principal,x='Error',color='Etapa',barmode='overlay',title='5. Distribución de residuos'),
          px.bar(principal,x='Indice',y='Absoluto',color='Etapa',title='6. Error absoluto por período'),
          px.bar(test.sort_values('Absoluto',ascending=False),x='Periodo',y='Absoluto',title='7. Mayores errores de TEST'),
          px.box(d,x='Etapa',y='demanda',points='all',title='8. Distribución por conjunto'),
          px.line(acf,x='Lag',y=['ACF','ACF_TRAIN'],title='9. Autocorrelación descriptiva'),
          px.line(folds[folds.Variante.isin([meta['thesis_selected'],'Naive estacional'])],x='Corte',y='RMSE',color='Variante',markers=True,title='10. Validación temporal histórica')]
    html='<html><meta charset="utf-8"><title>Auditoría LSTM</title><body><h1>Diagnóstico — 7 octubre 2026</h1><p>Resultados históricos; no implican precisión prospectiva. No se elimina ninguna observación.</p>'
    for i,fig in enumerate(figs):
        html+=fig.to_html(full_html=False,include_plotlyjs=True if i==0 else False)
    (out/'diagnostic.html').write_text(html+'</body></html>',encoding='utf-8')
    print(pd.DataFrame(stats).to_string(index=False))
    print(pd.DataFrame(sizes).to_string(index=False))
    print(pd.DataFrame(spikes).to_string(index=False))
    print(pd.DataFrame(sensitivity).to_string(index=False))
    print('Auditoría independiente de métricas: OK; fuente original y modelo preservados.')


if __name__=='__main__':
    main()
