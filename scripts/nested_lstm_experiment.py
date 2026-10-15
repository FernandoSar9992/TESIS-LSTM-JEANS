"""Seis ablations predefinidas, validación externa y una evaluación retrospectiva final."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
from dataclasses import replace, asdict
from hashlib import sha256
import numpy as np
import pandas as pd
from workshop_engine import (WorkshopConfig, fit_temporal_neural, predict, pairs, score,
                             evaluate_groups, finalize_workshop, forecast_workshop, select_neural_variant)
from auditar_taller import save, save_artifact
from verify_workshop_artifacts import load_artifact


def main():
    old=ROOT/'results/experiments/lstm_v2_2026-10-06/modelo'
    out=ROOT/'results/experiments/nested_lstm_2026-10-07'
    out.mkdir(exist_ok=False)
    meta=json.loads((old/'metadata.json').read_text(encoding='utf-8'))
    data=pd.read_csv(old/'processed.csv')
    source=json.loads((old.parent/'manifest.json').read_text(encoding='utf-8'))
    if sha256(data.to_csv(index=False).encode()).hexdigest() == meta['data_hash']:
        pass
    else:
        # CSV vacío se interpreta como NaN; restaurar el texto vacío original.
        data['tipo_estimacion'] = data['tipo_estimacion'].fillna('')
        if sha256(data.to_csv(index=False).encode()).hexdigest() != meta['data_hash']:
            raise ValueError('Los datos procesados no coinciden con su hash histórico.')
    if sha256(Path(source['source']).read_bytes()).hexdigest() != meta['source_hash']:
        raise ValueError('La fuente Excel cambió; cree una auditoría para la nueva versión.')
    config=WorkshopConfig(**meta['config'])
    base={'family':'LSTM','units':8,'dropout':.1,'lookback':24,'residual':'seasonal','loss':'mse'}
    variants={'Control anidado':base,'4 unidades':{**base,'units':4},'Ventana 12':{**base,'lookback':12},
              'Loss MAE':{**base,'loss':'mae'},'Loss Huber':{**base,'loss':'huber'},
              'Sin comerciales':{**base,'use_business':False}}
    folds=[(69,97),(97,125),(125,153)]
    protocol={'source':source['source'],'source_sha256':meta['source_hash'],'config':asdict(config),
              'variants':variants,'folds':folds,'seeds':[42,43,44],
              'selection':'Mejor RMSE entre variantes con RMSE y MAE medios <= control, mejora RMSE en >=2 cortes, sin peor RMSE máximo; de lo contrario control. MASE/RMSSE y R2 son diagnósticos, no se ocultan incumplimientos.',
              'test_status':'RETROSPECTIVO YA CONSULTADO; NO SE USA PARA SELECCIÓN NI VETO',
              'engine_sha256':sha256((ROOT/'workshop_engine.py').read_bytes()).hexdigest(),
              'processed_csv_sha256':sha256((old/'processed.csv').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    rows=[]; seeds=[]; models={}; horizon_rows=[]; peak_rows=[]
    for name,candidate in variants.items():
        for fold,(fit,stop) in enumerate(folds,1):
            origins,steps=pairs(fit,stop,config.horizon)
            members=[]
            for seed in (42,43,44):
                print(name,'corte',fold,'semilla',seed,flush=True)
                member=fit_temporal_neural(data.iloc[:stop],fit,stop,replace(config,seed=seed),candidate,None)
                pred=predict(member,data.iloc[:stop],origins,steps)
                seeds.append({'Variante':name,'Corte':fold,'Semilla':seed,'Epocas':member['epochs'],
                              'Inner_fin':member['inner_fit_end'],**score(data,origins+steps-1,pred,fit,48)})
                members.append(member)
            ensemble={**members[0],'members':members,'epochs':[m['epochs'] for m in members]}
            pred=predict(ensemble,data.iloc[:stop],origins,steps)
            values=score(data,origins+steps-1,pred,fit,48)
            rows.append({'Variante':name,'Corte':fold,'Train_fin':fit,'Validacion_fin':stop,**values})
            for h in range(1,9):
                mask=steps==h
                horizon_rows.append({'Variante':name,'Corte':fold,'Horizonte':h,
                                     **score(data,origins[mask]+h-1,pred[mask],fit,48)})
            target=data.demanda.to_numpy()[origins+steps-1]
            high=target>np.quantile(data.demanda.iloc[:fit],.9)
            for label,mask in [('altas_P90_TRAIN',high),('resto',~high)]:
                e=pred[mask]-target[mask]
                peak_rows.append({'Variante':name,'Corte':fold,'Segmento':label,'N':len(e),
                                  'MAE':np.mean(abs(e)) if len(e) else np.nan,
                                  'RMSE':np.sqrt(np.mean(e**2)) if len(e) else np.nan,
                                  'Sesgo':np.mean(e) if len(e) else np.nan})
            models[name]=ensemble
            save_artifact(name,ensemble,out/f'fold_{list(variants).index(name)}_{fold}')
            pd.DataFrame(rows).to_csv(out/'cv.csv',index=False)
            pd.DataFrame(seeds).to_csv(out/'seeds.csv',index=False)
            pd.DataFrame(horizon_rows).to_csv(out/'cv_horizons.csv',index=False)
            pd.DataFrame(peak_rows).to_csv(out/'cv_peaks.csv',index=False)
            print('RMSE externo',values['RMSE'],'MAE',values['MAE'],flush=True)
    cv=pd.DataFrame(rows)
    summary=cv.groupby('Variante')[['MAE','RMSE','MAPE_%','MSE','R2','MASE','RMSSE']].agg(['mean','median','std','min','max'])
    summary.to_csv(out/'cv_summary.csv')
    selected=select_neural_variant(cv,'Control anidado')
    frozen={'selected':selected,'candidate':variants[selected], 'epochs':models[selected]['epochs'],
            'selection_source':'SOLO CV EXTERNA','no_test_veto':True}
    (out/'selection_frozen.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print('SELECCIÓN FIJADA',frozen,flush=True)
    # Desde aquí, una sola evaluación final. No se cambia selected después.
    chosen_name='LSTM anidada: '+selected
    saved_models={}
    for path in old.glob('models_*.json'):
        name,model=load_artifact(path,data)
        saved_models[name]=model
    saved_models[chosen_name]=models[selected]
    result={**meta,'engine_version':'taller-lstm-nested-experiment-v1','selected':chosen_name,
            'thesis_selected':chosen_name,'lstm_name':chosen_name,'evaluated':False,
            'processed':data,'validation_folds':cv,'seed_metrics':pd.DataFrame(seeds),
            'search':cv.groupby('Variante',as_index=False).RMSE.mean().rename(columns={'RMSE':'RMSE_validacion'}),
            '_artifacts':{'data':data,'models':saved_models}}
    result['split_metrics'],result['detail'],result['horizon_metrics']=evaluate_groups(result,[('TRAIN',24,125),('VALIDATION',125,153)])
    finalize_workshop(result,progress=lambda s:print(s,flush=True))
    forecast_workshop(result,pd.read_csv(old/'forecast.csv').Periodo.tolist())
    save(result,out/'modelo')
    print(result['split_metrics'].query("Grupo == 'TEST'").to_string(index=False),flush=True)


if __name__=='__main__':main()
