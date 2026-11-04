"""Experimento cerrado de desarrollo. TEST ya observado, no evaluación ciega nueva."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import date
import pandas as pd
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from motor_lstm import prepare_base,extract_series,Settings,fit_one,TARGETS,make_forecast_table
from protocolo_final import confirm_pre_sales,demand_from_fit,select,evaluate
from lstm_validacion_horizonte import search,future_search

source=Path(sys.argv[1]); out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
digest=hashlib.sha256(source.read_bytes()).hexdigest()
base,_=prepare_base(pd.read_excel(source,sheet_name='DEMANDA'),today=date(2026,11,4))
base,_=confirm_pre_sales(base,True)
results={};manifest={'source_sha256':digest,'test_previously_observed':True,'selection':'validation RMSE 12 weeks','models':{}}
for name,key in TARGETS.items():
    print('SERIE',name,flush=True)
    s=extract_series(base,key)
    control=demand_from_fit(fit_one(name,s,Settings())) if name.startswith('Demanda') else select(s,True,Settings())
    p=search(s,key=='produccion_cp',Settings(),control=control,progress=lambda msg:print(msg,flush=True))
    p['validation'].to_csv(out/(key+'_validacion.csv'),index=False)
    p['validation_predictions'].to_csv(out/(key+'_validacion_predicciones.csv'),index=False)
    results[name]=p
    c=p['candidates'][p['selected']]
    manifest['models'][name]={'selected':p['selected'],'best_reference_or_lstm':p['best_comparison'],
        'epochs':c.epochs,'kind':c.kind,'settings':vars(c.settings),'mean':c.mean,'std':c.std,'smear':c.smear}
    print('SELECCION',name,p['selected'],flush=True)
# All series frozen before ANY test comparison in this experiment.
(out/'seleccion_antes_de_prueba.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
forecasts={};rows=[]
for name,p in results.items():
    key=TARGETS[name]
    table=evaluate(p);table.to_csv(out/(key+'_prueba.csv'),index=False)
    p['test_predictions'].to_csv(out/(key+'_prueba_predicciones.csv'),index=False)
    rows.append(table[table.Modelo==p['selected']].assign(Serie=name))
    frozen=table.copy(deep=True)
    forecasts[name]=future_search(p,pd.Timestamp('2026-11-02'),12,refit=(key=='produccion_cp')).rename(columns={'pronostico':key})
    pd.testing.assert_frame_equal(frozen,p['test'])
pd.concat(rows).to_csv(out/'metricas_seleccionadas.csv',index=False)
make_forecast_table(forecasts).to_csv(out/'pronostico_12_semanas.csv',index=False)
assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
print('EXPERIMENTO COMPLETO FUENTE INTACTA',flush=True)
