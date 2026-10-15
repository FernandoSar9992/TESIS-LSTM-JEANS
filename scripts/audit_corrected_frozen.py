"""Read-only model audit: frozen weights, corrected source, no fitting/tuning."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workshop_data import prepare_workshop
from verify_workshop_artifacts import load_artifact
from workshop_engine import predict
from workshop_planning import client_predictions, METHODS

source = Path(r'C:\Users\ASUS\Downloads\Chino U\10 Chino Décimo Ciclo\TESIS\DATOS_TALLER.xlsx')
folder = ROOT/'results/experiments/nested_lstm_2026-10-07/modelo'
out = ROOT/'results/experiments/corrected_frozen_audit_2026-10-07'
out.mkdir(exist_ok=False)
old = pd.read_csv(folder/'processed.csv')
data = prepare_workshop(pd.read_excel(source, sheet_name='PARA_MODELO'), declared_real=True).data
assert len(data) == len(old) == 180
pd.testing.assert_frame_equal(data[['periodo','demanda','cliente','precio_unitario','costo_unitario']].iloc[:153].reset_index(drop=True), old[['periodo','demanda','cliente','precio_unitario','costo_unitario']].iloc[:153].reset_index(drop=True), check_dtype=False)
changes = []
for c in ['periodo','demanda','cliente','precio_unitario','costo_unitario']:
    for i in np.flatnonzero(data[c].ne(old[c])):
        changes.append({'Indice':int(i),'Periodo':data.periodo.iloc[i],'Campo':c,'Anterior':old[c].iloc[i],'Actual':data[c].iloc[i]})
pd.DataFrame(changes).to_csv(out/'changes.csv',index=False)
name, model = load_artifact(folder/'models_9.json', old)
idx = np.arange(153,180)
pred = predict(model,data,idx,np.ones(27,int))
y = data.demanda.to_numpy(float)
diff = y[48:125]-y[:77]
scale = float(np.mean(abs(diff))); squared = float(np.mean(diff**2))
def measures(p):
    e = p-y[idx]; mse=float(np.mean(e**2)); mae=float(np.mean(abs(e)))
    return dict(N=27,MAE=mae,RMSE=np.sqrt(mse),MAPE=100*float(np.mean(abs(e/y[idx]))),R2=1-float(np.sum(e**2)/np.sum((y[idx]-y[idx].mean())**2)),MASE=mae/scale,RMSSE=np.sqrt(mse/squared))
comparison=pd.DataFrame([{'Modelo':n,**measures(p)} for n,p in [('LSTM congelada',pred),('Naive',y[idx-1]),('Seasonal Naive (48)',y[idx-48])]])
comparison.to_csv(out/'comparison.csv',index=False)
table=pd.DataFrame({'Periodo':data.periodo.iloc[idx].to_numpy(),'Real':y[idx],'Pronostico':pred,'Error_absoluto':abs(pred-y[idx])}).sort_values('Error_absoluto',ascending=False)
table.to_csv(out/'test_ranked.csv',index=False)
max_change=0.
for i,p in zip(idx,pred):
    changed=data.copy()
    changed.loc[i:,'demanda']=100000.
    changed.loc[i:,'cliente']='FUTURO_DESCONOCIDO'
    changed.loc[i:,'precio_unitario']=999.
    changed.loc[i:,'costo_unitario']=888.
    after=predict(model,changed,np.array([i]),np.array([1]))[0]
    max_change=max(max_change,abs(float(after-p)))
    np.testing.assert_allclose(after,p,atol=3e-4,rtol=1e-5)
    for method in METHODS:
        assert client_predictions(data,i,[i],48,method)==client_predictions(changed,i,[i],48,method)
evidence={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frozen_model':name,'training_and_validation_unchanged':True,'scale_abs':scale,'scale_squared':squared,'scale_rmse':np.sqrt(squared),'future_perturbation_max_delta':max_change,'retrained':False,'tuning':False,'scope':'Reevaluation of saved frozen ensemble; UI session weights not exported.'}
(out/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print(comparison.to_string(index=False)); print(table.to_string(index=False)); print(json.dumps(evidence,indent=2))
