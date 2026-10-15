"""Diagnóstico descriptivo con pesos congelados. No entrena ni selecciona."""
from pathlib import Path
import sys, json, hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workshop_data import prepare_workshop
from verify_workshop_artifacts import load_artifact
from workshop_engine import predict, residual_base
p=ROOT/'results/experiments/nested_lstm_2026-10-07'
source=Path(r'C:\Users\ASUS\Downloads\Chino U\10 Chino Décimo Ciclo\TESIS\DATOS_TALLER.xlsx')
expected=json.loads((ROOT/'results/experiments/corrected_frozen_audit_2026-10-07/evidence.json').read_text())['source_sha256']
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
raw=pd.read_excel(source,sheet_name='PARA_MODELO')
d=prepare_workshop(raw,declared_real=True).data
old=pd.read_csv(p/'modelo/processed.csv')
out=ROOT/'results/experiments/temporal_stability_2026-10-07'
out.mkdir(exist_ok=False)
rows=[]; details=[]; variables=[]
for label,a,b,path in [('Fold 1',69,97,p/'fold_3_1.json'),('Fold 2',97,125,p/'fold_3_2.json'),('Fold 3',125,153,p/'fold_3_3.json'),('TEST',153,180,p/'modelo/models_9.json')]:
    _,model=load_artifact(path,old)
    idx=np.arange(a,b); steps=np.ones(len(idx),int)
    assert np.all(idx-48<idx)
    y=d.demanda.iloc[idx].to_numpy(float); sn=d.demanda.iloc[idx-48].to_numpy(float)
    base=residual_base(d,idx,steps,model['encoder'],model['candidate'],np.zeros(len(idx)))
    np.testing.assert_array_equal(base,sn)
    pred=predict(model,d.iloc[:b],idx,steps)
    e=pred-y; r=y-sn
    def met(z):
        err=z-y
        return np.sqrt(np.mean(err**2)),np.mean(abs(err)),1-np.sum(err**2)/np.sum((y-y.mean())**2)
    rm,ma,r2=met(pred); nr,nm,nr2=met(sn)
    row=dict(Fold=label,N=len(idx),Inicio=d.periodo.iloc[a],Fin=d.periodo.iloc[b-1],Media=y.mean(),SD_demanda=y.std(ddof=1),Min=y.min(),Max=y.max(),CV=y.std(ddof=1)/y.mean(),LSTM_RMSE=rm,LSTM_MAE=ma,LSTM_R2=r2,SN_RMSE=nr,SN_MAE=nm,SN_R2=nr2,Mejora_pct=100*(nr-rm)/nr,Media_residual=r.mean(),SD_residual=r.std(ddof=1),Min_residual=r.min(),Max_residual=r.max(),Rango_residual=np.ptp(r),Correlacion=np.corrcoef(y,sn)[0,1],Error_max=abs(e).max(),P_lt5=100*np.mean(abs(e)<5),P_lt10=100*np.mean(abs(e)<10),P_lt20=100*np.mean(abs(e)<20),P_gt50=100*np.mean(abs(e)>50),Cambio_cliente_anual_pct=100*np.mean(d.cliente.iloc[idx].to_numpy()!=d.cliente.iloc[idx-48].to_numpy()))
    rows.append(row)
    for i,t in enumerate(idx):
        details.append(dict(Fold=label,Indice=int(t),Periodo=d.periodo.iloc[t],Real=y[i],Estacional=sn[i],Residual=r[i],Prediccion=pred[i],Error_abs=abs(e[i]),Cliente=d.cliente.iloc[t],Cliente_anterior=d.cliente.iloc[t-48]))
    for col in raw.columns:
        s=raw[col].iloc[a:b]
        variables.append(dict(Fold=label,Variable=str(col),Nulos=int(s.isna().sum()),Valores=json.dumps(s.value_counts(dropna=False).to_dict(),ensure_ascii=False,default=str) if s.nunique()<=12 else 'Variable continua: '+str(s.describe().to_dict())))
summary=pd.DataFrame(rows); detail=pd.DataFrame(details)
summary.to_csv(out/'summary.csv',index=False); detail.to_csv(out/'detail.csv',index=False); pd.DataFrame(variables).to_csv(out/'variables.csv',index=False)
print(summary.to_string(index=False))
print('COLUMNAS',list(raw.columns))
print(pd.DataFrame(variables).query("Variable.str.contains('Cliente|Modal|Precio|Costo',case=False)").to_string(index=False))
print('MAYORES',detail.sort_values('Error_abs',ascending=False).head(10).to_string(index=False))
print('INICIO_CAMBIO',detail[(detail.Indice>=120)&(detail.Indice<=140)].to_string(index=False))
