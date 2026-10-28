"""Pronóstico de producción intermitente; no cambia demanda ni datos originales.
TSB y Croston-SBA: selección exclusivamente por validación cronológica.
Valores pronosticados = esperanza semanal, no órdenes de fabricación.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from motor_lstm import metrics

ALPHAS=(0.05,0.10,0.20,0.30)
METHODS=("TSB","SBA")

def predict_one(history,method,alpha):
    y=np.asarray(history,dtype=float)
    if len(y)==0 or (y<0).any() or not np.isfinite(y).all():
        raise ValueError("Historial de producción inválido.")
    positive=np.flatnonzero(y>0)
    if len(positive)==0: return 0.0
    z=float(y[positive[0]])
    if method=="TSB":
        p=1.0 if positive[0]==0 else 1.0/float(positive[0]+1)
        for v in y[positive[0]+1:]:
            event=float(v>0)
            p=alpha*event+(1-alpha)*p
            if v>0: z=alpha*float(v)+(1-alpha)*z
        return max(0.0,p*z)
    if method=="SBA":
        gaps=np.diff(np.r_[-1,positive])
        interval=float(gaps[0])
        for v,gap in zip(y[positive[1:]],gaps[1:]):
            pass
        # Iterate event positions, updating size and interarrival interval.
        for i,pos in enumerate(positive[1:],start=1):
            z=alpha*float(y[pos])+(1-alpha)*z
            interval=alpha*float(pos-positive[i-1])+(1-alpha)*interval
        return max(0.0,(1-alpha/2)*z/max(1.0,interval))
    raise ValueError(method)

def walkforward(y,indices,method,alpha):
    arr=np.asarray(y,dtype=float)
    return np.array([predict_one(arr[:int(i)],method,alpha) for i in indices],dtype=float)

@dataclass(frozen=True)
class IntermittentSelection:
    method:str
    alpha:float
    validation_mae:float
    comparison:pd.DataFrame

def select_by_validation(fit):
    y=fit.series["valor"].to_numpy(float)
    actual=y[fit.idx_val]
    rows=[]
    for method in METHODS:
        for a in ALPHAS:
            pred=walkforward(y,fit.idx_val,method,a)
            value=metrics(actual,pred,fit.train_reference)
            rows.append({"Método":method,"Alpha":a,**value})
    # No usar TEST para seleccionar. Se conserva LSTM como referencia.
    rows.append({"Método":"LSTM original","Alpha":float("nan"),**fit.validation_metrics})
    table=pd.DataFrame(rows).sort_values(["MAE","RMSE","Método"]).reset_index(drop=True)
    best=table.iloc[0]
    return IntermittentSelection(str(best["Método"]),float(best["Alpha"]),float(best["MAE"]),table)

def evaluate_test(fit,selection):
    if selection.method=="LSTM original":
        return None
    y=fit.series["valor"].to_numpy(float)
    idx=np.asarray(fit.idx_test,dtype=int)
    actual=y[idx]
    pred=walkforward(y,idx,selection.method,selection.alpha)
    dates=fit.series.loc[idx,"fecha_inicio"].to_numpy()
    result=pd.DataFrame({"fecha_inicio":pd.to_datetime(dates),"real":actual,"Pronóstico_intermitente":pred})
    m=metrics(actual,pred,fit.train_reference)
    # Backtest recursivo: únicamente historial anterior al TEST.
    n=min(12,len(idx))
    fixed=predict_one(y[:int(idx[0])],selection.method,selection.alpha)
    recursive=np.repeat(fixed,n)
    rm=metrics(actual[:n],recursive,fit.train_reference)
    return result,m,rm

def forecast_future(fit,selection,dates):
    if selection.method=="LSTM original":
        return None
    y=fit.series["valor"].to_numpy(float)
    expected=predict_one(y,selection.method,selection.alpha)
    return pd.DataFrame({"fecha_inicio":pd.to_datetime(dates),"produccion_cp":np.repeat(expected,len(dates))})
