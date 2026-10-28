"""Evaluación a 12 semanas y pronóstico estable para la planificación.

Las redes LSTM se entrenan una sola vez con TRAIN (validación para parada
temprana). No se reajustan antes del pronóstico futuro; los valores TEST
solo sirven como contexto histórico después de la evaluación.
La elección de método de DEMANDA CP usa únicamente orígenes en VALIDACIÓN.
"""
from dataclasses import dataclass
from datetime import date,timedelta
import numpy as np
import pandas as pd
from motor_lstm import metrics

CANDIDATES=("LSTM validada","Persistencia fija","Media móvil 4",
            "Media móvil 8","Tendencia amortiguada","Estacional 4")
TARGET_DEMAND_CP="Demanda de corte propio"

def _periods(fit,first_plan,weeks):
    if hasattr(first_plan,"date") and not isinstance(first_plan,date):
        first_plan=first_plan.date()
    if first_plan.weekday()!=0:
        raise ValueError("El plan debe comenzar un lunes.")
    if not 1<=int(weeks)<=26:
        raise ValueError("El horizonte debe estar entre 1 y 26 semanas.")
    last=fit.series["fecha_inicio"].iloc[-1].date()
    next_week=last+timedelta(days=7)
    days=(first_plan-next_week).days
    if days<0 or days%7!=0 or days//7>16:
        raise ValueError("Inicio de plan anterior al historial o alejado más de 16 semanas.")
    return days//7

def _baseline(history,method,horizon):
    y=np.asarray(history,dtype=float)
    if len(y)<8 or not np.isfinite(y).all():
        raise ValueError("Historial insuficiente para comparar pronósticos.")
    h=int(horizon)
    if method=="Persistencia fija":
        pred=np.repeat(y[-1],h)
    elif method=="Media móvil 4":
        pred=np.repeat(np.mean(y[-4:]),h)
    elif method=="Media móvil 8":
        pred=np.repeat(np.mean(y[-8:]),h)
    elif method=="Tendencia amortiguada":
        level=float(np.mean(y[-4:]))
        previous=float(np.mean(y[-8:-4]))
        slope=(level-previous)/4.0
        k=np.arange(1,h+1,dtype=float)
        # La tendencia se amortigua progresivamente, evitando extrapolación lineal ilimitada.
        pred=level+slope*0.65*(1-np.power(0.65,k))/(1-0.65)
    elif method=="Estacional 4":
        pred=np.tile(y[-4:],int(np.ceil(h/4)))[:h]
    else:
        raise ValueError("Método no reconocido: "+method)
    return np.maximum(0.0,np.asarray(pred,dtype=float))

def lstm_from_origin(fit,origin,steps):
    """Proyección recursiva sin verdad futura, con pesos y escala ya validados."""
    look=int(fit.settings.lookback)
    y=fit.series["valor"].to_numpy(float)
    if origin<look or origin>len(y):
        raise ValueError("Origen fuera del historial válido.")
    window=list(((np.log1p(y[origin-look:origin])-fit.mean)/fit.std).astype(float))
    result=[]
    for _ in range(int(steps)):
        x=np.asarray(window[-look:],dtype="float32").reshape(1,look,1)
        zhat=float(fit.model.predict(x,verbose=0)[0,0])
        if not np.isfinite(zhat):
            raise ValueError("La LSTM produjo un número no finito.")
        window.append(zhat)
        value=np.expm1(np.clip(zhat*fit.std+fit.mean,-20,20))
        result.append(max(0.0,float(value)))
    return np.asarray(result,dtype=float)

def predict_at_origin(fit,method,origin,steps):
    if method=="LSTM validada":
        return lstm_from_origin(fit,origin,steps)
    return _baseline(fit.series["valor"].to_numpy(float)[:origin],method,steps)

@dataclass(frozen=True)
class DemandSelection:
    method:str
    comparison:pd.DataFrame
    origins:tuple
    horizon:int

def select_demand_by_validation(fit, horizon=12):
    """Valida proyecciones recursivas sobre orígenes ubicados en VALIDACIÓN.

    No inspecciona ni usa datos de TEST para elegir el método.
    """
    idx=np.asarray(fit.idx_val,dtype=int)
    n=len(idx)
    steps=min(int(horizon),n)
    if steps<8: raise ValueError("Validación necesita 8 semanas o más.")
    first=int(idx[0])
    last_origin=int(idx[-steps])
    offset=(last_origin-first)//2
    origins=tuple(sorted(set([first,first+offset,last_origin])))
    y=fit.series["valor"].to_numpy(float)
    actual=np.concatenate([y[o:o+steps] for o in origins])
    rows=[]
    for method in CANDIDATES:
        predicted=np.concatenate([predict_at_origin(fit,method,o,steps) for o in origins])
        vals=metrics(actual,predicted,fit.train_reference)
        rows.append({"Método":method,"Orígenes":len(origins),
                     "Horizonte":steps,**vals})
    report=pd.DataFrame(rows).sort_values(["MAE","RMSE","Método"]).reset_index(drop=True)
    return DemandSelection(str(report.iloc[0]["Método"]),report,origins,steps)

def evaluate_demand_test(fit,selection,horizon=12):
    idx=np.asarray(fit.idx_test,dtype=int)
    n=min(int(horizon),len(idx))
    first=int(idx[0])
    observed=fit.series["valor"].to_numpy(float)[first:first+n]
    predicted=predict_at_origin(fit,selection.method,first,n)
    dates=pd.to_datetime(fit.series.loc[idx[:n],"fecha_inicio"].to_numpy())
    result=pd.DataFrame({"fecha_inicio":dates,"real":observed,"Seleccionado":predicted})
    return result,metrics(observed,predicted,fit.train_reference)

def forecast_lstm_verified(fit,first_plan,weeks):
    bridge=_periods(fit,first_plan,weeks)
    y=fit.series["valor"].to_numpy(float)
    pred=lstm_from_origin(fit,len(y),bridge+int(weeks))[bridge:]
    return pd.DataFrame({"fecha_inicio":[first_plan+timedelta(days=7*k) for k in range(int(weeks))],
                         fit.target:pred})

def forecast_demand_selected(fit,selection,first_plan,weeks):
    bridge=_periods(fit,first_plan,weeks)
    y=fit.series["valor"].to_numpy(float)
    pred=predict_at_origin(fit,selection.method,len(y),bridge+int(weeks))[bridge:]
    return pd.DataFrame({"fecha_inicio":[first_plan+timedelta(days=7*k) for k in range(int(weeks))],
                         fit.target:pred})
