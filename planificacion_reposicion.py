"""Escenario PRELIMINAR de reposición CP: no sustituye PMP ni es LSTM de producción.
Demanda, inventario inicial, recepción tras lavado y capacidad compartida.
El TE de ambas modalidades sirve únicamente para estimar carga equivalente.
La reserva CP es una prioridad: CP puede aprovechar la holgura que quede tras MQ.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def calcular_plan_reposicion(pronostico, inventario_inicial, lavado_pendiente=0,
                            capacidad_equiv_mq=420, te_mq=22.04, te_cp=35.16,
                            semanas_cobertura=4, reserva_capacidad_cp_pct=0,
                            minimo_lote_cp=0, recursos=None):
    if recursos is not None:
        from capacidad_recursos import validate, maximum
        recursos=validate(recursos)
    for name,x in (("inventario",inventario_inicial),("lavado pendiente",lavado_pendiente),
                   ("capacidad",capacidad_equiv_mq),("TE MQ",te_mq),
                   ("TE CP",te_cp),("lote mínimo",minimo_lote_cp)):
        if not np.isfinite(float(x)) or float(x)<0:
            raise ValueError(name+" debe ser no negativo y finito")
    if not (te_mq>0 and te_cp>0 and capacidad_equiv_mq>0):
        raise ValueError("Tiempos y capacidad deben ser positivos")
    if not 1<=int(semanas_cobertura)<=8 or not 0<=int(reserva_capacidad_cp_pct)<=100:
        raise ValueError("Cobertura o reserva CP fuera de rango")
    required={"fecha_inicio","demanda_mq","demanda_cp"}
    if not required.issubset(pronostico.columns) or pronostico.empty:
        raise ValueError("Faltan fecha_inicio, demanda_mq o demanda_cp")
    f=pronostico.copy().sort_values("fecha_inicio").reset_index(drop=True)
    for key in ("demanda_mq","demanda_cp"):
        f[key]=pd.to_numeric(f[key],errors="raise")
        if f[key].isna().any() or (f[key]<0).any():
            raise ValueError("Demanda negativa o faltante: "+key)
    dates=pd.to_datetime(f.fecha_inicio)
    if dates.isna().any() or (dates.diff().dt.days.dropna()!=7).any():
        raise ValueError("Fechas semanales no consecutivas")
    mq=np.rint(f.demanda_mq).astype(int).to_numpy()
    cp=np.rint(f.demanda_cp).astype(int).to_numpy()
    ratio=float(te_cp)/float(te_mq)
    cap=float(capacidad_equiv_mq)
    reserved=float(reserva_capacidad_cp_pct)/100
    stock=int(inventario_inicial)
    arrival=int(lavado_pendiente)
    backlog=0
    result=[]
    for i,when in enumerate(dates):
        stock_start=stock
        stock+=arrival
        sold=min(stock,int(cp[i]))
        shortage=int(cp[i])-sold
        stock-=sold
        # Cobertura de demanda posterior. El último pronóstico se mantiene
        # para los pasos que exceden el horizonte, como supuesto explícito.
        target=int(sum(int(cp[min(i+j,len(cp)-1)])
                       for j in range(1,1+int(semanas_cobertura))))
        need=max(0,target-stock)
        mq_need=int(mq[i])+backlog
        cp_capacity_total=int(math.floor((cap+1e-9)/ratio))
        if recursos is not None:
            if reserved<=0:
                # Prioridad MQ: CP aprovecha la capacidad que quede libre.
                prod_mq=min(mq_need,maximum(recursos,'mq'))
                prod_cp=min(need,maximum(recursos,'cp',prod_mq))
            else:
                # Primero se reserva una cuota CP; MQ ocupa la capacidad restante.
                # Después CP puede aprovechar toda holgura compatible con MQ.
                cp_priority=min(need,int(math.floor(maximum(recursos,'cp')*reserved+1e-9)))
                prod_mq=min(mq_need,maximum(recursos,'mq',cp_priority))
                prod_cp=min(need,maximum(recursos,'cp',prod_mq))
        elif reserved<=0:
            # Prioridad MQ: CP utiliza la holgura real de la capacidad equivalente.
            prod_mq=min(mq_need,int(math.floor(cap+1e-9)))
            prod_cp=min(need,cp_capacity_total,
                        int(math.floor((cap-prod_mq+1e-9)/ratio)))
        else:
            # La reserva CP es una prioridad, no un límite máximo rígido.
            cp_priority=min(need,cp_capacity_total,
                            int(math.floor((cap*reserved+1e-9)/ratio)))
            prod_mq=min(mq_need,int(math.floor(cap-ratio*cp_priority+1e-9)))
            prod_cp=min(need,cp_capacity_total,
                        int(math.floor((cap-prod_mq+1e-9)/ratio)))
        backlog=mq_need-prod_mq
        used=prod_mq+ratio*prod_cp
        if recursos is None and used>cap+1e-7:
            raise ArithmeticError("Exceso de capacidad equivalente")
        result.append({
            "Semana desde":when.date(),
            "Demanda MQ":int(mq[i]),"Demanda CP":int(cp[i]),
            "Inventario CP al inicio":stock_start,
            "Ingreso desde lavado":int(arrival),
            "Ventas CP cubiertas":int(sold),
            "Ventas CP no cubiertas":int(shortage),
            "Inventario CP final":int(stock),
            "Inventario objetivo CP":int(target),
            "Necesidad CP antes de capacidad":int(need),
            "Producción CP propuesta":int(prod_cp),
            "Producción MQ propuesta":int(prod_mq),
            "Maquila pendiente al cierre":int(backlog),
            "CP en lavado para siguiente semana":int(prod_cp),
            "Carga utilizada (equiv. MQ)":round(float(used),2),
            "Holgura (equiv. MQ)":round(max(0.0,cap-used),2),
            "Lote CP bajo mínimo":bool(0<prod_cp<int(minimo_lote_cp)),
        })
        if recursos is not None:
            load=prod_mq*recursos.te_mq_s+prod_cp*recursos.te_cp_s
            if (load>recursos.segundos_disponibles+1e-7).any():
                raise ArithmeticError('Exceso de capacidad por recurso')
            utilization=load/recursos.segundos_disponibles
            row=result[-1]
            row.pop('Carga utilizada (equiv. MQ)'); row.pop('Holgura (equiv. MQ)')
            row['Utilización máxima de recurso (%)']=100*float(utilization.max())
            row['Recurso más utilizado']=recursos.loc[utilization.idxmax(),'recurso']
        arrival=prod_cp
    return pd.DataFrame(result)
