"""Capacidad semanal mixta: segundos por recurso; fuente de solo lectura."""
from io import BytesIO
from hashlib import sha256
import numpy as np
import pandas as pd

def validate(resources):
    needed=['recurso','te_mq_s','te_cp_s','segundos_disponibles']
    if not set(needed).issubset(resources.columns) or resources.empty:
        raise ValueError('Matriz de recursos incompleta')
    a=resources[needed[1:]].to_numpy(float)
    if not np.isfinite(a).all() or (a[:,:2]<0).any() or (a[:,2]<=0).any():
        raise ValueError('Tiempos o disponibilidades inválidos')
    if not (a[:,:2]>0).any(axis=0).all():
        raise ValueError('Cada modalidad debe consumir algún recurso')
    return resources

def read_capacity(payload):
    d=pd.read_excel(BytesIO(payload),sheet_name='CAPACIDAD',header=None)
    if d.shape[0]<39 or str(d.iat[8,0]).strip()!='Código':
        raise ValueError('Formato CAPACIDAD no reconocido')
    seconds_per_hour=float(d.iat[5,8])
    if seconds_per_hour!=3600: raise ValueError('Conversión segundos/hora inesperada')
    rows=[]; checks=[]
    for kind,start,end,hcol,capcols in [('maquina',9,23,13,(10,11)),('trabajador',25,33,8,(13,14))]:
        for i in range(start,end):
            mq,cp=float(d.iat[i,2]),float(d.iat[i,3])
            if mq==0 and cp==0: continue
            hours=float(d.iat[i,hcol])
            row={'recurso':str(d.iat[i,0]),'tipo':kind,'te_mq_s':mq,'te_cp_s':cp,
                 'segundos_disponibles':hours*seconds_per_hour,
                 'fuente':f'CAPACIDAD!A{i+1}:O{i+1}', 'sha256':sha256(payload).hexdigest()}
            rows.append(row)
            for modal,te,col in [('MQ',mq,capcols[0]),('CP',cp,capcols[1])]:
                if te>0:
                    calculated=int(np.floor(hours*seconds_per_hour/te+1e-9))
                    cached=float(d.iat[i,col])
                    checks.append({'recurso':row['recurso'],'modalidad':modal,
                                   'capacidad_calculada':calculated,'capacidad_excel':cached,
                                   'coincide':calculated==cached})
    return validate(pd.DataFrame(rows)),pd.DataFrame(checks)

def maximum(resources,mode,other_quantity=0):
    own=resources['te_'+mode+'_s'].to_numpy(float)
    other=resources['te_'+('cp' if mode=='mq' else 'mq')+'_s'].to_numpy(float)
    remaining=resources.segundos_disponibles.to_numpy(float)-other*other_quantity
    if (remaining < -1e-7).any(): raise ValueError('La otra modalidad ya excede capacidad')
    mask=own>0
    return max(0,int(np.floor(np.min(remaining[mask]/own[mask])+1e-9)))

def loads(plan,resources):
    validate(resources)
    rows=[]
    for _,p in plan.iterrows():
        for _,r in resources.iterrows():
            used=p['Producción MQ propuesta']*r.te_mq_s+p['Producción CP propuesta']*r.te_cp_s
            rows.append({'fecha_inicio':p['Semana desde'],'recurso':r.recurso,
                         'carga_segundos':used,'segundos_disponibles':r.segundos_disponibles,
                         'utilizacion_pct':100*used/r.segundos_disponibles})
    return pd.DataFrame(rows)
