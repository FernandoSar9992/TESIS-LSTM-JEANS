"""Exporta comparación y escenarios explícitos; no vuelve a elegir modelos."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from capacidad_recursos import read_capacity,loads
from planificacion_reposicion import calcular_plan_reposicion

out=Path('outputs/integracion_v4'); exp=Path('outputs/experimento_horizonte_v4')
f=pd.read_csv(out/'pronostico_12_semanas.csv',parse_dates=['fecha_inicio'])
cfg=json.loads((exp/'seleccion_antes_de_prueba.json').read_text())
capacity=Path(sys.argv[1]); payload=capacity.read_bytes()
resources,checks=read_capacity(payload)
high=f.copy(); error_rows=[]; comparison=[]
for key,name in [('demanda_mq','Demanda de maquila'),('demanda_cp','Demanda de corte propio'),('produccion_cp','Producción de corte propio')]:
    selected=cfg['models'][name]['selected']
    new=pd.read_csv(out/(key+'_prueba.csv')); old=pd.read_csv(Path('outputs/demandas_restauradas_v3')/(key+'_prueba.csv'))
    old_name='LSTM log1p Duan' if key=='produccion_cp' else 'LSTM demanda original'
    for version,table,model in [('v3',old,old_name),('v4',new,selected)]:
        comparison.append(table[table.Modelo==model].assign(Serie=name,Version=version))
    if key.startswith('demanda'):
        d=pd.read_csv(exp/(key+'_validacion_predicciones.csv'))
        d=d[d.Modelo==selected].copy();d['error']=d.real-d.prediccion
        delta=d.groupby('h').error.quantile(.9).reindex(range(1,13)).clip(lower=0).to_numpy()
        assert np.isfinite(delta).all()
        high[key]=high[key]+delta
        error_rows.extend({'Serie':name,'h':i+1,'incremento_escenario_alto':v} for i,v in enumerate(delta))
pd.concat(comparison).to_csv(out/'comparacion_v3_v4.csv',index=False)
high['demanda_total']=high.demanda_mq+high.demanda_cp
high.to_csv(out/'escenario_demanda_alta.csv',index=False)
pd.DataFrame(error_rows).to_csv(out/'errores_validacion_escenario.csv',index=False)
summaries=[]
for label,forecast in [('base',f),('alto',high)]:
    plan=calcular_plan_reposicion(forecast,698,0,420,22.04,35.16,4,35,80,recursos=resources)
    audit=loads(plan,resources)
    assert (audit.carga_segundos<=audit.segundos_disponibles+1e-7).all()
    plan.to_csv(out/('plan_'+label+'.csv'),index=False)
    audit.to_csv(out/('carga_'+label+'.csv'),index=False)
    summaries.append({'escenario':label,'produccion_mq':int(plan['Producción MQ propuesta'].sum()),
        'produccion_cp':int(plan['Producción CP propuesta'].sum()),'stock_final':int(plan['Inventario CP final'].iloc[-1]),
        'ventas_cp_no_cubiertas':int(plan['Ventas CP no cubiertas'].sum()),
        'mq_pendiente_final':int(plan['Maquila pendiente al cierre'].iloc[-1]),
        'max_utilizacion_pct':float(audit.utilizacion_pct.max())})
pd.DataFrame(summaries).to_csv(out/'resumen_escenarios.csv',index=False)
fig=make_subplots(rows=3,cols=1,subplot_titles=['Demanda MQ','Demanda CP','Producción CP descriptiva'])
for row,key in enumerate(['demanda_mq','demanda_cp','produccion_cp'],1):
    history=pd.read_csv(Path('outputs/lstm_principal_v2')/(key+'_serie.csv')).tail(26)
    fig.add_scatter(x=history.fecha_inicio,y=history.valor,name=key+' observado',row=row,col=1)
    fig.add_scatter(x=f.fecha_inicio,y=f[key],name=key+' LSTM',row=row,col=1)
    if key.startswith('demanda'):
        fig.add_scatter(x=high.fecha_inicio,y=high[key],name=key+' escenario alto',line={'dash':'dot'},row=row,col=1)
fig.update_layout(height=1000,title='Pronósticos y escenario alto basado en errores de validación; no intervalos garantizados',template='plotly_white')
fig.write_html(out/'historial_pronostico_escenarios.html',include_plotlyjs=True)
resources.to_csv(out/'recursos_fuente.csv',index=False); checks.to_csv(out/'auditoria_capacidad.csv',index=False)
(out/'LEEME.md').write_text('''# Resultados v4

Pronóstico: modelos elegidos por RMSE de validación; demandas sin reajuste y producción
reentrenada con historial completo y épocas congeladas. Las métricas de prueba
corresponden a redes anteriores a ese reajuste y se conservan intactas.
La prueba ya había sido observada: no es una nueva evaluación ciega.

Escenario alto: se agrega por horizonte el percentil 90 del error real-pronóstico
observado en VALIDACIÓN, acotado inferiormente en cero. Es un escenario de tensión,
no un intervalo con cobertura del 90% garantizada. Hay solo 10/19 orígenes solapados.

Plan: inventario inicial 698 confirmado; lavado inicial 0 como supuesto editable;
recepción de lo fabricado la semana siguiente; cobertura 4 semanas, cuota CP 35%,
lote mínimo orientativo 80. Esos valores son políticas, no mediciones.
Capacidad por máquinas y trabajadores de 03_CAPACIDADES.xlsx. Sin secuenciación
intrase semanal ni validación de capacidad de lavado. No optimiza costos.
El pronóstico estadístico de producción no interviene en las cantidades propuestas.
''',encoding='utf-8')
assert hashlib.sha256(capacity.read_bytes()).hexdigest()==hashlib.sha256(payload).hexdigest()
print(pd.DataFrame(summaries).to_string(index=False))
