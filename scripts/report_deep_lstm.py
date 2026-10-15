"""Informe A–J desde resultados congelados; no selecciona ni entrena modelos."""
from pathlib import Path
import sys
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import json
import numpy as np
import pandas as pd
import plotly.express as px

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def table(frame):
    def cell(value):
        if isinstance(value,(float,np.floating)):
            return '—' if not np.isfinite(value) else f'{value:.4f}'
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    audit=ROOT/'results/experiments/deep_audit_2026-10-07'
    out=ROOT/'results/experiments/nested_lstm_2026-10-07'
    model=out/'modelo'
    meta=json.loads((model/'metadata.json').read_text(encoding='utf-8'))
    frozen=json.loads((out/'selection_frozen.json').read_text(encoding='utf-8'))
    scores=pd.read_csv(model/'split_metrics.csv')
    cv=pd.read_csv(out/'cv.csv')
    history=pd.read_csv(audit/'historical_folds.csv')
    baseline_names=['Último valor','Media móvil 8','Media móvil 24','Naive estacional','Ridge 6']
    benchmark=history[history.Variante.isin(baseline_names)]
    allcv=pd.concat([cv,benchmark],ignore_index=True)
    measures=['MAE','RMSE','MAPE_%','MSE','R2','MASE','RMSSE']
    means=allcv.groupby('Variante',sort=False)[measures].mean().reset_index()
    means.to_csv(out/'cv_with_benchmarks.csv',index=False)
    allcv.groupby('Variante')[measures].agg(['mean','median','std','min','max']).to_csv(out/'all_cv_statistics.csv')
    principal=meta['thesis_selected']; original='LSTM 8 ventana 24 residual seasonal'
    compared=scores[scores.Modelo.isin([original,principal])]
    test=scores[scores.Grupo.eq('TEST') & scores.Modelo.isin([original,principal,*baseline_names])]
    detail=pd.read_csv(model/'detail.csv')
    one=detail[detail.Modelo.eq(principal)&detail.Grupo.eq('TEST')&detail.Horizonte.eq(1)].copy()
    # detail contiene solo orígenes con trayectoria completa: no confundir sus
    # 20 filas H1 con los 27 objetivos de la evaluación de un paso.
    from verify_workshop_artifacts import load_artifact
    from workshop_engine import predict
    data=pd.read_csv(model/'processed.csv')
    for path in model.glob('models_*.json'):
        if json.loads(path.read_text(encoding='utf-8'))['name']==principal:
            _,artifact=load_artifact(path,data)
            break
    idx=np.arange(meta['val_end'],len(data))
    prediction=predict(artifact,data,idx,np.ones(len(idx),int))
    one=pd.DataFrame({'Periodo':data.periodo.iloc[idx].to_numpy(),'Real':data.demanda.iloc[idx].to_numpy(),
                      'Prediccion':prediction})
    one['Error']=one.Prediccion-one.Real;one['Absoluto']=one.Error.abs()
    one.to_csv(out/'selected_one_step.csv',index=False)
    y=data.demanda.to_numpy();targets=y[idx]
    independent={'MAE':np.mean(abs(prediction-targets)),'MSE':np.mean((prediction-targets)**2),
                 'RMSE':np.sqrt(np.mean((prediction-targets)**2)),
                 'MAPE_%':np.mean(abs((prediction-targets)/targets))*100,
                 'R2':1-np.sum((prediction-targets)**2)/np.sum((targets-targets.mean())**2)}
    saved=test[test.Modelo.eq(principal)].iloc[0]
    for key,value in independent.items():np.testing.assert_allclose(value,saved[key],atol=3e-4,rtol=1e-5)
    altered=data.copy();altered.loc[153:,['demanda','precio_unitario','costo_unitario']]=1e10
    altered.loc[153:,'cliente']='FUTURO'
    np.testing.assert_array_equal(predict(artifact,data,np.full(8,153),np.arange(1,9)),
                                  predict(artifact,altered,np.full(8,153),np.arange(1,9)))
    (out/'verification.json').write_text(json.dumps({'independent_metrics':independent,'causal_future_test':True},indent=2),encoding='utf-8')
    peak=[]
    for label,mask in [('90% inferior',one.Real<=one.Real.quantile(.9)),('10% superior',one.Real>one.Real.quantile(.9))]:
        e=one.loc[mask,'Error']
        peak.append({'Segmento':label,'N':len(e),'MAE':e.abs().mean(),'RMSE':np.sqrt((e**2).mean()),
                     'Sesgo':e.mean(),'Max_abs':e.abs().max()})
    pd.DataFrame(peak).to_csv(out/'selected_peaks.csv',index=False)
    figs=[px.line(one,x='Periodo',y=['Real','Prediccion'],markers=True,title='LSTM seleccionada: prueba retrospectiva de un paso'),
          px.line(one,x='Periodo',y='Error',markers=True,title='Residuos de la LSTM seleccionada'),
          px.bar(one,x='Periodo',y='Absoluto',title='Errores absolutos'),
          px.line(allcv,x='Corte',y='RMSE',color='Variante',markers=True,title='Walk-forward: LSTM y referencias'),
          px.bar(test,x='Modelo',y='RMSE',title='Comparación retrospectiva: mismos 27 objetivos'),
          px.bar(pd.DataFrame(peak),x='Segmento',y=['MAE','RMSE'],barmode='group',title='Error en demandas altas')]
    html='<html><meta charset="utf-8"><title>LSTM: resultados finales</title><body><h1>Evaluación retrospectiva; selección congelada por CV</h1>'
    for i,fig in enumerate(figs):html+=fig.to_html(full_html=False,include_plotlyjs=i==0)
    (out/'comparison.html').write_text(html+'</body></html>',encoding='utf-8')
    final=saved
    report=f'''# Auditoría profunda LSTM — informe A–J

Fecha: 7 de octubre de 2026. Procedimiento Superpowers systematic-debugging: diagnóstico → evidencia → hipótesis → experimento → comparación → corrección → verificación.

## A. Diagnóstico de causa raíz

**El deterioro está dominado por desviaciones del patrón anual y picos no anticipados, no por un límite del escalado ni por una salida recortada al máximo de TRAIN.** Dos errores explican el 79,4057 % del SSE oficial de TEST: mayo 2026-Se1, 15,1070 %, y julio 2026-Se4, 64,2987 %. Esta atribución es una descomposición matemática, no prueba de la causa comercial de los pedidos.

En TRAIN, la diferencia respecto al mismo período del año anterior tiene media 7,065, mediana 7 y desviación 1,681 unidades. En TEST su mediana es 5, pero su desviación aumenta a 74,353. La ACF de rezago 48 es 0,999987 en TRAIN frente a 0,930951 en toda la serie. El patrón anual existe, pero las desviaciones finales son mayores. Las medias de bloques también difieren; por cubrir distintos meses, no prueban por sí solas un cambio estructural. No hay suficientes ciclos posteriores para confirmar concept drift sostenido.

El R² 0,940 de VALIDATION no era excepcionalmente bueno frente a los cortes anteriores: los R² multihorizonte históricos eran aproximadamente 0,9991; 0,9993; 0,9720. Son tareas y objetivos distintos del R² de un paso. La validación histórica seleccionaba además épocas en el mismo bloque puntuado; se corrigió esta dependencia mediante un bloque interno anterior.

{table(pd.read_csv(audit/'distribution.csv'))}

Hay 180 observaciones, 125 filas TRAIN, 28 VALIDATION y 27 TEST. Los 101 de la pantalla son objetivos de TRAIN después de la ventana, no 101 filas originales. Para la red de ventana 24 hay 94 orígenes, 752 pares origen/horizonte y 101 objetivos distintos. Se solapan: no son muestras independientes. Cada red tiene 1.313 parámetros entrenables; ratios 1,746 por par, 13,968 por origen y 13 por objetivo distinto. El ensamble tiene tres redes, 3.939 parámetros en total. El tamaño efectivo estadístico no puede deducirse contando ventanas; la dependencia temporal lo reduce.

## B. Errores de código/metodología y correcciones

| Archivo / función | Evidencia y efecto | Corrección / estado |
|---|---|---|
| workshop_engine.py, fit_neural, líneas 200–221; train_workshop antes del cambio | EarlyStopping y argmin consultaban el bloque que después se puntuaba. Validación de selección, optimista como evaluación independiente; TEST no entraba en ese ajuste. | fit_temporal_neural, línea 227: piloto interno y reajuste del prefijo con épocas fijas. train_workshop, línea 354, usa esta ruta. |
| workshop_engine.py, score, línea 140 | Faltaba RMSSE solicitado. MASE existente era correcto, pero no equivalente a comparar modelos en TEST. | Añadido RMSSE y escala cuadrática de TRAIN; indefinido cuando falta historia o escala cero. |
| workshop_app.py, pestaña Métricas | Faltaban MASE, RMSSE y sesgo en el resumen. | Se muestran junto con explicación de denominadores y signo. Exportación conserva las métricas. |
| scripts/nested_lstm_experiment.py | Necesidad de reproducir protocolo y selección sin veto posterior por TEST. | Se guarda protocolo, todas las semillas/cortes, pesos por corte y selection_frozen.json antes de TEST. Fuente verificada durante auditoría; comprobación directa de hashes añadida al ejecutor para futuras corridas. |
| verify_workshop_artifacts.py | Inicio TRAIN fijo en 12 detectado en auditoría previa. | Ya corregido anteriormente: intervalos guardados; no era causa del R² bajo. |

Comprobaciones de pipeline: fuente Excel coincide con hash histórico y datos procesados; no hay nulos en entradas ni etiquetas duplicadas; calendario consecutivo de 48 períodos/año; no hay imputación automática en modo taller; no se filtra por cliente; objetivos son origen+h−1; ventanas terminan en origen−1. Escalas y vocabulario se ajustan al prefijo. No hay shuffle temporal. La inversión es base causal + residual normalizado × desviación de TRAIN. La media no se suma otra vez porque se aprende un residual. Las cinco métricas fueron reconstruidas mediante una segunda implementación y con los mismos objetivos. MAPE excluiría ceros según el contrato general, pero esta serie no tiene ceros: aquí usa exactamente los mismos 27 objetivos.

Salida Dense(1) lineal; solo se impone cota inferior cero. No hay MinMaxScaler ni techo 573. El 735 equivale a z=4,152843 usando media 378,672 y escala 85,803. Las capas internas tanh están acotadas, por lo que su extrapolación aprendida es limitada, pero no existe un recorte numérico a 573. No hay evidencia de fallo del inverse transform.

## C. Ranking de causas e hipótesis

| Orden / hipótesis | A favor | En contra o límite | Prueba / conclusión |
|---|---|---|---|
| 1. Picos no aprendidos | Dos períodos aportan 79,4 % del SSE; delta anual hasta 317 | Son observaciones declaradas reales | Descomposición SSE y comparación anual: confirmado como fuente principal del error |
| 2. Información insuficiente sobre pedidos | Cliente B y precio 1,5 también existen en períodos ordinarios | No hay tipo de pedido ni eventos previos registrados | No se puede identificar la causa comercial; variables omitidas son hipótesis, no certeza |
| 3. Mayor variabilidad de desviaciones estacionales | SD de delta anual 1,68→74,35 | Los bloques son cortos y cubren meses distintos | Cambio descriptivo confirmado; cambio estructural sostenido no demostrado |
| 4. Pocos datos independientes | 3,75 ciclos y ventanas solapadas | La estacionalidad es muy informativa | Limita estimación de eventos raros; no justifica prometer casi exactitud |
| 5. Validación optimista por selección de épocas | El mismo bloque elegía early stopping y puntuaba | No había contaminación directa de TEST al entrenar | Control anidado empeora su media CV frente al protocolo histórico; corrección metodológica aplicada |
| 6. Overfitting / arquitectura excesiva | 1.313 parámetros/red para 101 objetivos distintos | Épocas originales 2/3/2; red de 4 unidades no mejora consistentemente | Riesgo, no causa demostrada exclusiva; no se afirma overfitting solo por caída de R² |
| 7. Ventana o pérdida | Experimentos aislados 12/24 y MSE/MAE/Huber | Ninguno garantiza anticipar picos nuevos | Elegir por CV externa; resultados en D |
| 8. Escalado, alineación, techo de salida | Hipótesis razonables a comprobar | Reconstrucción y perturbación causal correctas; salida lineal | No respaldadas como causa en la ejecución auditada |

## D. Experimentos y ablations

Seis configuraciones × tres cortes × tres semillas; 54 evaluaciones de redes, cada una con piloto interno y reajuste. TRAIN externo termina en 69, 97 y 125; bloques externos terminan en 97, 125 y 153. El último 20 % del prefijo, con al menos H+1 filas, elige épocas. Configuración común: una capa LSTM, residual anual, dropout 0,1, Adam 0,001, clipnorm 1, lote 32, máximo 180 épocas y paciencia 20. Se cambia un factor respecto al control.

{table(means)}

Las medias son aritméticas por corte, no métricas de concatenar todos los pares. [Media, mediana, desviación, mínimo y máximo de todas las métricas](../../results/experiments/nested_lstm_2026-10-07/all_cv_statistics.csv). [Cada semilla y época](../../results/experiments/nested_lstm_2026-10-07/seeds.csv).

Selección registrada: {frozen['selected']}. Se exige no empeorar MAE/RMSE medios ni peor RMSE, y mejorar RMSE en al menos dos cortes frente al control. No se eligió por mayor R² ni por TEST. Se comparó además con las referencias en los mismos cortes; los objetivos MASE/RMSSE se informan aunque no se cumplan en cada corte. La desviación entre cortes sigue siendo considerable: no se declara estabilidad completa porque el tercer corte es mucho más difícil.

{table(cv[cv.Variante.eq(frozen['selected'])][['Corte','MAE','RMSE','MAPE_%','R2','MASE','RMSSE']])}

No se amplió a 32/64 unidades, varias capas ni una parrilla de regularización/learning rate/batch size: la red más pequeña no dio mejora consistente y hay pocos ciclos. Se mantienen esos factores constantes para atribuir los efectos y evitar sobreajustar CV. Los ensayos previos de entrenamiento desde el primer ciclo completo siguen archivados, pero no determinan esta selección. RF/ARIMA/ETS no fueron necesarios para las referencias obligatorias; no se instalaron dependencias nuevas. Ridge aporta el benchmark de regresión.

MSE penaliza cuadráticamente errores grandes; MAE linealmente; Huber combina ambos regímenes (delta=1 en unidades del residual estandarizado). Una pérdida robusta no revela cuándo ocurrirá un pedido extraordinario. Se eligió MAE por CV, no por su comportamiento en TEST. [Definiciones oficiales de Keras](https://keras.io/api/losses/regression_losses/).

### Variables disponibles antes del origen

| Variable | Justificación | Disponible | Riesgo | Resultado CV / decisión |
|---|---|---|---|---|
| Rezagos t−1,t−2,t−3 y ventana | Dinámica reciente | Sí, respecto al origen | Bajo si se desplazan | Ya en secuencia; ventanas 12/24 ensayadas |
| Media 8/24 y SD de ventana | Nivel y variabilidad | Sí | Bajo | Ya incluidas; conservar |
| Rezago 48 y calendario seno/coseno | Patrón anual y posición de cuatro períodos/mes | Sí, solo si rezago anterior al origen | Bajo | Referencia estacional fuerte; conservar |
| Mes/trimestre | Posición temporal | Sí | Bajo | Parcialmente representados por calendario; no duplicar sin ablation |
| Semana ISO / 52 semanas | Calendario civil | No corresponde a estas etiquetas | Frecuencia incorrecta | Descartar; son 48 períodos/año |
| Cliente y precio históricos | Contexto previo | Sí | Bajo si solo históricos | Ablation sin comerciales peor CV medio; conservar conjuntamente, sin inferir efecto causal individual |
| Cliente/precio del pedido futuro | Puede explicar tamaño de pedido | No consta anticipación | Alto | No usar como entrada futura |
| Costo unitario constante | No varía en TRAIN | Sí | Bajo | Escala activa cero; no aporta señal |
| Ingreso/costo total/utilidad objetivo | Revelan cantidad objetivo | No | Fuga directa | Excluidos |
| Tendencia/mediana móvil | Adaptación robusta de nivel | Sí con historia previa | Bajo | Investigar en datos nuevos; no experimentado para evitar ampliar búsqueda |
| Tipo pedido/modalidad/promoción/evento | Posibles causas de picos | No existen en archivo auditado | Alto si se registran después | Recopilar fecha de conocimiento; investigar, no inventar |

## E. Original frente a la solución con evaluación corregida

{table(compared[['Modelo','Grupo','N',*measures,'Sesgo']])}

Las filas TRAIN/VALIDATION/TEST son de un paso y usan los mismos objetivos para comparar modelos. La tabla CV de D usa trayectorias completas: 168 pares por corte; no comparar sus promedios directamente con el MAE de 28 objetivos a un paso. TRAIN es ajuste, VALIDATION sigue participando en elegir la configuración, TEST es retrospectivo ya consultado. No existe aquí una prueba prospectiva nueva.

## F. Evaluación final y cumplimiento de objetivos

{table(test[['Modelo','N',*measures,'Sesgo']])}

R² final seleccionado: {final.R2:.4f}; MAE {final.MAE:.4f}; RMSE {final.RMSE:.4f}; MAPE {final['MAPE_%']:.4f} %; MASE {final.MASE:.4f}; RMSSE {final.RMSSE:.4f}. No se alcanzaron R²≥0,75, MASE/RMSSE<1 ni reducciones de 20 % en MAE/RMSE. El MAPE≤10 % sí se cumple. La diferencia de precisión frente al original es pequeña y mixta: no se declara mejora predictiva final confirmada. No se cambió la selección por ese resultado.

MASE y RMSSE usan el error estacional de TRAIN, respectivamente medio absoluto y raíz del medio cuadrático. El denominador RMSSE es √52,7013≈7,2609 unidades; de ahí que un RMSE alrededor de 74 produzca RMSSE≈10. Esto no contradice superar ligeramente Seasonal Naive en el mismo TEST: son comparaciones con denominadores distintos. [Evaluación de pronósticos y errores escalados](https://otexts.com/fpp3/accuracy.html).

## G. Análisis de los picos

735: rezagos de la misma posición anual 402→410→418; la LSTM original produjo 418+10,305=428,305. La media de los ocho períodos previos era 335,375. El cliente es B, precio 1,5; no existe etiqueta de pedido extraordinario. 574: secuencia anual 400→408→416 y corrección original +9,340. Ambos exceden el umbral de Tukey definido en TRAIN, 573,5; 574 apenas lo supera, por lo que no equivale a declarar un error de datos. El 735 es mucho más extremo. Se conservan ambos como demandas reales declaradas por el usuario.

No se puede demostrar que sean imposibles de anticipar con cualquier modelo, ni que sean un cambio estructural. Sí se demuestra que el patrón anual y las variables disponibles no permitieron anticiparlos bien en los modelos evaluados. El contexto no respalda multiplicar artificialmente la demanda para alcanzar 735.

Resultados seleccionados para el 10 % más alto del bloque (aproximado por cuantiles, tres observaciones) y el resto:

{table(pd.DataFrame(peak))}

Sensibilidad **solo diagnóstica del modelo original**, sin modificar datos ni métricas oficiales:

{table(pd.read_csv(audit/'sensitivity_NOT_OFFICIAL.csv'))}

Al excluir observaciones se recalcula también la media y el denominador de R²; no interpretar esta sensibilidad como una precisión obtenida por el sistema ni como rendimiento garantizado en ausencia de picos.

## H. Gráficos y estadística reproducible

- [Diez gráficos del diagnóstico original](../../results/experiments/deep_audit_2026-10-07/diagnostic.html): serie completa, particiones, real/predicho, residuos temporales, distribución de residuos, error absoluto, ranking, distribuciones, ACF y walk-forward.
- [Seis gráficos de la evaluación corregida](../../results/experiments/nested_lstm_2026-10-07/comparison.html): real/predicho, residuos, errores, cortes, benchmarks y picos.
- [ACF/PACF por bloque](../../results/experiments/deep_audit_2026-10-07/acf_pacf_by_split.csv), limitadas a N/4 y máximo 12 rezagos: descriptivas, sin inferencia asintótica con 27/28 filas.
- [Patrones mensuales](../../results/experiments/deep_audit_2026-10-07/month_patterns.csv) y [posición dentro del mes](../../results/experiments/deep_audit_2026-10-07/within_month_patterns.csv).

No se aplica un test iid de distribución a observaciones autocorrelacionadas como si fueran independientes. Se reportan tamaños de efecto, cuantiles, dispersión y diferencias estacionales. Las pendientes OLS por bloque son descriptivas y están confundidas con la estacionalidad; no prueban tendencia estructural.

## I. Modelo final recomendado y alcance

Se recomienda la solución **LSTM con evaluación anidada**, manteniendo la selección realizada por CV: una LSTM de 8 unidades, ventana 24, Dense(16,tanh), dropout 0,1, salida Dense(1,lineal), residual estacional, pérdida MAE, Adam 0,001 con clipnorm=1, lote 32, máximo 180 épocas y paciencia 20. Épocas finales por semilla 42/43/44: {frozen['epochs']}. El reajuste de producción usa esas épocas fijas, todo el historial y escalas nuevas separadas. La configuración no se convierte en una garantía de precisión del reajuste.

La aplicación compara las seis alternativas mediante esta metodología y no fuerza un ganador para otras fuentes. Los objetivos de precisión no se alcanzaron en el bloque retrospectivo; la mejora demostrada es metodológica y de selección CV, no una mejora sustancial de precisión final. Usar como escenario base revisable, no como demanda cierta. Para medir mejoras futuras, congelar predicciones y reunir observaciones nuevas, con pedidos/eventos fechados según cuándo se conocían.

Los originales y sus pesos se preservan. No hay Git en esta carpeta: el respaldo del código original y hashes está en `results/experiments/deep_audit_2026-10-07/original_code/`; el código exacto del ensayo está en `nested_lstm_2026-10-07/protocol_code/`. No se presume reproducción en un entorno limpio instalado desde cero; se usó el entorno existente y se recargaron modelos guardados. La verificación de pruebas completa se registra al cierre de este informe.

## J. Conclusión para tesis

La evaluación evidenció que la disminución del coeficiente de determinación entre validación y prueba se relaciona principalmente con demandas que se apartan del patrón estacional aprendido, especialmente dos períodos que concentran aproximadamente el 79,4 % del error cuadrático de la prueba original. Se verificaron la separación cronológica, la causalidad de las variables, la reconstrucción de escala y el cálculo independiente de métricas, sin encontrar evidencia de un recorte de las predicciones al máximo observado durante entrenamiento. Para reducir el optimismo de la validación, se separó la selección de épocas mediante un bloque interno y se compararon seis configuraciones con tres cortes temporales y tres semillas. La LSTM con pérdida MAE fue seleccionada exclusivamente mediante validación temporal, pero su evaluación retrospectiva no mostró una mejora sustancial frente al modelo original ni alcanzó los objetivos de R² y errores escalados. Por tanto, la solución conserva utilidad como referencia de demanda estacional, aunque su capacidad para anticipar picos extraordinarios permanece limitada y requiere validación prospectiva e información explicativa disponible antes del pronóstico.
'''
    (ROOT/'docs/reports/auditoria_profunda_lstm_2026-10-07.md').write_text(report,encoding='utf-8')
    print('Informe A–J generado. Métricas independientes y prueba causal del modelo final: OK.')


if __name__=='__main__':main()
