# Restauración acotada de las demandas

Fecha de trabajo: 2026-11-03. Rama: lstm-produccion-final.

El usuario aclara que el problema original era producción CP, no las dos
demandas. El protocolo v2 también sustituyó el entrenamiento de las demandas:
épocas elegidas en una partición interna y refit obligatorio con todo el
historial. Esto explica cambios de resultados; no era una actualización de
interfaz ni únicamente una corrección de producción.

## Cambio

- `protocolo_final.demand_from_fit`: reutiliza exactamente red, escala y épocas
  del `fit_one` existente. No vuelve a entrenar las demandas en otro protocolo.
- Se conserva comparación recursiva a 12 semanas en todos los orígenes completos.
- La validación de demanda elige épocas y también informa errores: es desarrollo,
  no evaluación independiente. No se emplean errores de prueba para esa elección.
- Por defecto, el futuro conserva los pesos anteriores; el reajuste de demandas
  con todo el historial queda disponible mediante una casilla explícita después
  de prueba. No se alteran las métricas ya calculadas.
- Producción mantiene el protocolo v2, sin elevar cantidades arbitrariamente.
- Los cinco ceros iniciales de ventas confirmados por el usuario siguen vigentes.
  Por eso demanda CP puede diferir de la versión previa que omitía esas semanas.

## Limitaciones y trazabilidad

La restauración responde al alcance solicitado, no demuestra superioridad.
El conjunto de prueba ya fue observado y esta revisión no es una nueva evaluación
ciega. Los archivos anteriores en `outputs/lstm_principal_v2` se conservan.
Mayor variación del futuro no es criterio de selección ni prueba de exactitud.
La producción estadística sigue siendo descriptiva, nunca una orden de fabricación.

Verificación reproducible:
`python scripts/verificar_streamlit_final.py RUTA_EXCEL outputs/demandas_restauradas_v3`
El recorrido exige igualdad con `forecast_lstm_verified` para las dos demandas,
conservación de métricas después de pronosticar y ejecución de la planificación.

## Verificación terminada el 2026-11-04

12 pruebas específicas pasan. Recorrido real de Streamlit completo: carga,
entrenamiento, prueba, pronóstico, exportaciones y plan de reposición.
Se corrigió la unión de fechas `date` del motor anterior con `datetime64` del
protocolo de producción; el adaptador normaliza el tipo sin cambiar las fechas.

Comparación sobre los mismos orígenes de prueba completos (12 semanas):

| Serie | MAE v2 | MAE v3 | RMSE v2 | RMSE v3 |
|---|---:|---:|---:|---:|
| Demanda MQ | 30.9505 | 26.3560 | 37.6741 | 32.9092 |
| Demanda CP | 10.6318 | 15.5249 | 13.4691 | 17.4883 |
| Producción CP | 76.7045 | 76.7045 | 114.2274 | 114.2274 |

No existe mejora general: MQ mejora en este registro de prueba, demanda CP
empeora y producción no cambia. Los R² siguen negativos. No se usaron estos
resultados para elegir épocas ni para ajustar esta restauración.

MQ conserva 99 épocas. CP usa 38, frente a las 51 de la ejecución antigua que
omitía los cinco ceros iniciales de ventas; no se deshace la confirmación del
usuario para recuperar números. Los resultados nuevos se guardan separados en
`outputs/demandas_restauradas_v3`, junto con comparación y gráfico HTML.
