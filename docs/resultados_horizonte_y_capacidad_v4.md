# Revisión LSTM y capacidad compartida — 2026-11-04

## Alcance y decisiones

Se ejecutó la parrilla predefinida de `experimento_horizonte_v4.md` con la base
real y semilla fija. Las tres selecciones se guardaron antes de calcular prueba.
No se volvió a buscar tras observar sus resultados. Los modelos principales
son LSTM; las referencias siguen visibles. La validación es de desarrollo,
con orígenes solapados. La prueba ya había sido observada en revisiones previas:
no se presenta esta comparación como nueva evaluación ciega.

| Serie | Configuración elegida en validación | RMSE prueba v3 | RMSE prueba v4 | MAE v3 | MAE v4 |
|---|---|---:|---:|---:|---:|
| Demanda MQ | log1p, ventana 12, 16 unidades, 30 épocas | 32.9092 | 22.6802 | 26.3560 | 19.2586 |
| Demanda CP | log1p, ventana 12, 16 unidades, 100 épocas | 17.4883 | 18.4223 | 15.5249 | 16.4627 |
| Producción CP | hurdle LSTM, ventana 12, 16 unidades, 100 épocas | 114.2274 | 109.2033 | 76.7045 | 86.5239 |

MQ mejora en esta comparación. CP demanda empeora. Producción mejora RMSE pero
empeora MAE. R² de prueba: MQ 0.1247, demanda CP -1.1898, producción CP -0.0089.
No existe evidencia de precisión alta para las tres series. No se afirma una
mejora general ni se cambian modelos retrospectivamente por estos indicadores.

## Matemática de producción

La LSTM comparte una representación recurrente y tiene dos salidas:
p(X)=sigmoid(a), q(X)=softplus(b). La primera estima probabilidad de producción;
la segunda cantidad positiva, escalada por RMS de TRAIN. La pérdida por muestra
es BCE(1[Y>0],p) + 1[Y>0]*(Y/s-q)^2/pi_train. El divisor pi_train equilibra la
cantidad frente a la frecuencia de ceros; no multiplica artificialmente el futuro.
La predicción es s*p*q, conforme a E[Y|X]=P(Y>0|X)*E[Y|Y>0,X].
No predice fechas exactas de lotes ni convierte probabilidades en eventos ficticios.
Las LSTM log1p/Huber de demanda son pronósticos puntuales: no se afirma que su
inversión sea exactamente la media condicional en unidades originales.

## Capacidad e inventario

Fuente de solo lectura: `03_CAPACIDADES.xlsx`, hoja CAPACIDAD. Se leen tiempos
estándar C:D, horas reales N de máquinas (filas 10:23) e I de trabajadores
(26:33). Se omite REC4 porque no tiene operación registrada, no porque se le
asigne capacidad adicional. Se verificaron 38 cocientes horas*3600/tiempo contra
los resultados guardados del Excel; todos coinciden. Son 21 recursos activos.

Cada semana se impone, para todo recurso r:
TE_MQ,r * Q_MQ + TE_CP,r * Q_CP <= 3600 * horas_r.
No se suman las capacidades exclusivas (604 MQ o 256 CP). Con 320 MQ, el límite
remanente es 151 CP en la mezcla leída. No se usa 420 como capacidad del Excel.

El usuario confirmó 698 CP disponibles para vender. La app inicializa inventario
con el último valor registrado y permite corregirlo. Las políticas de cobertura,
cuota CP y lote orientativo son escenarios, no mediciones de capacidad. El lavado
se supone de una semana; no está validada su capacidad externa. El plan no resuelve
precedencias intraseanales ni optimiza costos. Se reporta como plan preliminar.

## Archivos

- `lstm_validacion_horizonte.py`: búsqueda, doble salida, configuración congelada.
- `configuracion_lstm_validada.json`: arquitectura/épocas elegidas; sin registros.
- `capacidad_recursos.py`: lectura y auditoría del Excel, cargas por recurso.
- `planificacion_reposicion.py`: restricciones mixtas, preserva modo equivalente.
- `app.py`: perfil validado, inventario, carga de capacidades y exportación.
- `scripts/experimento_horizonte.py`: reproducción del experimento completo.
- `tests/test_lstm_horizonte.py`, `tests/test_capacidad_recursos.py`: causalidad,
  reajuste congelado y restricciones productivas; se mantienen pruebas anteriores.

La configuración congelada solo se activa con el hash exacto de la base auditada,
los ceros iniciales confirmados y los ajustes manuales originales. Otra base
requiere ejecutar de nuevo validación, no heredar una supuesta mejora.

## Verificación y escenarios exportados

17 pruebas específicas pasan (12 protocolo, 3 horizonte, 2 recursos). El recorrido
AppTest con los dos Excel termina entrenamiento, prueba, pronóstico, Excel y plan.
Los indicadores de las tres selecciones se reproducen entre experimento y app
con tolerancia 1e-7, y permanecen intactos después del pronóstico final.

Con stock 698, lavado inicial supuesto 0, cobertura 4, cuota CP 35% y lote
orientativo 80: escenario base fabrica 3699 MQ y 63 CP; escenario alto, 4604 MQ
y 561 CP. Ambos sin ventas CP descubiertas ni pendientes MQ finales bajo esos
supuestos. Uso máximo por recurso: 58.15% y 90.96%, respectivamente.

El escenario alto agrega por horizonte el percentil 90 positivo del error de
validación. No es un intervalo con cobertura garantizada ni una nueva LSTM.
Los archivos de `outputs/integracion_v4` contienen futuro definitivo (producción
reajustada), planes y gráficos. El CSV inicial de `outputs/experimento_horizonte_v4`
conserva producción sin reajustar; es diagnóstico del experimento, no el entregable
final. Se conserva para trazabilidad, sin sustituir métricas.

La suite heredada completa tenía 14 fallos previos documentados; esta revisión
verifica los tests específicos y el flujo real, no declara toda la suite libre de
fallos. No se modificaron los dos Excel originales.
