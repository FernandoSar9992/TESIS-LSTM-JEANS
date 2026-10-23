# Pronóstico semanal de demanda de pantalones jeans

Dos LSTM independientes para **demanda de maquila** y **demanda comercial de corte propio**; suma de ambos pronósticos y métricas cronológicas de TEST.

## Ejecutar desde VS Code

Con la rama `tesis-presentacion-pronostico` descargada mediante GitHub Desktop:

```powershell
& "C:\Users\ASUS\Downloads\FERNANDO.SARMIENTO.CODIGO\FERNANDO.SARMIENTO.CODIGO\.venv\Scripts\python.exe" -m streamlit run dual_app.py
```

## Formatos de entrada

- Nuevo Excel conciliado: `Periodo (semana)`, `Fecha_inicio`, `Fecha_fin`, `DEMANDA MAQUILA LSTM`, `DEMANDA CORTE LSTM`. Lee encabezados en cualquiera de las primeras diez filas.
- Las fechas semanales deben ser **lunes a viernes**, consecutivas sin saltos. Se1–Se5 siguen la posición del lunes de cada semana en su mes.
- Se excluyen automáticamente las semanas no concluidas a la fecha de corte (seleccionable, nunca posterior al día actual).
- Se conserva el soporte para el Excel anterior `Periodo (4/mes)` sin fechas, con cuatro segmentos mensuales.
- Valores anteriores al inicio de corte propio: dejar las celdas vacías. Después de su inicio, valores numéricos no negativos.
- `PRODUCCION CORTE REGISTRADA` es fabricación histórica y **no** se utiliza como objetivo de demanda. `CONFECCIÓN MAQUILA Y CORTE` es maquila más fabricación, no maquila más ventas.

## Interpretación estadística

El modelo utiliza un ciclo anual **aproximado de 52 semanas** para fechas semanales y no presume que un año contiene 48 semanas. Los años ISO con semana 53 introducen un pequeño desfase que conviene evaluar antes de la tesis definitiva.

La tabla de métricas TEST evalúa períodos históricos reservados, no pronósticos futuros. **Una base sintética, imputada o sin registros verificables únicamente permite una evaluación experimental**, no certificar la precisión real en el taller.

La planificación de producción de corte propio (inventario, reposición y capacidad compartida) será un módulo separado, aún no implementado en esta etapa.

## Mejora de las dos LSTM

Con `adaptive_search=True`, la selección temporal compara ventanas de 2 a 24 semanas, transformaciones local/nivel/cambio, pérdida MSE o Huber, ponderación reciente opcional y algunos modelos estacionales cuando existe historial. Los hiperparámetros se seleccionan únicamente en VALIDACIÓN. El TEST se evalúa después y no modifica la configuración. En pantalla se compara la LSTM con una referencia simple seleccionada por VALIDACIÓN; no se garantiza que la red neuronal la supere.

Respaldo anterior: rama `respaldo-presentacion-antes-optimizacion`. La ejecución integral con TensorFlow debe verificarse en el equipo local.
