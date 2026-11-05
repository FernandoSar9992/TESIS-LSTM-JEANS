# TESIS-LSTM-JEANS

Aplicación para el pronóstico semanal de demanda de jeans en las modalidades de **maquila** y **corte propio**, con integración posterior a la planificación agregada y al plan maestro de producción.

## Inicio rápido

1. Instalar dependencias: `pip install -r requirements.txt`.
2. Ejecutar: `streamlit run app.py`.
3. Cargar el libro histórico actualizado con una hoja denominada `DEMANDA`.
4. Confirmar explícitamente los cinco períodos iniciales sin ventas de corte propio solo si la fuente del taller respalda esa ausencia.
5. Entrenar, revisar VALIDACIÓN, evaluar TEST y generar el pronóstico.

## Modelación

- LSTM recursiva como modelo de referencia.
- LSTM directa multi-horizonte, con 12 salidas semanales simultáneas.
- Comparación con orígenes temporales equivalentes; selección exclusiva con VALIDACIÓN.
- Transformación log1p y pérdida Huber para la variante directa.
- Calibración de sesgo de corte propio estimada únicamente con VALIDACIÓN.
- Reajuste operativo con arquitectura y épocas congeladas.

## Documentación

- [Protocolo metodológico LSTM](docs/metodologia_lstm_final.md)
- [Experimento de horizonte v4](docs/experimento_horizonte_v4.md)
- [Resultados de horizonte y capacidad](docs/resultados_horizonte_y_capacidad_v4.md)
- [Guía de LSTM dual](GUIA_DUAL_LSTM.md)

## Advertencia metodológica

La LSTM directa es una alternativa experimental y no se considera superior por definición. Su adopción como modelo final depende de los resultados obtenidos con la base histórica vigente. No se deben modificar los valores reales, recortar picos para mejorar métricas ni utilizar TEST para seleccionar o calibrar el modelo.
