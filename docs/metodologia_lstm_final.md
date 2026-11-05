# Protocolo final de pronóstico LSTM para el taller de jeans

## 1. Objetivo y series

El sistema pronostica por separado la demanda semanal de:
- **Maquila (MQ):** demanda atendida mediante confección de piezas previamente cortadas.
- **Corte propio (CP):** ventas de prendas producidas desde materia prima.

La demanda total se calcula como la suma de ambos pronósticos alineados por semana. La producción histórica de CP es una serie complementaria y no se confunde con demanda ni se convierte automáticamente en una orden de fabricación.

## 2. Preparación y trazabilidad

La aplicación recibe el libro Excel con la hoja `DEMANDA`. Se validan fechas semanales consecutivas de lunes a viernes, duplicados, cantidades no negativas, semanas cerradas y campos requeridos. No se imputan automáticamente faltantes.

Los cinco períodos iniciales de CP solo se convierten en cero cuando el usuario confirma explícitamente que no hubo ventas durante ese intervalo y la condición temporal coincide con el inicio de actividad registrado. La transformación queda documentada en la traza de preparación; los períodos anteriores al inicio de actividad permanecen fuera de la serie.

## 3. Modelos comparados

### 3.1 LSTM recursiva

Utiliza una ventana de las últimas semanas y predice un período. Para proyectar varios períodos, incorpora cada predicción a la ventana de entrada del siguiente paso.

### 3.2 LSTM directa multi-horizonte

Utiliza una ventana histórica y emite simultáneamente los siguientes 12 valores. La capa de salida tiene 12 neuronas, una por horizonte semanal. Para horizontes superiores a 12 semanas, el pronóstico avanza en bloques de 12 usando los valores ya emitidos.

Ambas arquitecturas se comparan como candidatas LSTM. No se presupone que la directa sea superior: la selección depende de la validación temporal de cada serie.

## 4. Entrenamiento sin fuga temporal

La separación se realiza cronológicamente, por defecto 70 % TRAIN, 15 % VALIDACIÓN y 15 % TEST. Los escaladores y parámetros de transformación se ajustan únicamente con TRAIN.

En la LSTM directa, cada ventana de entrenamiento debe cumplir:
- la entrada utiliza observaciones anteriores al origen;
- los 12 valores objetivo pertenecen íntegramente a TRAIN;
- ninguna ventana de entrenamiento incluye objetivos de VALIDACIÓN o TEST.

La selección de arquitectura, ventana y épocas se realiza con VALIDACIÓN. El bloque TEST no participa en selección, calibración ni ajuste de hiperparámetros.

## 5. Búsqueda experimental directa

Parámetros predefinidos:
- Horizonte de salida: 12 semanas.
- Ventanas de entrada: 8, 12 y 16 semanas.
- Unidades LSTM: valor definido en la configuración de entrenamiento (por defecto 16).
- Épocas candidatas: 10, 30, 60 y 100.
- Transformación: log(1 + demanda).
- Pérdida: Huber.
- Optimizador: Adam, tasa de aprendizaje 0,001 y clipnorm 1.
- Semilla reproducible: 42 por defecto.
- Dropout: valor definido en Settings (por defecto 0,15).

La búsqueda incluye las referencias y la LSTM recursiva disponible en el protocolo de control. Se comparan los mismos orígenes de VALIDACIÓN y el mismo horizonte de 12 semanas.

## 6. Criterios de evaluación

Indicador principal de selección: RMSE de VALIDACIÓN a 12 semanas. MAE se utiliza como criterio secundario de desempate y como medida de interpretación.

En VALIDACIÓN y TEST se reportan, cuando proceda:
- MAE: error absoluto medio, expresado en pantalones por semana.
- RMSE: raíz del error cuadrático medio, penaliza más los errores grandes.
- R²: proporción de variabilidad explicada respecto a la media de los valores observados del bloque evaluado.
- WAPE: error absoluto total relativo al volumen real.
- MASE: error absoluto escalado respecto a la variación media del TRAIN.
- Sesgo: promedio de predicción menos valor real; positivo indica sobreestimación y negativo subestimación.

Un R² bajo o negativo se informa sin ocultarlo ni modificar el bloque TEST. No se fijan rangos objetivo obligatorios para forzar la aceptación de un modelo.

## 7. Calibración de sesgo en corte propio

Cuando se activa para CP, la corrección aditiva se estima como el promedio de los residuos de VALIDACIÓN del candidato seleccionado (real menos pronosticado). Se conserva la evaluación original de VALIDACIÓN y se aplica la corrección al candidato antes de evaluar TEST y al reajuste operativo.

La calibración no utiliza residuos de TEST. Maquila no recibe esta corrección por defecto. La corrección se limita inferiormente a cero al emitir cantidades pronosticadas.

## 8. Evaluación final y pronóstico operativo

Después de seleccionar el candidato en VALIDACIÓN, se evalúa una sola vez sobre TEST. La evaluación rolling-origin de 12 semanas permite que cada nuevo origen use el historial real disponible hasta esa fecha; no incorpora observaciones posteriores al origen evaluado. Se conserva además el diagnóstico recursivo fijo desde el primer origen para distinguir ambos esquemas.

Para el pronóstico operativo se puede reajustar el modelo seleccionado con todo el historial disponible, manteniendo fijos la arquitectura, la ventana, las épocas y la calibración estimadas antes de consultar TEST. Las métricas de TEST no se recalculan ni se alteran por ese reajuste.

## 9. Integración con planificación

El pronóstico de demanda alimenta la planificación agregada y el plan maestro. La producción histórica estimada no sustituye las decisiones de fabricación. El plan de reposición considera inventario CP, lavado, tiempos estándar y capacidad compartida por recursos; las cantidades propuestas no pueden exceder la capacidad disponible por máquina o trabajador.

## 10. Alcance de la validación

Las pruebas automatizadas verifican integridad temporal, límites de ventanas, dimensiones de salida, ausencia de uso de TEST en selección, calibración basada en VALIDACIÓN y coherencia del pronóstico por bloques.

La superioridad predictiva de la LSTM directa no se afirma hasta ejecutar el entrenamiento TensorFlow con el archivo histórico vigente y comparar las métricas reales de VALIDACIÓN y TEST para MQ y CP. Las ventanas de backtesting con orígenes solapados no son observaciones independientes y deben interpretarse como evaluaciones rolling-origin.
