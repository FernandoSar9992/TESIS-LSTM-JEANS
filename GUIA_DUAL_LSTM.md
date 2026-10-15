# Pronóstico dual LSTM (rama de desarrollo)

## Objetivo

Pronosticar **maquila** y **corte propio** mediante dos ejecuciones independientes del motor LSTM existente (`thesis_engine.py`), y sumar sus pronósticos por período. La producción y el inventario NO son variables objetivo de la LSTM: corresponden a la planeación agregada.

## Inicio rápido

1. En **GitHub Desktop**, usa **Fetch origin** y cambia a la rama `mejora-lstm-maquila-corte`. Descarga los cambios con **Pull origin** si aparece.
2. Abre la carpeta del repositorio nuevo en VS Code / Codex.
3. Activa tu entorno virtual local, o crea uno nuevo, e instala las dependencias con `pip install -r requirements.txt` (TensorFlow tarda más que las otras dependencias).
4. Inicia `streamlit run dual_app.py` desde esa misma carpeta.
5. Carga `BASE_DEMANDA_LIMPIA_LSTM.xlsx` (recomendado), `DEMANDA_JEANS_TESIS.xlsx` o `BASE_DEMANDA_MAQUILA_CORTE_PROPIO_LSTM_TESIS.xlsx`. Selecciona `01_DEMANDA_LSTM`; la aplicación lee los encabezados de la fila 5. La pestaña `BASE_LIMPIA` del primer libro es una vista para consulta, no la entrada técnica con trazabilidad.
6. Entrena las dos LSTM y revisa VALIDACIÓN; después abre TEST una única vez para esa base/configuración y descarga la hoja de pronósticos.

La aplicación histórica sigue en `app.py` y se puede ejecutar por separado con `streamlit run app.py`. **No se ha fusionado esta rama con `main`.**

## Contratos de datos

Se exigen columnas `Periodo (4/mes)`, `DEMANDA MAQUILA LSTM`, `Origen maquila`, `DEMANDA CORTE LSTM`, `Origen corte`, `DEMANDA TOTAL LSTM` y `Bloque temporal`. Los objetivos deben ser finitos y no negativos. La LSTM de maquila usa 180 períodos (enero 2023–septiembre 2026) y la de corte propio 113 períodos (mayo 2024-Se4–septiembre 2026). Los historiales tienen duraciones distintas y el TEST del total se evalúa únicamente en fechas coincidentes. Los períodos son consecutivos con **4 segmentos por mes y 48/año**. Se excluyen filas históricas anteriores a la primera etapa conjunta y filas FUTURO sin demanda, pero no se rellenan ni sustituyen valores.

- Maquila: `ORIGINAL` se etiqueta `REAL`; `SIMULADA` es `IMPUTADO`; `ORIGINAL_REVISAR` se deja `SIN_DECLARAR`.
- Corte propio: `SOLO ESCENARIO` es `IMPUTADO`; `PARCIAL REGISTRADO` y `PARCIAL+ESCENARIO` son `MIXTO`. Un registro parcial NO prueba que la demanda TOTAL del período sea observada.
- Las cantidades de compras de tela indican potencial material, no unidades producidas realmente. Esta versión no entrena con producción ni inventario.

## Interpretación científica

La evaluación descrita como «todos los objetivos» combina escenarios y registros parciales: es **descriptiva**, no validación de demanda real. El total solo tendrá métricas `SOLO_REAL` si al menos dos objetivos TEST de ambas modalidades están clasificados como `REAL`. Con la base reconstruida actualmente eso podría no ocurrir.

`RMSE`, `MAE` y `R²` del total se calculan con la suma de las predicciones en el mismo período: **no son la suma ni el promedio de las métricas individuales**. No se fuerza ninguna métrica objetivo. El motor conserva los cortes cronológicos y no debe ajustarse después de mirar el TEST.

## Pruebas y límites de esta entrega

Se añadió `tests/test_dual_forecast.py` y un flujo de GitHub Actions para comprobar importación, secuencia temporal, sumas y procedencia sin instalar TensorFlow. **Esto no reemplaza el entrenamiento real**: el entrenamiento de ambas redes, la interfaz Streamlit y las descargas deben probarse en tu PC con TensorFlow y el Excel concreto; evalúa también tiempo de ejecución y memoria. No se han comprobado todavía las métricas del modelo con la nueva base.

## Integración con Codex

Abre la rama de desarrollo en VS Code con Codex y pide: «Revisa `dual_app.py`, `dual_forecast.py` y las pruebas. Ejecuta `python -m unittest discover -s tests -p test_dual_forecast.py -v` y `streamlit run dual_app.py`. Corrige fallos reproducibles en esta rama, preserva `main`, no modifiques la demanda para mejorar métricas y no abras repetidamente TEST».

No subas `.env`, archivos con datos confidenciales del taller ni `.streamlit/secrets.toml`. El repositorio debe permanecer **privado**.

## Presentación de demanda

La interfaz principal muestra período, demanda de maquila, demanda de corte propio y total. La información metodológica sobre origen y reconstrucción se conserva en el desplegable **Calidad de los datos y notas metodológicas**, sin alterar ni reclasificar los registros. Las observaciones anteriores a mayo de 2024-Se4 no se tratan como demanda real cero de corte propio.
