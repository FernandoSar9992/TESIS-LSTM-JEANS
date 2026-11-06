# Motor V9: LSTM directa con contexto estacional — protocolo reproducible

## Alcance

Fuente: hoja `DEMANDA` del archivo `04_BASE_DE_DATOS(6).xlsx`. SHA-256:
`137bbc645825b7dce2acc810d06c632a1bf7807946638c762a888f39ae9d10ef`.

- 200 observaciones semanales de demanda de maquila.
- Corte propio: únicamente desde el inicio efectivo de producción. Se excluyen 62 semanas preoperativas.
- Las cinco semanas iniciales de corte propio sin venta se incorporan como cero exclusivamente por la confirmación explícita del usuario.
- La producción propia histórica es un análisis complementario, NO una orden de fabricación ni la demanda que alimenta la planificación.
- No se eliminan picos de demanda ni se alteran fechas o registros originales.

## Arquitectura y parámetros

Una LSTM directa multihorizonte pronostica la corrección residual, en log1p, respecto del mismo período de hace 52 semanas. Las entradas contienen únicamente historia observada anterior al origen, diferencias, rezagos anuales y características de calendario calculables anticipadamente. Un contexto por horizonte aporta promedios recientes, tendencia, volatilidad y características del período objetivo.

- Horizonte evaluado: **12 semanas**.
- Dos arquitecturas acotadas según controles de Streamlit: por defecto ventana 4 / 4 unidades y ventana 8 / 8 unidades.
- Tres semillas: 42, 43 y 44 por defecto.
- Pérdida Huber, Adam, dropout, regularización L2 y clipping de gradiente.
- Parada temprana en bloque cronológico **interno a TRAIN**. Se escogen épocas y se reentrena con todo TRAIN.
- Corrección de retransformación `log1p` estimada solo en bloque interno.
- La arquitectura se escoge por menor RMSE **sin calibración** del bloque VALIDACIÓN.
- Calibración de sesgo: media de `real - pronosticado` en VALIDACIÓN, congelada antes de TEST.
- La aplicación informa las métricas de VALIDACIÓN sin calibrar para selección y las calibradas para diagnóstico, por separado.

## Evaluación cronológica

Cada serie usa separación 70 % TRAIN, 15 % VALIDACIÓN y 15 % TEST por defecto. Se mantienen bloques completos de 12 semanas.

- VALIDACIÓN rolling-origin: cada pronóstico usa únicamente el contexto conocido en el origen. Los orígenes se solapan, por lo que sus pares (origen, horizonte) **no son observaciones independientes**.
- TEST rolling-origin: pesos reentrenados con TRAIN+VALIDACIÓN sin retocar épocas, unidades, arquitectura o sesgo. Cada origen actualiza el contexto con observaciones disponibles hasta ese momento; los pesos **no** se vuelven a ajustar entre orígenes.
- TEST fijo: un único pronóstico de 12 semanas emitido desde el primer origen de TEST, sin incorporar ventas reales intermedias.
- Referencias estacionales 48/52 se calculan sobre exactamente los mismos orígenes y horizontes.
- **Precaución:** el bloque TEST ya había sido examinado en iteraciones anteriores. Este es diagnóstico retrospectivo, no una prueba ciega nueva. Las mejoras no deben describirse como superioridad independiente verificada.

## Cómo probar en Codespaces

En una terminal situada en el repositorio, con rama `v9-lstm-directo-contexto-20261106`:

```bash
git fetch origin
git switch --track origin/v9-lstm-directo-contexto-20261106
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_lstm_residual_v9.py" -v
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Abra el puerto 8501 en el navegador. Cargue el Excel; confirme las cinco semanas sin venta; ejecute entrenamiento, VALIDACIÓN, TEST y pronóstico. No haga `Merge` hacia producción hasta verificar funcionamiento real de TensorFlow, consistencia de métricas y resultados de capacidad/PMP.

La app exporta las hojas `VALIDACION_12`, `VALIDACION_CALIBRADA`, `PRUEBA_12`, `TRAZA_TEST_12` y `SELECCION_FINAL`, además del pronóstico y planificación.

## Estado de evidencia

El prototipo exploratorio con Keras sobre backend PyTorch produjo resultados retrospectivos alentadores con una sola semilla y una formulación similar. No es idéntico al motor TensorFlow V9 de tres semillas. **No se trasladan sus métricas a la tesis como si provinieran del código final.** Deben evaluarse y guardarse las métricas V9 realmente producidas en el Codespace.
