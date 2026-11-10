# Auditoría de validación cruzada V9 (rama separada)

Esta auditoría NO modifica app.py, lstm_residual_estacional.py,
los parámetros ni los pronósticos originales de la LSTM V9.

## Abrir

Instala dependencias con: pip install -r requirements.txt
Ejecuta: streamlit run app.py
Abre la página 'Validacion Cruzada V9' en el menú de Streamlit.
Alternativamente: streamlit run pages/5_Validacion_Cruzada_V9.py
Usa el mismo Excel de V9 (hoja DEMANDA), selecciona modalidad y ejecuta.
Al finalizar, exporta cinco tablas a Excel y revisa loss/val_loss.

## Método

- Folds expansivos cuyos objetivos completos de 12 semanas terminan ANTES
  del comienzo de TEST. Los objetivos externos no se solapan.
- Selección anidada con 12 semanas anteriores a cada fold; se elige por RMSE
  y MAE, se calibra el sesgo solo con el pasado y luego se reentrena.
- Tres semillas, parada temprana, dropout y L2 según la arquitectura V9.
- Evaluación de horizonte fijo de 12 semanas sin introducir objetivos
  intermedios. Comparación con Naive estacional 48 y 52.
- Curvas loss/val_loss de las redes piloto; no se modifican las redes
  empleadas por la aplicación principal.

## Limitaciones importantes para la tesis

1. Un histórico corto puede permitir un solo fold. No es evidencia de
   estabilidad entre varios cortes; la página lo advierte explícitamente.
2. Las ventanas de entrenamiento pueden solaparse; los objetivos externos no.
3. Los registros imputados o reconstruidos no constituyen demanda real
   observada. Las métricas globales deben interpretarse según su procedencia.
4. El TEST del desarrollo original ya se había consultado, como consta
   en configuracion_lstm_validada.json; la auditoría NO lo utiliza.
5. La validación cruzada no demuestra ausencia absoluta de sobreajuste:
   revisar curvas, métricas por fold y datos realmente observados.
6. La auditoría no reemplaza el modelo V9 ni vuelve a elegir pronósticos.

## Pruebas unitarias

python -m unittest discover -s tests -p test_cv_v9_diagnostico.py -v
Comprueba cronología, folds sin superposición y ausencia de influencia de
los valores de TEST en cualquiera de las salidas de la auditoría.