# Plan ejecutable

1. tests/test_lstm_horizonte.py: verificar que modificar TEST no cambia selección,
   que el entrenador recibe solo TRAIN y que las épocas son las predefinidas.
2. lstm_validacion_horizonte.py: entrenador original/log/hurdle, selección por
   validación recursiva, reutilizar métricas y contratos Candidate existentes.
3. scripts/experimento_horizonte.py: ejecutar con fuente real de solo lectura,
   congelar resultados de validación antes de TEST y generar comparativas/futuro.
4. Integrar en app únicamente si el experimento justifica el cambio. Mantener
   referencias y resultados históricos; declarar falta de prueba ciega nueva.
5. Ejecutar tests y recorrido AppTest real, exportar informe y comprobar rama.
