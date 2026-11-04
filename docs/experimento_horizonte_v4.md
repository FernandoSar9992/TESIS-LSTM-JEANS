# Experimento predefinido — 2026-11-04

Rama exclusiva lstm-produccion-final. No modificar fuente ni resultados v2/v3.
Objetivo: elegir LSTM por error recursivo a 12 semanas, no por apariencia de curva.

Configuraciones fijadas antes de ejecutar: ventanas 8 y 12, 16 unidades,
semilla 42, dropout 0.15, Adam 0.001. Escalas log1p/Huber y original/MSE.
Épocas candidatas 10, 30, 60, 100, elegidas usando todos los orígenes completos
de VALIDACIÓN. Cada configuración se entrena desde cero para cada época candidata
para asegurar que el refit reproduzca el procedimiento.
Para producción se agrega una LSTM hurdle (Bernoulli y cantidad positiva), con
salida esperada p*q y pérdidas BCE + MSE condicional en escala original.
La escala se ajusta exclusivamente a TRAIN. Los cinco ceros iniciales de demanda
se incorporan únicamente por declaración del usuario.

Se incluyen como controles las LSTM v3 y las referencias sencillas. Criterio
único RMSE de validación, MAE como desempate. No explorar semillas hasta obtener
un resultado favorable. No modificar esta parrilla tras consultar TEST.

La selección se guarda antes de evaluar TEST. El TEST conocido permite una
comparación retrospectiva, no una nueva afirmación de generalización ciega.
El pronóstico futuro principal de demanda conserva pesos seleccionados; el refit
con todo el historial es opcional después de prueba. No usar pronóstico de
producción histórica como orden de fabricación.

Producción mantiene el reajuste final con todo el historial después de evaluar
prueba, con arquitectura y épocas congeladas, tal como en v3.
