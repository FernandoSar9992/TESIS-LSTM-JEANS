# Pronóstico de demanda para tesis

Aplicación limpia de presentación: dos redes LSTM independientes (maquila y corte propio) y pronóstico agregado.

Ejecutar en la carpeta del proyecto:

```powershell
python -m streamlit run dual_app.py
```

Cargar un Excel de una hoja con 3 encabezados en la fila 4:
- Periodo (4/mes)
- DEMANDA MAQUILA LSTM
- DEMANDA CORTE LSTM

Los valores de corte propio previos al inicio de esta modalidad deben quedar vacíos.

**Los archivos de demanda sintética son escenarios de planificación y no pedidos reales comprobados.**
El modelo no garantiza precisión estadística sobre pedidos no documentados.
