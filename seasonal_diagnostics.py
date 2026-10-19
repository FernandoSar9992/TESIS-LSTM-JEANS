"""Diagnóstico explicable de calidad/estacionalidad para dos series del taller.

Solo usa historia previa y registros de procedencia. No modifica ni imputan
valores; tampoco entrena redes. Por defecto el calendario son 48 períodos/año.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def profile_temporal(data: pd.DataFrame, *, train_ratio=.70, val_ratio=.15, period=48) -> dict:
    """Diagnostica TRAIN y una referencia estacional en VALIDATION, nunca TEST."""
    y = pd.to_numeric(data["demanda"], errors="coerce").to_numpy(float)
    n = len(y)
    if n < period + 10 or not np.isfinite(y).all():
        raise ValueError("Datos insuficientes o demanda no finita.")
    tr, va = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    if tr <= period or va <= tr or va >= n:
        raise ValueError("Partición incompatible con el ciclo anual.")
    origin = data["origen"].astype(str).str.upper().str.strip()
    available = np.flatnonzero(np.arange(tr, va) >= period)
    targets = np.arange(tr, va)[available]
    real = y[targets]
    season = y[targets-period]
    complete = (origin.iloc[targets].to_numpy() == "REAL")
    recent = y[:tr]
    if tr - period >= 10 and np.std(recent[period:]) > 1e-8 and np.std(recent[:-period]) > 1e-8:
        corr = float(np.corrcoef(recent[period:], recent[:-period])[0, 1])
    else:
        corr = None

    def measures(mask):
        yy, pp = real[mask], season[mask]
        if len(yy) < 2:
            return {"N": int(len(yy)), "R2": None, "MAE": None, "RMSE": None}
        error = yy - pp
        denom = float(np.sum((yy - yy.mean())**2))
        return {
            "N": int(len(yy)),
            "R2": None if denom < 1e-12 else float(1 - np.sum(error**2) / denom),
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error**2))),
        }

    return {
        "N_total": n, "TRAIN": tr, "VALIDACION": va-tr,
        "REAL_TOTAL": int(origin.eq("REAL").sum()),
        "REAL_VALIDACION": int(origin.iloc[tr:va].eq("REAL").sum()),
        "IMPUTADO_TOTAL": int(origin.eq("IMPUTADO").sum()),
        "MIXTO_TOTAL": int(origin.eq("MIXTO").sum()),
        "CORR_ANUAL_TRAIN": corr,
        "ESTACIONAL_TODOS_DESCRIPTIVO": measures(np.ones(len(targets), bool)),
        "ESTACIONAL_SOLO_REAL": measures(complete),
        "ADVERTENCIA": (
            "La precisión sobre valores sintéticos o reconstruidos no demuestra "
            "precisión frente a pedidos observados. No se consultó TEST."
        ),
    }
