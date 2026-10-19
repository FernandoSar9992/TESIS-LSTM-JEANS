"""Auditoría de errores de VALIDATION sin consultar objetivos TEST.

La tabla procede de predicciones ya generadas por run_thesis; no selecciona
configuraciones ni reentrena redes. Los diagnósticos sobre la demanda actual
son retrospectivos, NO características utilizadas para pronosticar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def validation_error_audit(result: dict) -> pd.DataFrame:
    """Una fila por período VALIDATION, por la misma LSTM y referencia."""
    data = result["_artifacts"]["data"]
    tr, va = int(result["train_end"]), int(result["val_end"])
    periods = np.arange(tr, va)
    forecast = result["validation_detail"].copy().reset_index(drop=True)
    if len(forecast) != len(periods):
        raise ValueError("La cantidad de predicciones no coincide con VALIDATION.")
    if forecast["Periodo"].astype(str).tolist() != data.periodo.iloc[periods].astype(str).tolist():
        raise ValueError("Las predicciones no coinciden con los períodos de VALIDATION.")
    needed = {"Real", "LSTM", "Naive", "Naive estacional", "Ridge"}
    if not needed.issubset(forecast.columns):
        raise ValueError("Faltan predicciones de comparación para auditar VALIDATION.")

    actual = data.demanda.iloc[periods].to_numpy(float)
    predicted = forecast.LSTM.to_numpy(float)
    if not np.isfinite(predicted).all() or not np.allclose(actual, forecast.Real.to_numpy(float), rtol=0, atol=1e-8):
        raise ValueError("Valores reales o predicciones desalineados.")

    origins = data.origen.iloc[periods].astype(str).to_numpy()
    last_year = int(result["config"]["periodicity"])
    y = data.demanda.to_numpy(float)
    train_q90 = float(np.percentile(y[:tr], 90))
    train_median = float(np.median(y[:tr]))
    observed_last_year = np.array([y[t-last_year] if t >= last_year else np.nan for t in periods])
    previous = y[periods-1]
    deviation = actual-predicted
    baseline = result.get("reference_model")
    if baseline not in forecast.columns:
        baseline = "Naive"

    audit = pd.DataFrame({
        "Periodo": data.periodo.iloc[periods].astype(str).to_numpy(),
        "Indice_temporal": periods+1,
        "Procedencia": origins,
        "Tipo_estimacion": data.tipo_estimacion.iloc[periods].fillna("").astype(str).to_numpy(),
        "Demanda_utilizada": actual,
        "Demanda_documentada_original": pd.to_numeric(data.demanda_original.iloc[periods], errors="coerce").to_numpy(float),
        "Pronostico_LSTM": predicted,
        "Error_real_menos_LSTM": deviation,
        "Error_absoluto_LSTM": np.abs(deviation),
        "Error_porcentual_absoluto": np.divide(
            np.abs(deviation)*100, actual,
            out=np.full(len(actual), np.nan), where=actual > 0),
        "Referencia_elegida": baseline,
        "Pronostico_referencia": forecast[baseline].to_numpy(float),
        "Error_absoluto_referencia": np.abs(actual-forecast[baseline].to_numpy(float)),
        "LSTM_mejor_que_referencia": np.abs(deviation) < np.abs(actual-forecast[baseline].to_numpy(float)),
        "Demanda_previa": previous,
        "Demanda_mismo_periodo_anio_anterior": observed_last_year,
        "Cambio_vs_periodo_previo": actual-previous,
        "Cambio_vs_anio_anterior": actual-observed_last_year,
        "Umbral_p90_TRAIN": train_q90,
        "Demanda_sobre_p90_TRAIN": actual > train_q90,
        "Demanda_bajo_mediana_TRAIN": actual < train_median,
    })
    # El porcentaje no está definido con demanda cero. No sustituir con cero.
    # Esta tabla NO incluye TEST, aun si existen valores TEST en data.
    return audit


def validation_error_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Resumen por origen: Nunca equiparar MI/XTO o IMPUTADO con REAL."""
    records = []
    for name, sub in [("TODOS_DESCRIPTIVO", audit),
                      ("SOLO_REAL", audit.loc[audit.Procedencia.eq("REAL")]),
                      ("MIXTO", audit.loc[audit.Procedencia.eq("MIXTO")]),
                      ("IMPUTADO", audit.loc[audit.Procedencia.eq("IMPUTADO")]),
                      ("SIN_DECLARAR", audit.loc[audit.Procedencia.eq("SIN_DECLARAR")]),
                      ("DEMANDA_SOBRE_P90_TRAIN", audit.loc[audit.Demanda_sobre_p90_TRAIN])]:
        n = len(sub)
        errors = sub.Error_real_menos_LSTM.to_numpy(float)
        y = sub.Demanda_utilizada.to_numpy(float)
        preds = sub.Pronostico_LSTM.to_numpy(float)
        denom = float(np.sum((y-y.mean())**2)) if n else 0.0
        total = float(np.sum(y))
        records.append({
            "Grupo": name, "N": n,
            "MAE": float(np.mean(np.abs(errors))) if n else np.nan,
            "RMSE": float(np.sqrt(np.mean(errors**2))) if n else np.nan,
            "Sesgo_real_menos_predicho": float(np.mean(errors)) if n else np.nan,
            "WAPE_pct": float(np.sum(np.abs(errors))/total*100) if n and total > 0 else np.nan,
            "R2": float(1-np.sum(errors**2)/denom) if n >= 2 and denom > 1e-12 else np.nan,
            "LSTM_gana_en_N_periodos": int(sub.LSTM_mejor_que_referencia.sum()),
        })
    return pd.DataFrame(records)


def validation_error_priority(audit: pd.DataFrame, n=10) -> pd.DataFrame:
    """Lista de revisión, no prueba automática de fallas en los registros."""
    ranked = audit.sort_values("Error_absoluto_LSTM", ascending=False).head(n).copy()
    flags = []
    for row in ranked.itertuples(index=False):
        notes = []
        if row.Procedencia != "REAL":
            notes.append("Objetivo no totalmente verificado")
        if row.Demanda_sobre_p90_TRAIN:
            notes.append("Demanda sobre percentil 90 de TRAIN")
        if row.Error_real_menos_LSTM > 0:
            notes.append("LSTM subestimó")
        elif row.Error_real_menos_LSTM < 0:
            notes.append("LSTM sobreestimó")
        if pd.notna(row.Cambio_vs_anio_anterior) and abs(row.Cambio_vs_anio_anterior) > 0.5*max(float(row.Demanda_utilizada), 1):
            notes.append("Gran variación frente al año anterior")
        flags.append("; ".join(notes))
    ranked["Puntos_para_revisar"] = flags
    return ranked
