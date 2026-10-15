"""Preparación auditable de dos demandas y conciliación de sus pronósticos.

Las compras de tela y la producción estimada NO son variables objetivo.
El archivo de trabajo contiene 4 períodos por mes (48/año), no semanas ISO.
"""
import numpy as np
import pandas as pd

from forecast_engine import metrics, period_positions

PERIOD = "Periodo (4/mes)"
TARGETS = {"Maquila": "DEMANDA MAQUILA LSTM", "Corte propio": "DEMANDA CORTE LSTM"}
SOURCE = {"Maquila": "Origen maquila", "Corte propio": "Origen corte"}
OBSERVED = {"Maquila": "Maquila: registrado base", "Corte propio": "Corte: documentado neto"}
REQUIRED = [PERIOD, "Bloque temporal", "DEMANDA TOTAL LSTM", *TARGETS.values(), *SOURCE.values()]

MAQUILA_ORIGINS = {
    "ORIGINAL": "REAL",
    "ORIGINAL_REVISAR": "SIN_DECLARAR",
    "SIMULADA": "IMPUTADO",
}
CORTE_ORIGINS = {
    "SOLO ESCENARIO": "IMPUTADO",
    "PARCIAL REGISTRADO": "MIXTO",
    "PARCIAL+ESCENARIO": "MIXTO",
}


def prepare_dual(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Una fila por período compartido, sin inventar ceros ni cambiar la demanda."""
    missing = [col for col in REQUIRED if col not in frame.columns]
    if missing:
        raise ValueError("Faltan columnas de la base dual: " + ", ".join(missing))
    raw = frame.loc[frame["Bloque temporal"].astype(str).str.upper().str.strip().isin(
        {"TRAIN", "VALIDACIÓN", "VALIDACION", "PRUEBA", "TEST"})].copy()
    if len(raw) < 40:
        raise ValueError("La base común tiene menos de 40 períodos de entrenamiento/evaluación.")
    labels = raw[PERIOD].astype(str).str.strip().reset_index(drop=True)
    positions = period_positions(labels)
    if positions is None or not np.all(np.diff(positions) == 1):
        raise ValueError("Los períodos deben ser consecutivos en el calendario de 48 períodos/año.")
    if labels.duplicated().any():
        raise ValueError("Existen períodos duplicados.")
    total = pd.to_numeric(raw["DEMANDA TOTAL LSTM"], errors="coerce").to_numpy(float)
    results = {}
    values = []
    for name in TARGETS:
        demand = pd.to_numeric(raw[TARGETS[name]], errors="coerce").to_numpy(float)
        if not np.isfinite(demand).all() or (demand < 0).any():
            raise ValueError(f"{name}: demanda vacía, infinita o negativa.")
        sources = raw[SOURCE[name]].fillna("").astype(str).str.upper().str.strip().reset_index(drop=True)
        mapping = MAQUILA_ORIGINS if name == "Maquila" else CORTE_ORIGINS
        unexpected = set(sources.unique()) - set(mapping)
        if unexpected:
            raise ValueError(f"{name}: procedencia desconocida: {sorted(unexpected)}")
        origin = sources.map(mapping)
        if OBSERVED[name] in raw.columns:
            observed = pd.to_numeric(raw[OBSERVED[name]], errors="coerce").reset_index(drop=True)
        else:
            observed = pd.Series(np.nan, index=range(len(raw)))
        # El campo original solo documenta lo registrado; no sustituye la demanda reconstruida.
        observed = observed.mask(origin.eq("IMPUTADO"))
        results[name] = pd.DataFrame({
            "periodo": labels, "demanda": demand,
            "origen": origin.to_numpy(),
            "demanda_original": observed.to_numpy(),
            "tipo_estimacion": sources.to_numpy(),
        })
        values.append(demand)
    if not np.isfinite(total).all() or not np.allclose(values[0] + values[1], total, rtol=0, atol=1e-5):
        raise ValueError("La demanda total no coincide con maquila + corte propio.")
    if not results["Maquila"].periodo.equals(results["Corte propio"].periodo):
        raise ValueError("Las modalidades no están alineadas en el tiempo.")
    return results


def combined_forecast(maquila: pd.DataFrame, corte: pd.DataFrame) -> pd.DataFrame:
    """Suma pronósticos independientes sin redondearlos antes del cálculo."""
    columns = ["Periodo", "Horizonte", "LSTM"]
    a = maquila[columns].rename(columns={"LSTM": "Maquila"}).copy()
    b = corte[columns].rename(columns={"LSTM": "Corte propio"}).copy()
    out = a.merge(b, on=["Periodo", "Horizonte"], how="inner", validate="one_to_one", sort=False)
    if len(out) != len(a) or len(out) != len(b):
        raise ValueError("Los horizontes de ambas modalidades no coinciden.")
    for col in ("Maquila", "Corte propio"):
        if not np.isfinite(out[col].to_numpy(float)).all() or (out[col] < 0).any():
            raise ValueError(f"Pronósticos inválidos en {col}.")
    out["Total"] = out["Maquila"] + out["Corte propio"]
    return out


def combined_test(maquila: pd.DataFrame, corte: pd.DataFrame):
    """Evaluación del total en los mismos períodos; nunca suma R² individuales."""
    columns = ["Periodo", "Origen", "Demanda_real", "Prediccion_LSTM"]
    a = maquila[columns].rename(columns={
        "Origen": "Origen_maquila", "Demanda_real": "Real_maquila",
        "Prediccion_LSTM": "Pred_maquila"})
    b = corte[columns].rename(columns={
        "Origen": "Origen_corte", "Demanda_real": "Real_corte",
        "Prediccion_LSTM": "Pred_corte"})
    out = a.merge(b, on="Periodo", how="inner", validate="one_to_one", sort=False)
    if len(out) != len(a) or len(out) != len(b):
        raise ValueError("Los períodos TEST no coinciden entre modalidades.")
    out["Real_total"] = out["Real_maquila"] + out["Real_corte"]
    out["Pred_total"] = out["Pred_maquila"] + out["Pred_corte"]
    out["Error_total"] = out["Real_total"] - out["Pred_total"]
    descriptive = metrics(out["Real_total"].to_numpy(float), out["Pred_total"].to_numpy(float))
    # No hay evaluación puramente observada si una de las demandas fue reconstruida.
    only_real = out["Origen_maquila"].eq("REAL") & out["Origen_corte"].eq("REAL")
    documented = metrics(out.loc[only_real, "Real_total"].to_numpy(float),
                         out.loc[only_real, "Pred_total"].to_numpy(float)) if only_real.sum() >= 2 else None
    return out, descriptive, documented, int(only_real.sum())
