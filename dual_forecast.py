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
    """Prepara el historial completo de cada modalidad.

    Maquila abarca todo el archivo histórico; corte propio solo comienza
    donde aparece su primera actividad. Antes de comenzar corte propio,
    la ausencia de demanda NO se trata como una demanda real igual a cero.
    """
    missing = [col for col in REQUIRED if col not in frame.columns]
    if missing:
        raise ValueError("Faltan columnas de la base dual: " + ", ".join(missing))
    blocks = frame["Bloque temporal"].fillna("").astype(str).str.upper().str.strip()
    historical = blocks.isin({
        "ANTERIOR AL PERIODO COMUN", "TRAIN", "VALIDACIÓN", "VALIDACION", "PRUEBA", "TEST"
    })
    all_history = frame.loc[historical].copy()
    cut_origins = all_history[SOURCE["Corte propio"]].fillna("").astype(str).str.upper().str.strip()
    # Incluye Mayo 2024-Se4: tiene demanda propia aunque está fuera del
    # período conjunto usado por la versión anterior.
    own_history = all_history.loc[cut_origins.isin(CORTE_ORIGINS)].copy()
    if own_history.empty:
        raise ValueError("No hay períodos identificados de actividad de corte propio.")
    subsets = {"Maquila": all_history, "Corte propio": own_history}
    results = {}
    for name, selected in subsets.items():
        if len(selected) < 40:
            raise ValueError(f"{name}: menos de 40 períodos históricos disponibles.")
        labels = selected[PERIOD].astype(str).str.strip().reset_index(drop=True)
        positions = period_positions(labels)
        if positions is None or not np.all(np.diff(positions) == 1) or labels.duplicated().any():
            raise ValueError(f"{name}: períodos duplicados, no reconocidos o no consecutivos.")
        demand = pd.to_numeric(selected[TARGETS[name]], errors="coerce").to_numpy(float)
        if not np.isfinite(demand).all() or (demand < 0).any():
            raise ValueError(f"{name}: demanda vacía, infinita o negativa.")
        sources = selected[SOURCE[name]].fillna("").astype(str).str.upper().str.strip().reset_index(drop=True)
        mapping = MAQUILA_ORIGINS if name == "Maquila" else CORTE_ORIGINS
        unexpected = set(sources.unique()) - set(mapping)
        if unexpected:
            raise ValueError(f"{name}: procedencia desconocida: {sorted(unexpected)}")
        origins = sources.map(mapping)
        if OBSERVED[name] in selected.columns:
            original = pd.to_numeric(selected[OBSERVED[name]], errors="coerce").reset_index(drop=True)
        else:
            original = pd.Series(np.nan, index=range(len(selected)))
        original = original.mask(origins.eq("IMPUTADO"))
        results[name] = pd.DataFrame({
            "periodo": labels, "demanda": demand,
            "origen": origins.to_numpy(),
            "demanda_original": original.to_numpy(),
            "tipo_estimacion": sources.to_numpy(),
        })

    # Comprobar el total histórico SOLO en fechas donde opera corte propio.
    cuts = own_history
    total = pd.to_numeric(cuts["DEMANDA TOTAL LSTM"], errors="coerce").to_numpy(float)
    mq = pd.to_numeric(cuts[TARGETS["Maquila"]], errors="coerce").to_numpy(float)
    ct = results["Corte propio"].demanda.to_numpy(float)
    if not np.isfinite(total).all() or not np.allclose(mq + ct, total, rtol=0, atol=1e-5):
        raise ValueError("La demanda total no coincide con maquila + corte propio.")
    if results["Maquila"].periodo.iloc[-1] != results["Corte propio"].periodo.iloc[-1]:
        raise ValueError("Las series no terminan en el mismo período: no se puede sumar el futuro.")
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
    # Los segmentos TEST comienzan en momentos diferentes porque los
    # historiales tienen longitudes distintas. El total se evalúa únicamente
    # en la intersección cronológica de ambos conjuntos TEST.
    if len(out) < 2:
        raise ValueError("Hay menos de dos períodos TEST compartidos entre modalidades.")
    out["Real_total"] = out["Real_maquila"] + out["Real_corte"]
    out["Pred_total"] = out["Pred_maquila"] + out["Pred_corte"]
    out["Error_total"] = out["Real_total"] - out["Pred_total"]
    descriptive = metrics(out["Real_total"].to_numpy(float), out["Pred_total"].to_numpy(float))
    # No hay evaluación puramente observada si una de las demandas fue reconstruida.
    only_real = out["Origen_maquila"].eq("REAL") & out["Origen_corte"].eq("REAL")
    documented = metrics(out.loc[only_real, "Real_total"].to_numpy(float),
                         out.loc[only_real, "Pred_total"].to_numpy(float)) if only_real.sum() >= 2 else None
    return out, descriptive, documented, int(only_real.sum())
