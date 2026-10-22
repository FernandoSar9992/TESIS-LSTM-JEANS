"""Lectura y validación del Excel de demanda sin alterar sus registros.

Semana = lunes a viernes; la etiqueta Se1..Se5 se asigna según
la posición del lunes dentro de su mes. No es un calendario de 4/mes.
"""
from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
import re

import numpy as np
import pandas as pd

from forecast_engine import future_labels, period_positions

MONTHS = ("Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
          "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre")
TARGETS = {
    "Maquila": "DEMANDA MAQUILA LSTM",
    "Corte propio": "DEMANDA CORTE LSTM",
}


def _header(value):
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def _excel_dates(values, column):
    def parse(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (int, float, np.integer, np.floating)):
            return pd.to_datetime(value, unit="D", origin="1899-12-30", errors="coerce")
        return pd.to_datetime(value, errors="coerce", dayfirst=True)

    parsed = pd.to_datetime(values.map(parse), errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError(f"{column}: hay fechas vacías o inválidas.")
    return parsed.reset_index(drop=True)


def week_label(monday):
    monday = pd.Timestamp(monday)
    if monday.weekday() != 0:
        raise ValueError("La fecha de inicio de un período debe ser lunes.")
    week = (monday.day - 1) // 7 + 1
    return f"{MONTHS[monday.month - 1]} {monday.year}-Se{week}"


def upcoming_weeks(last_monday, horizon):
    start = pd.Timestamp(last_monday)
    return [week_label(start + pd.Timedelta(weeks=i)) for i in range(1, int(horizon) + 1)]


def load_demand_excel(contents: bytes, cutoff: date | datetime | pd.Timestamp):
    """Devuelve (series, etiquetas futuras generables, metadatos).

    Si hay fechas, solo incluye períodos FINALIZADOS ANTES del día de corte.
    Excel de cuatro segmentos por mes sin fechas se admite como formato legado.
    """
    preview = pd.read_excel(BytesIO(contents), sheet_name=0, header=None, nrows=10)
    expected = set(TARGETS.values())
    candidates = [
        index for index, row in preview.iterrows()
        if expected.issubset({_header(value) for value in row if pd.notna(value)})
    ]
    if len(candidates) != 1:
        raise ValueError("No se encontraron encabezados únicos de DEMANDA MAQUILA LSTM y DEMANDA CORTE LSTM.")
    frame = pd.read_excel(BytesIO(contents), sheet_name=0, header=candidates[0])
    frame.columns = [_header(value) for value in frame.columns]
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("Existen nombres de columnas duplicados.")
    period_col = next((x for x in ("PERIODO (SEMANA)", "PERIODO (4/MES)") if x in frame), None)
    if period_col is None:
        raise ValueError("Falta la columna Periodo (semana) o Periodo (4/mes).")
    frame = frame.dropna(subset=[period_col]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("El archivo no contiene períodos.")
    periods = frame[period_col].astype(str).str.strip()
    if periods.eq("").any() or periods.duplicated().any():
        raise ValueError("Hay períodos vacíos o duplicados.")

    weekly = "FECHA_INICIO" in frame and "FECHA_FIN" in frame
    if ("FECHA_INICIO" in frame) != ("FECHA_FIN" in frame):
        raise ValueError("Para fechas semanales hacen falta Fecha_inicio y Fecha_fin.")
    excluded = 0
    if weekly:
        starts = _excel_dates(frame["FECHA_INICIO"], "Fecha_inicio")
        ends = _excel_dates(frame["FECHA_FIN"], "Fecha_fin")
        if not starts.dt.weekday.eq(0).all() or not ends.dt.weekday.eq(4).all():
            raise ValueError("Cada período debe comenzar lunes y finalizar viernes.")
        if not ((ends - starts).dt.days == 4).all():
            raise ValueError("Las fechas de cada semana no coinciden: lunes a viernes.")
        if len(starts) > 1 and not ((starts.diff().dt.days.iloc[1:]) == 7).all():
            raise ValueError("Hay semanas saltadas o fechas de inicio no consecutivas.")
        expected_labels = [week_label(monday) for monday in starts]
        if not all(actual.casefold() == expected.casefold()
                   for actual, expected in zip(periods, expected_labels)):
            raise ValueError("Los nombres Se1–Se5 no coinciden con las fechas de inicio.")
        day = pd.Timestamp(cutoff).normalize()
        completed = ends < day
        if not completed.any():
            raise ValueError("No hay semanas completas anteriores a la fecha de corte.")
        if completed.tolist() != sorted(completed.tolist(), reverse=True):
            raise ValueError("Las fechas completas deben formar un prefijo cronológico.")
        excluded = int((~completed).sum())
        frame = frame.loc[completed].reset_index(drop=True)
        starts = starts.loc[completed].reset_index(drop=True)
        periods = periods.loc[completed].reset_index(drop=True)
        periodicity = 52  # Aproximación anual semanal, no 48 segmentos fijos
        future = lambda n: upcoming_weeks(starts.iloc[-1], n)
        last_complete_date = ends.loc[completed].iloc[-1].date().isoformat()
    else:
        positions = period_positions(periods)
        if positions is None or not np.all(np.diff(positions) == 1):
            raise ValueError("Sin fechas solo se admite el formato legado consecutivo de cuatro períodos por mes.")
        periodicity = 48
        future = lambda n: future_labels(pd.DataFrame({"periodo": periods}), n)
        starts = None
        last_complete_date = "Sin fechas: verificación temporal no disponible"

    for name in ("DEMANDA TOTAL LSTM", "PRODUCCION CORTE REGISTRADA"):
        if name in frame:
            raw = pd.to_numeric(frame[name], errors="coerce")
            if raw.isna().any() or (raw < 0).any():
                raise ValueError(f"{name}: contiene datos vacíos, no numéricos o negativos.")

    series = {}
    for title, column in TARGETS.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if title == "Corte propio":
            valid = np.flatnonzero(values.notna().to_numpy())
            if not len(valid):
                raise ValueError("No hay demanda de corte propio en las semanas completas.")
            first = int(valid[0])
        else:
            first = 0
        part = values.iloc[first:].reset_index(drop=True)
        if not np.isfinite(part).all() or (part < 0).any():
            raise ValueError(f"{title}: hay datos faltantes, no numéricos o negativos.")
        item = pd.DataFrame({
            "periodo": periods.iloc[first:].to_numpy(),
            "demanda": part.to_numpy(float),
            "origen": "SIN_DECLARAR",
        })
        if weekly:
            item["fecha_inicio"] = starts.iloc[first:].to_numpy()
        series[title] = item

    if "DEMANDA TOTAL LSTM" in frame:
        total = pd.to_numeric(frame["DEMANDA TOTAL LSTM"], errors="coerce").to_numpy(float)
        maquila = pd.to_numeric(frame[TARGETS["Maquila"]], errors="coerce").to_numpy(float)
        corte = pd.to_numeric(frame[TARGETS["Corte propio"]], errors="coerce").fillna(0).to_numpy(float)
        if not np.allclose(total, maquila + corte, atol=0.01):
            raise ValueError("DEMANDA TOTAL LSTM no coincide con maquila + demanda de corte propio.")
    if "PRODUCCION CORTE REGISTRADA" in frame and "CONFECCIÓN MAQUILA Y CORTE" in frame:
        produced = pd.to_numeric(frame["PRODUCCION CORTE REGISTRADA"], errors="coerce").to_numpy(float)
        activity = pd.to_numeric(frame["CONFECCIÓN MAQUILA Y CORTE"], errors="coerce").to_numpy(float)
        maquila = pd.to_numeric(frame[TARGETS["Maquila"]], errors="coerce").to_numpy(float)
        if not np.allclose(activity, maquila + produced, atol=0.01):
            raise ValueError("CONFECCIÓN MAQUILA Y CORTE debe sumar maquila + producción, no ventas.")

    info = {
        "calendar": "Semanas lunes–viernes (Se1–Se5)" if weekly else "Segmentos de 4/mes (legado)",
        "periodicity": periodicity,
        "excluded": excluded,
        "n_historical": len(frame),
        "last_period": str(periods.iloc[-1]),
        "last_complete_date": last_complete_date,
        "cutoff": pd.Timestamp(cutoff).date().isoformat(),
        "production_present": "PRODUCCION CORTE REGISTRADA" in frame,
    }
    return series, future, info
