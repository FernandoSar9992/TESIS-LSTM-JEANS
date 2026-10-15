"""Preparación genérica: ningún nombre de archivo, producto o calendario especial."""
from dataclasses import dataclass
from io import BytesIO
import unicodedata

import numpy as np
import pandas as pd

FREQUENCIES = {"Detectar frecuencia": None, "Diaria": "D", "Semanal (lunes a domingo)": "W-SUN",
               "Mensual": "MS", "Trimestral": "QS", "Anual": "YS"}


def detect_column(columns, *names):
    def key(value):
        return "".join(c for c in unicodedata.normalize("NFKD", str(value)).lower() if c.isalnum())
    mapping = {key(c): c for c in columns}
    return next((mapping[key(n)] for n in names if key(n) in mapping), None)


def merge_origins(values):
    known = set(values.dropna())
    if not known:
        return "IMPUTADO"
    if known == {"REAL"}:
        return "REAL"
    if known == {"IMPUTADO"}:
        return "IMPUTADO"
    return "MIXTO" if "IMPUTADO" in known or "MIXTO" in known else "SIN_DECLARAR"


def format_periods(index):
    """Conservar la hora para frecuencias intradía; no colapsar períodos distintos."""
    fmt = "%Y-%m-%d %H:%M:%S" if (index != index.normalize()).any() else "%Y-%m-%d"
    return index.strftime(fmt).tolist()


def read_upload(content: bytes, name: str, sheet=0, decimal="."):
    if name.lower().endswith(".csv"):
        try:
            return pd.read_csv(BytesIO(content), sep=None, engine="python", decimal=decimal)
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(content), sep=None, engine="python", decimal=decimal, encoding="cp1252")
    return pd.read_excel(BytesIO(content), sheet_name=sheet, decimal=decimal)


@dataclass
class PreparedSeries:
    data: pd.DataFrame
    frequency: str | None
    last_date: pd.Timestamp | None
    notes: list[str]

    def future_labels(self, horizon, periodicity=None):
        if self.frequency and self.last_date is not None:
            return format_periods(pd.date_range(self.last_date, periods=horizon+1, freq=self.frequency)[1:])
        if periodicity == 48:
            from forecast_engine import future_labels
            return future_labels(self.data, horizon)
        return [f"Período +{h}" for h in range(1, horizon+1)]


def prepare_series(frame, period_col, demand_col, *, date_mode=False, dayfirst=True,
                   frequency=None, fill_missing_zero=False, declared_real=False):
    """Fechas se ordenan; sumar transacciones solo con frecuencia explícita.

    Con etiquetas se respeta el orden original, que el usuario debe verificar.
    Un cero ausente solo se agrega cuando el usuario declara esa interpretación.
    """
    if period_col == demand_col:
        raise ValueError("Seleccione columnas diferentes para período y demanda.")
    if frame.empty:
        raise ValueError("La selección no contiene registros.")
    values = pd.to_numeric(frame[demand_col], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("La demanda debe contener números finitos y no negativos, sin vacíos. Revise la columna y el separador decimal.")
    if frame[period_col].isna().any():
        raise ValueError("Hay períodos vacíos. Corrija el archivo antes de entrenar.")
    notes = []
    origin_col = detect_column(frame.columns, "Origen")
    original_col = detect_column(frame.columns, "Demanda")
    type_col = detect_column(frame.columns, "Tipo_estimacion")
    origins = frame[origin_col].fillna("SIN_DECLARAR").astype(str).str.strip().str.upper() if origin_col is not None else pd.Series("REAL" if declared_real else "SIN_DECLARAR", index=frame.index)
    if not origins.isin(["REAL", "IMPUTADO", "MIXTO", "SIN_DECLARAR"]).all():
        raise ValueError("Origen admite REAL, IMPUTADO, MIXTO o SIN_DECLARAR. Corrija los valores desconocidos.")
    original = pd.to_numeric(frame[original_col], errors="coerce") if original_col is not None else values.copy()
    kinds = frame[type_col].fillna("").astype(str) if type_col is not None else pd.Series("", index=frame.index)
    if origin_col is None and declared_real:
        notes.append("El usuario declaró que todos los registros cargados son observaciones reales; no es una verificación documental independiente.")
    elif origin_col is None:
        notes.append("No existe Origen: no se puede distinguir qué valores son reales o imputados. Las métricas se etiquetan sin trazabilidad.")
    else:
        notes.append("Se conserva Origen. Verifique que las imputaciones no utilicen información futura; la etiqueta por sí sola no garantiza causalidad.")
    trace = pd.DataFrame({"demanda_original": original.to_numpy(float), "origen": origins.to_numpy(), "tipo_estimacion": kinds.to_numpy()})
    if not date_mode:
        labels = frame[period_col].astype(str).str.strip()
        if labels.eq("").any() or labels.duplicated().any():
            raise ValueError("Las etiquetas deben ser únicas y no vacías. Filtre un producto o agrupe las transacciones por fecha.")
        from forecast_engine import period_positions
        positions = period_positions(labels)
        if positions is not None and not np.all(np.diff(positions) == 1):
            raise ValueError("El calendario de cuatro períodos por mes contiene saltos o está desordenado. Corrija el orden y los períodos ausentes antes de entrenar.")
        notes.append("Etiquetas: se conserva el orden del archivo. Cada fila debe representar un período consecutivo de igual duración; el futuro se indica como +1, +2…")
        result = pd.DataFrame({"periodo": labels.to_numpy(), "demanda": values.to_numpy(float)})
        return PreparedSeries(pd.concat([result, trace], axis=1), None, None, notes)
    if pd.api.types.is_numeric_dtype(frame[period_col]):
        raise ValueError("La columna de fecha es numérica. Use fechas reconocibles o el modo de etiquetas.")
    dates = pd.to_datetime(frame[period_col], errors="coerce", dayfirst=dayfirst, format="mixed")
    if dates.isna().any():
        raise ValueError("No se pudieron interpretar todas las fechas. Revise el formato día/mes o use etiquetas ordenadas.")
    if dates.dt.tz is not None:
        raise ValueError("Use fechas sin zona horaria, convertidas previamente a la zona de su negocio.")
    series = pd.Series(values.to_numpy(float), index=pd.DatetimeIndex(dates)).sort_index()
    trace.index = pd.DatetimeIndex(dates)
    trace = trace.sort_index()
    if frequency:
        if frequency not in set(FREQUENCIES.values()) - {None}:
            raise ValueError("Frecuencia no admitida.")
        series = series.resample(frequency).sum(min_count=1)
        trace = trace.resample(frequency).agg({"demanda_original": lambda x: x.sum(min_count=1),
                                             "origen": merge_origins,
                                             "tipo_estimacion": lambda x: "; ".join(sorted(set(x.dropna()) - {""}))})
        notes.append("Las demandas se sumaron por la frecuencia elegida. Revise que el primer y último período estén completos.")
    else:
        if series.index.has_duplicates:
            raise ValueError("Hay fechas repetidas. Elija una frecuencia para sumar transacciones o filtre una sola serie.")
        frequency = pd.infer_freq(series.index) if len(series) >= 3 else None
        if frequency is None:
            raise ValueError("Las fechas no tienen una frecuencia regular reconocible. Elija la frecuencia y revise los períodos faltantes.")
    missing = int(series.isna().sum())
    if missing:
        if not fill_missing_zero:
            raise ValueError(f"Faltan {missing} períodos. Corrija la fuente o declare que los períodos sin registros equivalen a demanda cero.")
        series = series.fillna(0)
        trace.loc[trace.demanda_original.isna() & trace.tipo_estimacion.eq(""), "tipo_estimacion"] = "Cero declarado para período sin registros"
        notes.append(f"Se asignó demanda cero a {missing} períodos sin registros por su indicación.")
    if not np.isfinite(series).all():
        raise ValueError("La suma excede el rango numérico admitido. Revise unidades y valores.")
    result = pd.DataFrame({"periodo": format_periods(series.index), "demanda": series.to_numpy(float)})
    result = pd.concat([result, trace.reset_index(drop=True)], axis=1)
    return PreparedSeries(result, frequency, series.index[-1], notes)
