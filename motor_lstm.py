"""Motor reproducible para pronóstico semanal de demanda y producción de jeans.

La aplicación entrena tres series independientes:
  - Demanda de maquila
  - Demanda de corte propio
  - Producción registrada de corte propio
Los totales se calculan por suma; no se suman métricas ni se inventan registros.

Separación temporal: entrenamiento -> validación -> prueba (sin fuga de información).
Las escalas se ajustan exclusivamente con entrenamiento en la evaluación.
Los hiperparámetros se seleccionan únicamente con validación; prueba es final.
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

APP_TIMEZONE = ZoneInfo("America/Guayaquil")
TARGETS = {
    "Demanda de maquila": "demanda_mq",
    "Demanda de corte propio": "demanda_cp",
    "Producción de corte propio": "produccion_cp",
}
REQUIRED_COLUMNS = {
    "fecha de inicio": "fecha_inicio",
    "fecha de fin": "fecha_fin",
    "demanda maquila lstm": "demanda_mq",
    "demanda corte lstm": "demanda_cp",
    "produccion corte registrada": "produccion_cp",
}
OPTIONAL_COLUMNS = {
    "periodo (semana)": "periodo",
    "inventario inicial cp": "inventario_inicial_cp",
    "inventario final cp": "inventario_final_cp",
    "produccion total registrada": "produccion_total_archivo",
    "trazabilidad": "fuente_declarada",
}


def normalize_header(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce").dt.normalize()
    return pd.to_datetime(series, dayfirst=True, errors="coerce").dt.normalize()


def prepare_base(raw: pd.DataFrame, today: date | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepara la base tal cual se cargó; no rellena objetivos ni edita valores.

    Excluye semanas cuyo FIN sea posterior a la fecha actual de Ecuador, ya que
    todavía no representan semanas cerradas. extract_series delimita producción
    desde su primer positivo. La demanda desconocida no se convierte en cero;
    una declaración explícita de ausencia de ventas se registra por separado.
    """
    if today is None:
        today = datetime.now(APP_TIMEZONE).date()
    rename = {}
    known = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}
    for original in raw.columns:
        key = normalize_header(original)
        if key in known:
            rename[original] = known[key]
    present = set(rename.values())
    missing = [k for k in REQUIRED_COLUMNS.values() if k not in present]
    if missing:
        raise ValueError("La hoja seleccionada no contiene las columnas necesarias: " + ", ".join(missing))
    data = raw.rename(columns=rename).copy()
    data["fecha_inicio"] = _dates(data["fecha_inicio"])
    data["fecha_fin"] = _dates(data["fecha_fin"])
    if data[["fecha_inicio", "fecha_fin"]].isna().any().any():
        raise ValueError("Hay filas con fechas incompletas o no reconocibles. Revisa FECHA DE INICIO y FECHA DE FIN.")
    for c in REQUIRED_COLUMNS.values():
        if c.startswith("fecha"):
            continue
        original = data[c]
        converted = pd.to_numeric(original, errors="coerce")
        if ((original.notna()) & (original.astype(str).str.strip() != "") & converted.isna()).any():
            raise ValueError(f"La columna {c} contiene cantidades que no son numéricas.")
        if (converted.dropna() < 0).any():
            raise ValueError(f"La columna {c} contiene cantidades negativas.")
        data[c] = converted
    for c in ["inventario_inicial_cp", "inventario_final_cp", "produccion_total_archivo"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.sort_values("fecha_inicio", kind="stable").reset_index(drop=True)
    if data["fecha_inicio"].duplicated().any():
        raise ValueError("Hay semanas duplicadas por fecha de inicio.")
    step = data["fecha_inicio"].diff().dt.days.dropna()
    if not step.eq(7).all():
        raise ValueError("Las fechas de inicio no son semanales consecutivas (saltos de siete días).")
    if not ((data["fecha_fin"] - data["fecha_inicio"]).dt.days == 4).all():
        raise ValueError("Las fechas de fin deben corresponder al viernes de cada semana laboral (5 días).")
    current = data.loc[data["fecha_fin"].dt.date <= today].copy().reset_index(drop=True)
    count_excluded = len(data) - len(current)
    if len(current) < 50:
        raise ValueError("No hay al menos 50 semanas completas para la evaluación temporal.")
    if current["demanda_mq"].isna().any():
        raise ValueError("La demanda de maquila tiene celdas sin dato dentro de las semanas cerradas.")
    if "produccion_total_archivo" in current.columns:
        check = current["produccion_total_archivo"] - (current["demanda_mq"] + current["produccion_cp"].fillna(0))
        n_inconsistent = int((check.abs() > 0.01).sum())
    else:
        n_inconsistent = None
    info = {
        "filas_originales": len(raw),
        "semanas_completas": len(current),
        "semanas_no_cerradas": count_excluded,
        "fecha_ultima_semana": current["fecha_fin"].iloc[-1].date(),
        "fecha_primera_semana": current["fecha_inicio"].iloc[0].date(),
        "diferencias_en_total_archivo": n_inconsistent,
        "fuentes_declaradas": sorted(current["fuente_declarada"].dropna().astype(str).unique().tolist()) if "fuente_declarada" in current else [],
    }
    return current, info


def extract_series(history: pd.DataFrame, target: str) -> pd.DataFrame:
    """Respeta el inicio de cada modalidad; conserva ceros válidos posteriores."""
    if target not in TARGETS.values():
        raise ValueError(f"Serie desconocida: {target}")
    nonnull = history[target].notna().to_numpy()
    if not nonnull.any():
        raise ValueError(f"No hay datos para {target}.")
    first = int(np.flatnonzero(nonnull)[0])
    # La producción comienza con la primera semana con movimiento positivo,
    # porque semanas previas podrían pertenecer a una fase sin actividad.
    if target == "produccion_cp":
        positive = np.flatnonzero(history[target].fillna(0).to_numpy(float) > 0)
        if len(positive) == 0:
            raise ValueError("No hay semanas con producción positiva de corte propio.")
        first = int(positive[0])
    sub = history.loc[first:, ["fecha_inicio", target]].copy().reset_index(drop=True)
    if sub[target].isna().any():
        raise ValueError(f"Existen valores faltantes dentro del historial de {target}; no se reemplazan por cero.")
    if len(sub) < 50:
        raise ValueError(f"La serie {target} necesita al menos 50 semanas cerradas.")
    if np.var(sub[target].to_numpy(float)) < 1e-12:
        raise ValueError(f"La serie {target} no presenta variación suficiente para entrenar.")
    return sub.rename(columns={target: "valor"})


def metrics(y_true: np.ndarray, y_pred: np.ndarray, reference_train: np.ndarray | None = None) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if len(y) != len(p) or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Métricas: cantidades vacías o incompatibles.")
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    bias = float(np.mean(err))
    denom = float(np.sum(np.abs(y)))
    wape = float(np.sum(np.abs(err)) / denom * 100) if denom > 1e-12 else None
    ss_total = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - np.sum(np.square(err))/ss_total) if len(y) >= 2 and ss_total > 1e-12 else None
    mase = None
    if reference_train is not None:
        t = np.asarray(reference_train, dtype=float)
        scale = float(np.mean(np.abs(np.diff(t)))) if len(t) >= 2 else 0
        if scale > 1e-12:
            mase = mae / scale
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "Sesgo": bias, "WAPE (%)": wape, "MASE": mase}


def split_indices(n: int, lookback: int, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.55 <= train_ratio <= 0.8 or not 0.1 <= val_ratio <= 0.25 or train_ratio + val_ratio > 0.9:
        raise ValueError("Distribución temporal fuera de rango.")
    tr_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if tr_end - lookback < 10 or val_end - tr_end < 8 or n - val_end < 8:
        raise ValueError("No hay suficientes semanas en entrenamiento, validación o prueba para esa ventana.")
    return np.arange(lookback, tr_end), np.arange(tr_end, val_end), np.arange(val_end, n)


def _transform(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.log1p(y) - mean) / std


def _restore(yz: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.clip(np.asarray(yz, dtype=float) * std + mean, -20, 20)))


def _xy(z: np.ndarray, targets: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([z[i-lookback:i] for i in targets], axis=0).reshape(-1,lookback,1)
    y = z[targets].reshape(-1,1)
    return x.astype("float32"), y.astype("float32")


def _make_lstm(units: int, lookback: int, dropout: float, seed: int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback,1)),
        tf.keras.layers.LSTM(units, name="memoria_lstm"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(max(8,units//2), activation="relu"),
        tf.keras.layers.Dense(1, name="proximo_periodo"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), loss=tf.keras.losses.Huber())
    return model


@dataclass(frozen=True)
class Settings:
    lookback: int = 12
    epochs: int = 100
    units: int = 24
    batch_size: int = 16
    patience: int = 12
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    dropout: float = 0.15


@dataclass
class FitResult:
    name: str
    target: str
    series: pd.DataFrame
    settings: Settings
    model: Any
    z: np.ndarray
    mean: float
    std: float
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    train_reference: np.ndarray
    validation: pd.DataFrame
    training: pd.DataFrame
    train_metrics: dict
    validation_metrics: dict
    baseline_validation_metrics: dict
    loss: pd.DataFrame
    best_epoch: int


def _prediction_table(series: pd.DataFrame, idx: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    observed = series.loc[idx,"valor"].to_numpy(float)
    baselines = series.loc[idx-1,"valor"].to_numpy(float)
    return pd.DataFrame({
        "fecha_inicio": pd.to_datetime(series.loc[idx,"fecha_inicio"].to_numpy()),
        "real": observed,
        "LSTM": np.asarray(predictions,dtype=float).reshape(-1),
        "Referencia_ultimo_valor": baselines,
    })


def fit_one(name: str, series: pd.DataFrame, settings: Settings) -> FitResult:
    """Entrena LSTM y calcula TRAIN/VALIDACIÓN, sin utilizar métricas de TEST."""
    import tensorflow as tf
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    tf.keras.utils.set_random_seed(settings.seed)
    tf.config.experimental.enable_op_determinism()
    s = series.reset_index(drop=True)
    y = s["valor"].to_numpy(float)
    idx_train, idx_val, idx_test = split_indices(len(y),settings.lookback,settings.train_ratio,settings.val_ratio)
    end_train = int(idx_train[-1]+1)
    log_train=np.log1p(y[:end_train])
    mean=float(np.mean(log_train))
    std=float(np.std(log_train)) or 1.0
    z=_transform(y,mean,std)
    x_train,y_train=_xy(z,idx_train,settings.lookback)
    x_val,y_val=_xy(z,idx_val,settings.lookback)
    model=_make_lstm(settings.units,settings.lookback,settings.dropout,settings.seed)
    stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=settings.patience,restore_best_weights=True,min_delta=1e-4)
    hist=model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=settings.epochs,
                   batch_size=settings.batch_size,verbose=0,shuffle=False,callbacks=[stop])
    pred_train=_restore(model.predict(x_train,verbose=0).ravel(),mean,std)
    pred_val=_restore(model.predict(x_val,verbose=0).ravel(),mean,std)
    dtr=_prediction_table(s,idx_train,pred_train)
    dva=_prediction_table(s,idx_val,pred_val)
    best=int(np.argmin(hist.history["val_loss"])+1)
    return FitResult(name=name,target=TARGETS[name],series=s,settings=settings,model=model,z=z,mean=mean,std=std,
                     idx_train=idx_train,idx_val=idx_val,idx_test=idx_test,train_reference=y[:end_train],
                     validation=dva,training=dtr,
                     train_metrics=metrics(dtr.real.to_numpy(),dtr.LSTM.to_numpy(),y[:end_train]),
                     validation_metrics=metrics(dva.real.to_numpy(),dva.LSTM.to_numpy(),y[:end_train]),
                     baseline_validation_metrics=metrics(dva.real.to_numpy(),dva.Referencia_ultimo_valor.to_numpy(),y[:end_train]),
                     loss=pd.DataFrame({"epoch":np.arange(1,len(hist.history["loss"])+1),"train_loss":hist.history["loss"],"val_loss":hist.history["val_loss"]}),
                     best_epoch=best)


def final_test(fit: FitResult) -> tuple[pd.DataFrame, dict, dict]:
    """Única evaluación de test: no modifica arquitectura ni hiperparámetros."""
    x, _ = _xy(fit.z, fit.idx_test, fit.settings.lookback)
    pred=_restore(fit.model.predict(x,verbose=0).ravel(),fit.mean,fit.std)
    table=_prediction_table(fit.series,fit.idx_test,pred)
    actual=table.real.to_numpy(float)
    return table, metrics(actual,table.LSTM.to_numpy(float),fit.train_reference), metrics(actual,table.Referencia_ultimo_valor.to_numpy(float),fit.train_reference)


def full_refit_forecast(fit: FitResult, first_plan: date, weeks: int = 12) -> pd.DataFrame:
    """Ajuste final con TODOS los períodos históricos después del test cerrado.

    Usa la arquitectura fija y el número de épocas seleccionado SOLO en validación.
    Genera semanas puente para llegar a la fecha solicitada sin saltarse períodos.
    """
    import tensorflow as tf
    if first_plan.weekday() != 0:
        raise ValueError("La fecha inicial del plan debe ser lunes.")
    if weeks < 1 or weeks > 26:
        raise ValueError("El horizonte debe estar entre 1 y 26 semanas.")
    original=fit.series.reset_index(drop=True)
    last_start=original["fecha_inicio"].iloc[-1].date()
    next_date=last_start+timedelta(days=7)
    if first_plan < next_date:
        raise ValueError("El inicio del pronóstico debe ser posterior a la última semana completa.")
    bridge=(first_plan-next_date).days//7
    if next_date+timedelta(days=7*bridge)!=first_plan or bridge>16:
        raise ValueError("El inicio del horizonte no es una semana válida o está demasiado lejos.")
    y=original["valor"].to_numpy(float)
    log=np.log1p(y)
    mean=float(np.mean(log))
    std=float(np.std(log)) or 1.0
    z=_transform(y,mean,std)
    indices=np.arange(fit.settings.lookback,len(z))
    x_train,y_train=_xy(z,indices,fit.settings.lookback)
    model=_make_lstm(fit.settings.units,fit.settings.lookback,fit.settings.dropout,fit.settings.seed)
    model.fit(x_train,y_train,epochs=fit.best_epoch,batch_size=fit.settings.batch_size,
              shuffle=False,verbose=0)
    current=list(map(float,z[-fit.settings.lookback:]))
    predictions=[]
    for _ in range(bridge+weeks):
        batch=np.array(current[-fit.settings.lookback:],dtype="float32").reshape(1,fit.settings.lookback,1)
        nxt=float(model.predict(batch,verbose=0)[0,0])
        current.append(nxt)
        predictions.append(float(_restore(np.array([nxt]),mean,std)[0]))
    final=predictions[bridge:]
    return pd.DataFrame({
        "fecha_inicio":[first_plan+timedelta(days=7*i) for i in range(weeks)],
        fit.target:final,
    })


def sum_aligned(a: pd.DataFrame, a_col: str, b: pd.DataFrame, b_col: str, new: str) -> pd.DataFrame:
    sub=a[["fecha_inicio",a_col]].merge(b[["fecha_inicio",b_col]],on="fecha_inicio",how="inner",validate="one_to_one")
    if not sub.empty:
        sub[new]=sub[a_col].to_numpy(float)+sub[b_col].to_numpy(float)
    return sub


def make_forecast_table(forecasts: dict[str,pd.DataFrame]) -> pd.DataFrame:
    data=forecasts["Demanda de maquila"].copy()
    for name in ("Demanda de corte propio", "Producción de corte propio"):
        if name in forecasts:
            data=data.merge(forecasts[name],on="fecha_inicio",validate="one_to_one")
    data["demanda_total"]=data["demanda_mq"]+data["demanda_cp"]
    # Total conforme a la regla operativa del libro cargado:
    # demanda maquila asumida como producción MQ + producción registrada/pronosticada CP.
    if "produccion_cp" in data.columns:
        data["produccion_total_referencia"]=data["demanda_mq"]+data["produccion_cp"]
    data["fecha_fin"]=pd.to_datetime(data["fecha_inicio"])+pd.Timedelta(days=4)
    ordered=["fecha_inicio","fecha_fin","demanda_mq","demanda_cp","demanda_total"]
    if "produccion_cp" in data.columns:
        ordered+=["produccion_cp","produccion_total_referencia"]
    data=data[ordered]
    for c in ordered[2:]:
        data[c]=data[c].clip(lower=0)
    return data


def combined_test_metrics(tests: dict[str,tuple]) -> dict[str,tuple[pd.DataFrame,dict,dict]]:
    """Métricas calculadas SOBRE LAS SUMAS de semanas TEST coincidentes.

    A diferencia de sumar las métricas individuales, se alinean fechas y se
    evalúa la serie total observada y la predicción total correspondiente.
    """
    specs={
        "Demanda total":("Demanda de maquila","Demanda de corte propio"),
        "Producción total de referencia":("Demanda de maquila","Producción de corte propio"),
    }
    result={}
    for label,(a,b) in specs.items():
        if a not in tests or b not in tests:
            continue
        xa=tests[a][0]
        xb=tests[b][0]
        merged=xa.merge(xb,on="fecha_inicio",suffixes=("_a","_b"),how="inner",validate="one_to_one")
        if len(merged)<2:
            continue
        combined=pd.DataFrame({
            "fecha_inicio":merged.fecha_inicio,
            "real":merged.real_a+merged.real_b,
            "LSTM":merged.LSTM_a+merged.LSTM_b,
            "Referencia_ultimo_valor":merged.Referencia_ultimo_valor_a+merged.Referencia_ultimo_valor_b,
        })
        m=metrics(combined.real.to_numpy(float),combined.LSTM.to_numpy(float))
        baseline=metrics(combined.real.to_numpy(float),combined.Referencia_ultimo_valor.to_numpy(float))
        result[label]=(combined,m,baseline)
    return result


def recursive_test(fit: FitResult, weeks: int = 12) -> tuple[pd.DataFrame, dict, dict]:
    """Backtest multi-paso, sin introducir valores reales del TEST en la ventana.

    Se parte del último dato anterior a la prueba, se proyecta recursivamente
    un horizonte de hasta 12 semanas y SOLO después se compara con test.
    """
    n=min(int(weeks),len(fit.idx_test))
    if n<2:
        raise ValueError("Se necesitan por lo menos dos semanas de prueba recursiva.")
    first=int(fit.idx_test[0])
    current=list(map(float,fit.z[first-fit.settings.lookback:first]))
    preds=[]
    for _ in range(n):
        x=np.array(current[-fit.settings.lookback:],dtype="float32").reshape(1,fit.settings.lookback,1)
        zhat=float(fit.model.predict(x,verbose=0)[0,0])
        preds.append(float(_restore(np.array([zhat]),fit.mean,fit.std)[0]))
        current.append(zhat)
    idx=fit.idx_test[:n]
    actual=fit.series.loc[idx,"valor"].to_numpy(float)
    # Referencia multisemana: valor fijo del último punto antes del TEST.
    last=float(fit.series.loc[first-1,"valor"])
    reference=np.repeat(last,n)
    table=pd.DataFrame({
        "fecha_inicio":pd.to_datetime(fit.series.loc[idx,"fecha_inicio"].to_numpy()),
        "real":actual,
        "LSTM":preds,
        "Referencia_ultimo_valor":reference,
    })
    return table,metrics(actual,np.array(preds),fit.train_reference),metrics(actual,reference,fit.train_reference)
