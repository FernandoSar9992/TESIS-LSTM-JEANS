"""Motor reusable para pronóstico semanal y evaluación cronológica.

La interfaz importa este módulo. TensorFlow es opcional: si no está instalado,
los modelos estadísticos siguen funcionando y el diagnóstico informa el motivo.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib import metadata, util
import math
import random
import re
from typing import Any

import numpy as np
import pandas as pd

# Contrato de variables por cada paso temporal. Si se agrega o elimina una
# variable, actualice FEATURE_NAMES, _features y _future_lstm_input juntos.
FEATURE_NAMES = (
    "demanda",
    "media_movil_4",
    "media_movil_8",
    "diferencia",
    "estacionalidad_seno",
    "estacionalidad_coseno",
)
FEATURE_COUNT = len(FEATURE_NAMES)


@dataclass
class ForecastConfig:
    """Configuración serializable de un experimento completo.

    La interfaz construye una instancia y el motor la recibe sin depender de
    Streamlit. Esto permite reproducir un experimento desde una prueba o script.
    """
    horizon: int = 8
    lookback: int = 12
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    seasonal_period: int = 52
    moving_window: int = 4
    ridge_alpha: float = 100.0
    lstm_units_1: int = 32
    lstm_units_2: int = 16
    dense_units: int = 16
    dropout: float = 0.20
    learning_rate: float = 0.001
    epochs: int = 150
    patience: int = 20
    batch_size: int = 8
    seed: int = 42

    def validate(self, n: int) -> list[str]:
        errors: list[str] = []
        if self.horizon < 1: errors.append("El horizonte debe ser mayor que cero.")
        if self.lookback < 2: errors.append("El lookback debe ser al menos 2.")
        if self.lookback + self.horizon >= n: errors.append("Lookback + horizonte excede la serie disponible.")
        if not 0.50 <= self.train_ratio <= 0.90: errors.append("Entrenamiento debe estar entre 50 % y 90 %.")
        if not 0.05 <= self.validation_ratio <= 0.30: errors.append("Validación debe estar entre 5 % y 30 %.")
        if self.train_ratio + self.validation_ratio >= 0.95: errors.append("Debe reservarse al menos 5 % para prueba.")
        if self.seasonal_period < 2 or self.moving_window < 1: errors.append("Los períodos y ventanas deben ser positivos.")
        if self.ridge_alpha <= 0: errors.append("Alpha debe ser positivo.")
        if min(self.lstm_units_1, self.lstm_units_2, self.dense_units, self.epochs, self.patience, self.batch_size) < 1:
            errors.append("Unidades, épocas, paciencia y lote deben ser positivos.")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0: errors.append("Dropout o tasa de aprendizaje inválidos.")
        return errors


MONTHS = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()


def period_positions(labels):
    """Reconoce el calendario del proyecto: cuatro períodos por mes, 48 al año."""
    positions = []
    for label in labels:
        match = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})-Se([1-4])", str(label).strip(), re.I)
        if not match or match[1].lower() not in MONTHS:
            return None
        positions.append(int(match[2])*48 + MONTHS.index(match[1].lower())*4 + int(match[3])-1)
    return np.asarray(positions) if positions else None


def suggested_seasonal_period(data):
    return 48 if period_positions(data["periodo"]) is not None else 52


def future_labels(data, horizon):
    positions = period_positions(data["periodo"])
    if positions is not None:
        labels = []
        for pos in range(int(positions[-1])+1, int(positions[-1])+horizon+1):
            year, within = divmod(pos, 48)
            month, week = divmod(within, 4)
            labels.append(f"{MONTHS[month].capitalize()} {year}-Se{week+1}")
        return labels
    return [f"Período +{h}" for h in range(1, horizon+1)]


def clean_series(frame: pd.DataFrame, label_col: str, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Selecciona y limpia la serie sin alterar el DataFrame de origen.

    Returns:
        Un DataFrame estándar con columnas ``periodo`` y ``demanda``, más una
        lista de advertencias descriptivas para la interfaz.
    """
    if label_col == target_col:
        raise ValueError("Seleccione columnas distintas para período y demanda.")
    out = frame[[label_col, target_col]].copy()
    out.columns = ["periodo", "demanda"]
    if out["periodo"].isna().any():
        raise ValueError("Hay períodos vacíos. Corrija la fuente antes de continuar.")
    out["periodo"] = out["periodo"].astype(str).str.strip()
    before = len(out)
    out["demanda"] = pd.to_numeric(out["demanda"], errors="coerce")
    if not np.isfinite(out["demanda"]).all():
        raise ValueError("Hay demandas vacías, no numéricas o infinitas. Corríjalas; eliminar filas alteraría el tiempo.")
    out = out.reset_index(drop=True)
    notes = [f"Se eliminaron {before-len(out)} filas no numéricas o vacías."] if before != len(out) else []
    if (out["demanda"] < 0).any(): raise ValueError("La demanda debe ser no negativa.")
    if out["periodo"].duplicated().any() or out["periodo"].eq("").any():
        raise ValueError("Hay períodos duplicados o vacíos.")
    positions = period_positions(out["periodo"])
    if positions is not None:
        if not np.all(np.diff(positions) == 1):
            raise ValueError("Los períodos deben estar ordenados, sin saltos: cuatro por mes.")
        notes.append("Calendario de cuatro períodos por mes: 48 por año; no equivale a semanas ISO.")
    else:
        notes.append("Se conserva el orden del archivo. Confirme que los períodos sean consecutivos y de igual frecuencia.")
    return out, notes


def metrics(real: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas comunes sobre matrices de cualquier forma compatible.

    Las entradas se aplanan deliberadamente para la métrica global. Las métricas
    por horizonte se calculan por separado en ``run_experiment``.
    """
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    if real.shape != pred.shape or real.size == 0 or not (np.isfinite(real).all() and np.isfinite(pred).all()):
        raise ValueError("Las métricas requieren datos finitos, no vacíos y con la misma forma.")
    real, pred = real.ravel(), pred.ravel()
    err = real - pred
    mse = float(np.mean(err ** 2))
    nz = np.abs(real) > 1e-8
    denom_smape = np.abs(real) + np.abs(pred)
    ss_tot = float(np.sum((real - real.mean()) ** 2))
    return {
        "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err[nz] / real[nz])) * 100) if nz.any() else np.nan,
        "WAPE_%": float(np.sum(np.abs(err)) / np.sum(np.abs(real)) * 100) if np.sum(np.abs(real)) else np.nan,
        "sMAPE_%": float(np.mean(np.divide(2*np.abs(err), denom_smape, out=np.zeros_like(err), where=denom_smape > 1e-8))*100),
        "MSE": mse, "R2": float(1 - np.sum(err**2)/ss_tot) if ss_tot else np.nan,
        "Sesgo": float(np.mean(pred-real)),
    }


def _features(y: np.ndarray, lookback: int, seasonal_period: int, horizon: int):
    """Crea ventanas supervisadas sin incluir valores posteriores al origen.

    Cada fila de ``xs`` representa ``lookback`` pasos por ``FEATURE_COUNT``
    variables; ``ys`` contiene las siguientes ``horizon`` demandas.
    """
    xs, ys, origins = [], [], []
    for t in range(lookback, len(y)-horizon+1):
        lag = y[t-lookback:t]
        idx = np.arange(t-lookback, t)
        roll4 = pd.Series(lag).rolling(min(4, lookback), min_periods=1).mean().to_numpy()
        roll8 = pd.Series(lag).rolling(min(8, lookback), min_periods=1).mean().to_numpy()
        diff = np.r_[0, np.diff(lag)]
        x = np.column_stack([lag, roll4, roll8, diff,
                             np.sin(2*np.pi*idx/seasonal_period), np.cos(2*np.pi*idx/seasonal_period)]).ravel()
        xs.append(x); ys.append(y[t:t+horizon]); origins.append(t)
    return np.asarray(xs), np.asarray(ys), np.asarray(origins)


def _ridge_fit(x, y, alpha):
    """Ajusta Ridge multisalida; el intercepto no recibe penalización."""
    mu, sd = x.mean(0), x.std(0); sd[sd == 0] = 1
    z = (x-mu)/sd; design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1])*alpha; penalty[0, 0] = 0
    coef = np.linalg.solve(design.T@design + penalty, design.T@y)
    return mu, sd, coef


def _ridge_predict(model, x):
    """Aplica exactamente la media y desviación aprendidas en entrenamiento."""
    mu, sd, coef = model
    return np.column_stack([np.ones(len(x)), (x-mu)/sd])@coef


def tensorflow_status() -> tuple[bool, str]:
    """Comprueba TensorFlow sin hacer que sea obligatorio para modelos base."""
    # No importamos TensorFlow aquí: cargar sus librerías nativas puede tardar
    # varios segundos. La importación real ocurre únicamente al solicitar LSTM.
    if util.find_spec("tensorflow") is None:
        return False, "TensorFlow no está instalado en el entorno activo."
    try:
        version = metadata.version("tensorflow")
    except metadata.PackageNotFoundError:
        version = "versión desconocida"
    return True, f"TensorFlow {version} disponible"


def _build_lstm(config: ForecastConfig, n_features: int):
    """Construye una LSTM directa: una salida por cada paso del horizonte."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((config.lookback, n_features)),
        tf.keras.layers.LSTM(config.lstm_units_1, return_sequences=True),
        tf.keras.layers.Dropout(config.dropout),
        tf.keras.layers.LSTM(config.lstm_units_2),
        tf.keras.layers.Dropout(config.dropout),
        tf.keras.layers.Dense(config.dense_units, activation="relu"),
        tf.keras.layers.Dense(config.horizon),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate), loss="mse")
    return model


def _lstm_arrays(y: np.ndarray, config: ForecastConfig, fit_end: int):
    """Ajusta escala SOLO antes de fit_end; transforma sin recortar todo el histórico."""
    if not 0 < fit_end <= len(y): raise ValueError("Corte de escalado inválido.")
    scale_min, scale_max = float(y[:fit_end].min()), float(y[:fit_end].max())
    span = scale_max-scale_min or 1.0
    scaled = (y-scale_min)/span
    xflat, targets, origins = _features(scaled, config.lookback, config.seasonal_period, config.horizon)
    x = xflat.reshape(len(xflat), config.lookback, FEATURE_COUNT)
    return x, targets, origins, scale_min, span


def run_experiment(data: pd.DataFrame, config: ForecastConfig, models: list[str]) -> dict[str, Any]:
    """Ejecuta un experimento cronológico y devuelve resultados tabulares.

    El conjunto de prueba nunca se usa para ajustar los modelos. El modelo
    ganador se determina por menor RMSE de validación. Consulte ``GUIA_TECNICA.md`` para
    el contrato completo del diccionario devuelto.
    """
    data, _ = clean_series(data, "periodo", "demanda")
    y = data["demanda"].to_numpy(float); n = len(y)
    validation_errors = config.validate(n)
    if validation_errors: raise ValueError(" ".join(validation_errors))
    random.seed(config.seed); np.random.seed(config.seed)
    train_end = int(n*config.train_ratio); val_end = int(n*(config.train_ratio+config.validation_ratio))
    x, target, origins = _features(y, config.lookback, config.seasonal_period, config.horizon)
    train_val = origins + config.horizon - 1 < val_end
    train = origins + config.horizon - 1 < train_end
    val = (origins >= train_end) & (origins + config.horizon - 1 < val_end)
    test = (origins >= val_end) & (origins + config.horizon - 1 < n)
    if train.sum() < 20 or val.sum() < 2 or test.sum() < 2:
        raise ValueError("La partición deja muy pocas ventanas completas. Reduzca horizonte/lookback o cambie porcentajes.")
    predictions: dict[str, np.ndarray] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    future: dict[str, np.ndarray] = {}
    histories: dict[str, Any] = {}
    test_origins, real_test = origins[test], target[test]

    if "Ridge directa" in models:
        ridge_train = _ridge_fit(x[train], target[train], config.ridge_alpha)
        validation_predictions["Ridge directa"] = np.maximum(_ridge_predict(ridge_train, x[val]), 0)
        ridge = _ridge_fit(x[train_val], target[train_val], config.ridge_alpha)
        predictions["Ridge directa"] = np.maximum(_ridge_predict(ridge, x[test]), 0)
        idx = np.arange(n-config.lookback, n); lag = y[-config.lookback:]
        r4 = pd.Series(lag).rolling(min(4, config.lookback), min_periods=1).mean().to_numpy()
        r8 = pd.Series(lag).rolling(min(8, config.lookback), min_periods=1).mean().to_numpy()
        xf = np.column_stack([lag, r4, r8, np.r_[0,np.diff(lag)], np.sin(2*np.pi*idx/config.seasonal_period), np.cos(2*np.pi*idx/config.seasonal_period)]).ravel()[None,:]
        ridge_all = _ridge_fit(x, target, config.ridge_alpha)
        future["Ridge directa"] = np.maximum(_ridge_predict(ridge_all, xf)[0], 0)
    if "Último valor" in models:
        validation_predictions["Último valor"] = np.asarray([np.repeat(y[t-1], config.horizon) for t in origins[val]])
        predictions["Último valor"] = np.asarray([np.repeat(y[t-1], config.horizon) for t in test_origins])
        future["Último valor"] = np.repeat(y[-1], config.horizon)
    if "Media móvil" in models:
        w = config.moving_window
        validation_predictions["Media móvil"] = np.asarray([np.repeat(y[max(0,t-w):t].mean(), config.horizon) for t in origins[val]])
        predictions["Media móvil"] = np.asarray([np.repeat(y[max(0,t-w):t].mean(), config.horizon) for t in test_origins])
        future["Media móvil"] = np.repeat(y[-w:].mean(), config.horizon)
    if "Media histórica" in models:
        validation_predictions["Media histórica"] = np.full(target[val].shape, y[:train_end].mean())
        predictions["Media histórica"] = np.full(real_test.shape, y[:val_end].mean())
        future["Media histórica"] = np.repeat(y.mean(), config.horizon)
    if "Estacional ingenuo" in models and config.seasonal_period >= config.horizon and origins[val].min() >= config.seasonal_period:
        p = config.seasonal_period
        validation_predictions["Estacional ingenuo"] = np.asarray([[y[t+h-p] for h in range(config.horizon)] for t in origins[val]])
        predictions["Estacional ingenuo"] = np.asarray([[y[t+h-p] for h in range(config.horizon)] for t in test_origins])
        future["Estacional ingenuo"] = np.asarray([y[n+h-p] for h in range(config.horizon)])
    elif "Estacional ingenuo" in models:
        histories["Estacional_error"] = "No hay historia estacional suficiente en validación, o el horizonte supera el ciclo."
    if "LSTM" in models:
        ok, status = tensorflow_status()
        if not ok: histories["LSTM_error"] = status
        else:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            xl, yl, ol, minimum, span = _lstm_arrays(y, config, train_end)
            tr = ol + config.horizon - 1 < train_end
            va = (ol >= train_end) & (ol + config.horizon - 1 < val_end)
            te = (ol >= val_end) & (ol + config.horizon - 1 < n)
            model = _build_lstm(config, FEATURE_COUNT)
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True)]
            hist = model.fit(xl[tr], yl[tr], validation_data=(xl[va], yl[va]), epochs=config.epochs,
                             batch_size=min(config.batch_size, int(tr.sum())), shuffle=False, verbose=0, callbacks=callbacks)
            validation_predictions["LSTM"] = np.maximum(model.predict(xl[va], verbose=0)*span+minimum, 0)
            best_epochs = int(np.argmin(hist.history["val_loss"])+1)
            # Fijar épocas antes de prueba y ajustar de nuevo en train + validación,
            # con su propio escalador, igual que Ridge.
            xt, yt, _, test_minimum, test_span = _lstm_arrays(y, config, val_end)
            test_model = _build_lstm(config, FEATURE_COUNT)
            test_model.fit(xt[train_val], yt[train_val], epochs=best_epochs,
                           batch_size=min(config.batch_size, int(train_val.sum())), shuffle=False, verbose=0)
            predictions["LSTM"] = np.maximum(test_model.predict(xt[te], verbose=0)*test_span+test_minimum, 0)
            xf, yf, _, final_minimum, final_span = _lstm_arrays(y, config, n)
            final_model = _build_lstm(config, FEATURE_COUNT)
            final_model.fit(xf, yf, epochs=best_epochs, batch_size=min(config.batch_size,len(xf)), shuffle=False, verbose=0)
            future_input = _future_lstm_input(y, config, final_minimum, final_span)
            future["LSTM"] = np.maximum(
                final_model.predict(future_input, verbose=0)[0] * final_span + final_minimum,
                0,
            )
            histories["LSTM"] = {"loss": hist.history["loss"], "val_loss": hist.history["val_loss"], "best_epochs": best_epochs,
                                 "parameters": model.count_params(), "train_windows": int(tr.sum()),
                                 "scale_train": [minimum, span], "scale_test": [test_minimum, test_span]}

    if not predictions: raise ValueError("Ningún modelo pudo ejecutarse.")
    metric_rows, horizon_rows, detail_rows = [], [], []
    for name, pred in predictions.items():
        metric_rows.append({"Modelo": name, **metrics(real_test, pred)})
        for h in range(config.horizon): horizon_rows.append({"Modelo": name, "Horizonte": h+1, **metrics(real_test[:,h], pred[:,h])})
    for i,t in enumerate(test_origins):
        for h in range(config.horizon):
            row = {"Origen": int(t), "Periodo": data.iloc[t+h]["periodo"], "Horizonte": h+1, "Real": real_test[i,h]}
            row.update({name: pred[i,h] for name,pred in predictions.items()}); detail_rows.append(row)
    metric_df = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    validation_df = pd.DataFrame([{"Modelo": name, **metrics(target[val], pred)} for name, pred in validation_predictions.items()]).sort_values("RMSE").reset_index(drop=True)
    winner = str(validation_df.iloc[0]["Modelo"])
    residual = np.abs(target[val]-validation_predictions[winner])
    q95 = np.quantile(residual, .95, axis=0, method="higher")
    forecast = pd.DataFrame({"Periodo": future_labels(data, config.horizon), "Horizonte": np.arange(1,config.horizon+1), "Modelo": winner,
                             "Pronóstico": future[winner], "Límite inferior aprox.": np.maximum(future[winner]-q95,0),
                             "Límite superior aprox.": future[winner]+q95})
    return {"config": asdict(config), "metrics": metric_df, "validation_metrics": validation_df,
            "future_all": pd.DataFrame({"Periodo": future_labels(data, config.horizon), "Horizonte": np.arange(1,config.horizon+1), **future}),
            "horizon_metrics": pd.DataFrame(horizon_rows), "validation_origins": int(val.sum()), "train_windows": int(train.sum()),
            "detail": pd.DataFrame(detail_rows), "forecast": forecast, "winner": winner,
            "history": histories, "train_end": train_end, "val_end": val_end, "test_origins": int(test.sum())}


def _future_lstm_input(y: np.ndarray, config: ForecastConfig, minimum: float, span: float):
    """Replica para el futuro las mismas seis variables usadas al entrenar."""
    scaled = (y-minimum)/span; lag = scaled[-config.lookback:]; idx = np.arange(len(y)-config.lookback,len(y))
    r4 = pd.Series(lag).rolling(min(4,config.lookback),min_periods=1).mean().to_numpy()
    r8 = pd.Series(lag).rolling(min(8,config.lookback),min_periods=1).mean().to_numpy()
    return np.column_stack([lag,r4,r8,np.r_[0,np.diff(lag)],np.sin(2*np.pi*idx/config.seasonal_period),np.cos(2*np.pi*idx/config.seasonal_period)])[None,:,:]


def diagnostic_report(data: pd.DataFrame, result: dict[str, Any] | None, config: ForecastConfig) -> pd.DataFrame:
    """Genera controles interpretables; no modifica resultados ni modelos."""
    rows = []
    def add(check, ok, detail, action=""):
        rows.append({"Control": check, "Estado": "CORRECTO" if ok else "REVISAR", "Detalle": detail, "Acción recomendada": action})
    add("Cantidad de datos", len(data) >= 100, f"{len(data)} observaciones", "Idealmente use 150–300+ semanas para LSTM.")
    add("Valores faltantes", not data["demanda"].isna().any(), f"{int(data['demanda'].isna().sum())} faltantes")
    add("Variabilidad", data["demanda"].std() > 0, f"Desviación estándar: {data['demanda'].std():.2f}")
    ok_tf, tf_msg = tensorflow_status(); add("TensorFlow / LSTM", ok_tf, tf_msg, "Ejecute: pip install -r requirements.txt")
    if result:
        best = result["metrics"].set_index("Modelo").loc[result["winner"]]
        add("Ventanas de prueba", result["test_origins"] >= 10, f"{result['test_origins']} orígenes completos", "Reduzca horizonte o consiga más datos.")
        add("R² del mejor modelo", best["R2"] > 0, f"R²={best['R2']:.3f}", "Un R² negativo indica capacidad predictiva insuficiente.")
        add("WAPE ≤ 20 % (referencia, no garantía)", best["WAPE_%"] <= 20, f"WAPE={best['WAPE_%']:.2f}%", "Defina la tolerancia operativa; 20 % no es un umbral universal.")
        scores = result["metrics"].set_index("Modelo")
        if "LSTM" in scores.index and "Media histórica" in scores.index:
            add("LSTM supera media histórica en prueba", scores.loc["LSTM", "RMSE"] < scores.loc["Media histórica", "RMSE"],
                f"RMSE LSTM={scores.loc['LSTM', 'RMSE']:.2f}; media={scores.loc['Media histórica', 'RMSE']:.2f}",
                "Compruebe estabilidad en varios cortes temporales antes de aceptar LSTM.")
        add("Calendario y ciclo", period_positions(data["periodo"]) is None or config.seasonal_period == 48,
            f"Ciclo configurado: {config.seasonal_period} períodos", "El calendario de cuatro períodos mensuales requiere 48 para un ciclo anual.")
        add("Pronóstico generado", len(result["forecast"]) == config.horizon, f"{len(result['forecast'])} periodos")
    return pd.DataFrame(rows)
