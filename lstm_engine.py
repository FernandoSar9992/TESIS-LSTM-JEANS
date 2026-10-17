"""Pronóstico genérico exclusivamente LSTM con selección temporal por archivo.

El motor anterior se conserva para reproducir la auditoría histórica. Este motor
no usa sus calendarios ni sus modelos: solo reutiliza las fórmulas de métricas.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Callable

import numpy as np
import pandas as pd

from forecast_engine import metrics, tensorflow_status

ENGINE_VERSION = "lstm-adaptativo-v2"


@dataclass(frozen=True)
class LSTMConfig:
    horizon: int = 8
    automatic: bool = True
    lookback: int = 24
    units: int = 32
    representation: str = "nivel"
    dropout: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 120
    patience: int = 15
    batch_size: int = 16
    members: int = 3
    seed: int = 42
    recent_weight: bool = True

    def validate(self):
        for name in ("horizon", "lookback", "units", "epochs", "patience", "batch_size", "members", "seed"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} debe ser un entero válido.")
        if not 2 <= self.lookback <= 256 or not 4 <= self.units <= 128:
            raise ValueError("Ventana debe estar entre 2 y 256; unidades entre 4 y 128.")
        if self.members not in (1, 3):
            raise ValueError("Seleccione uno o tres entrenamientos LSTM.")
        if not 0 <= self.dropout <= 0.6 or not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .1:
            raise ValueError("Dropout o tasa de aprendizaje inválidos.")
        if self.representation not in ("nivel", "cambio", "local"):
            raise ValueError("Representación LSTM inválida.")


def split_points(n, horizon):
    """Bloques completos incluso con horizontes mayores al 15 % de la serie."""
    block = max(horizon+2, int(np.ceil(n*.15)))
    train_end, val_end = n-2*block, n-block
    if train_end < horizon+21:
        raise ValueError("Histórico insuficiente para entrenar y reservar validación y prueba completas. Agregue períodos o reduzca el horizonte.")
    return train_end, val_end


def suggest_cycle(train):
    """Indicio de ciclo a partir SOLO de entrenamiento, sin calendario impuesto."""
    if len(train) < 24:
        return None
    idx = np.arange(len(train))
    detrended = train - np.polyval(np.polyfit(idx, train, 1), idx)
    if np.std(detrended) < 1e-8:
        return None
    limit = min(128, len(train)//2)
    corr = np.array([np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1]
                     if np.std(detrended[:-lag]) > 1e-8 and np.std(detrended[lag:]) > 1e-8 else 0
                     for lag in range(1, limit+1)])
    peaks = [lag for lag in range(2, limit+1) if corr[lag-1] > .35 and corr[lag-1] >= corr[lag-2]
             and (lag == limit or corr[lag-1] >= corr[lag])]
    if not peaks:
        return None
    strongest = max(corr[lag-1] for lag in peaks)
    return min(lag for lag in peaks if corr[lag-1] >= .9*strongest)


def candidates_for(train, config):
    cap = min(128, len(train)-config.horizon-19)
    if not config.automatic:
        if config.lookback > cap:
            raise ValueError(f"La ventana manual deja pocas muestras. Use como máximo {cap} o active ajuste automático.")
        return [{"lookback": config.lookback, "units": config.units, "representation": config.representation}]
    short = min(cap, max(6, min(24, config.horizon*2)))
    cycle = suggest_cycle(train)
    long = min(cap, max(short*2, 2*cycle if cycle else 0))
    windows = list(dict.fromkeys([short, long]))
    return [{"lookback": w, "units": 16 if i == 0 else 32, "representation": mode}
            for mode in ("nivel", "cambio", "local") for i, w in enumerate(windows)]


def validation_folds(train_end, val_end, config):
    """Orígenes expansivos anteriores a prueba, objetivos de bloques disjuntos."""
    earlier = train_end-(val_end-train_end)
    min_window = 6 if config.automatic else config.lookback
    folds = []
    if earlier >= min_window+config.horizon+19:
        folds.append((earlier, train_end))
    folds.append((train_end, val_end))
    return folds


def window_encoding(lag, representation):
    """Transformación causal, idéntica al evaluar y pronosticar.

    Local usa nivel reciente y variabilidad de la propia ventana. El factor
    nunca depende de valores posteriores al origen.
    """
    if representation == "local":
        anchor = float(np.mean(lag[-min(8, len(lag)):]))
        factor = max(float(np.std(lag)), .25)
        return np.column_stack(((lag-anchor)/factor, np.r_[0., np.diff(lag)]/factor)), np.array([anchor, factor])
    anchor = lag[-1] if representation == "cambio" else 0.
    return np.column_stack((lag-anchor, np.r_[0., np.diff(lag)])), anchor


def anchor_parts(anchors):
    values = np.asarray(anchors)
    if values.ndim == 2:
        return values[:, 0], values[:, 1]
    return values, np.ones_like(values)


def sample_weights(anchors, recent=False):
    """Factor² devuelve MSE a escala común; recencia solo pesa entrenamiento."""
    _, factors = anchor_parts(anchors)
    weights = factors**2
    if recent:
        weights = weights * np.exp2(np.linspace(-2., 0., len(weights)))
    return weights.astype(np.float32)


def seasonal_encoding(z, origin, lookback, representation, anchor, periodicity,
                      calendar_offset=0, calendar_extra=False):
    """Variables conocidas antes del pronóstico. Nunca usan demanda futura.

    La fase se alinea con el calendario real: 0=Enero-Se1 ... 47=Diciembre-Se4.
    La feria es una indicadora aproximada (Diciembre-Se3 y Se4), no ventas reales.
    """
    indices = np.arange(origin-lookback+1, origin+1)
    offset, factor = anchor if representation == "local" else (anchor, 1.)
    previous = indices-periodicity
    available = previous >= 0
    lag = np.zeros(lookback)
    lag[available] = (z[previous[available]]-offset)/factor
    phase = 2*np.pi*(indices + calendar_offset)/periodicity
    features = [np.sin(phase), np.cos(phase), lag, available.astype(float)]
    if calendar_extra:
        if periodicity != 48:
            raise ValueError("La codificación de ferias requiere 48 períodos por año.")
        within_year = (indices + calendar_offset) % 48
        month = within_year // 4  # Enero=0, Diciembre=11
        segment = within_year % 4  # Se1=0, Se4=3
        fair = ((month == 11) & (segment >= 2)).astype(float)
        low_season = (month <= 1).astype(float)
        features.extend([np.sin(2*phase), np.cos(2*phase), fair, low_season])
    return np.column_stack(features)


def scaled_windows(y, fit_end, lookback, horizon, representation, periodicity=None,
                   calendar_offset=0, calendar_extra=False):
    """Escala aprendida en prefijo. Cambio predice desplazamientos desde último valor."""
    center = float(np.mean(y[:fit_end]))
    scale = float(np.std(y[:fit_end]))
    if scale < 1e-8:
        scale = max(abs(center)*1e-6, 1.)
    z = (y-center)/scale
    origins = np.arange(lookback, len(y)-horizon+1)
    x, target, anchors = [], [], []
    for origin in origins:
        lag = z[origin-lookback:origin]
        encoded, anchor = window_encoding(lag, representation)
        if periodicity is not None:
            encoded = np.column_stack((encoded, seasonal_encoding(z, origin, lookback, representation, anchor, periodicity,
                                                                         calendar_offset, calendar_extra)))
        x.append(encoded)
        offset, factor = (anchor[0], anchor[1]) if representation == "local" else (anchor, 1.)
        target.append((z[origin:origin+horizon]-offset)/factor)
        anchors.append(anchor)
    arrays = np.asarray(x, np.float32), np.asarray(target, np.float32)
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("La magnitud de los datos supera la estabilidad numérica. Revise sus unidades.")
    return (*arrays, origins, np.asarray(anchors), center, scale)


def future_input(y, lookback, representation, center, scale, periodicity=None,
                 calendar_offset=0, calendar_extra=False):
    lag = (y[-lookback:]-center)/scale
    encoded, anchor = window_encoding(lag, representation)
    if periodicity is not None:
        encoded = np.column_stack((encoded, seasonal_encoding((np.asarray(y)-center)/scale, len(y), lookback, representation,
                                                                     anchor, periodicity, calendar_offset, calendar_extra)))
    x = encoded[None].astype(np.float32)
    return x, anchor


def build_network(candidate, config, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    feature_count = 2 + (4 if candidate.get("periodicity") else 0) + (4 if candidate.get("calendar_extra") else 0)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((candidate["lookback"], feature_count)),
        tf.keras.layers.LSTM(candidate["units"]),
        tf.keras.layers.Dropout(config.dropout),
        tf.keras.layers.Dense(config.horizon, kernel_initializer="zeros"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate, clipnorm=1.), loss="mse")
    return model


def inverse_prediction(model, x, anchors, center, scale):
    prediction = np.asarray(model.predict(x, verbose=0), float)
    offsets, factors = anchor_parts(anchors)
    values = (prediction*factors.reshape(-1, 1)+offsets.reshape(-1, 1))*scale+center
    if not np.isfinite(values).all():
        raise ValueError("LSTM produjo valores no finitos; revise la escala y el entrenamiento.")
    return np.maximum(values, 0)


def run_lstm(data, config, future_periods=None, progress: Callable[[str], None] | None = None):
    config.validate()
    y = pd.to_numeric(data["demanda"], errors="coerce").to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any() or len(y) == 0:
        raise ValueError("Use una serie no vacía de demandas finitas y no negativas.")
    if data["periodo"].isna().any() or data["periodo"].duplicated().any():
        raise ValueError("Cada período debe ser único y no vacío.")
    train_end, val_end = split_points(len(y), config.horizon)
    folds = validation_folds(train_end, val_end, config)
    candidates = candidates_for(y[:folds[0][0]], config)
    ok, message = tensorflow_status()
    if not ok:
        raise RuntimeError(message + " Instale requirements.txt para entrenar LSTM.")
    import tensorflow as tf
    notify = progress or (lambda message: None)
    rows, fold_rows, selected, best_score, selected_history = [], [], None, float("inf"), None
    for number, candidate in enumerate(candidates, 1):
        fold_real, fold_pred, fold_rmse = [], [], []
        for fold_no, (fit_end, stop) in enumerate(folds, 1):
            notify(f"LSTM {number}/{len(candidates)} · corte {fold_no}/{len(folds)} · ventana {candidate['lookback']} · {candidate['representation']}.")
            tf.keras.backend.clear_session()
            x, target, origins, anchors, center, scale = scaled_windows(y[:stop], fit_end, candidate["lookback"], config.horizon, candidate["representation"])
            train = origins+config.horizon <= fit_end
            val = origins >= fit_end
            model = build_network(candidate, config, config.seed)
            hist = model.fit(x[train], target[train], sample_weight=sample_weights(anchors[train], config.recent_weight),
                             validation_data=(x[val], target[val], sample_weights(anchors[val])),
                             epochs=config.epochs, batch_size=min(config.batch_size, int(train.sum())), shuffle=False, verbose=0,
                             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True),
                                        tf.keras.callbacks.TerminateOnNaN()])
            if not np.isfinite(hist.history["val_loss"]).all():
                raise ValueError("El entrenamiento perdió estabilidad. Reduzca la tasa de aprendizaje.")
            prediction = inverse_prediction(model, x[val], anchors[val], center, scale)
            real = np.array([y[t:t+config.horizon] for t in origins[val]])
            score = metrics(real, prediction)
            epochs = int(np.argmin(hist.history["val_loss"])+1)
            fold_real.append(real)
            fold_pred.append(prediction)
            fold_rmse.append(score["RMSE"])
            fold_rows.append({"Variante": number, "Corte": fold_no, "Entrenamiento_hasta": fit_end,
                              "Validacion_hasta": stop, "Origenes": int(val.sum()), "Epocas": epochs, **score})
        pooled = metrics(np.concatenate(fold_real), np.concatenate(fold_pred))
        # Penalización pequeña, fija y explícita a la inestabilidad entre cortes.
        instability = float(np.std(fold_rmse))
        selection_score = pooled["RMSE"] + .1*instability
        rows.append({"Variante": number, **candidate, "epochs": epochs, "RMSE_validacion": pooled["RMSE"],
                     "MAE_validacion": pooled["MAE"], "Dispersion_RMSE": instability, "Criterio_seleccion": selection_score,
                     "Cortes": len(folds)})
        if selection_score < best_score:
            selected = {**candidate, "epochs": epochs}
            best_score = selection_score
            selected_history = {"loss": hist.history["loss"], "val_loss": hist.history["val_loss"], "parameters": model.count_params()}
    # Todos los parámetros y épocas están fijados antes de consultar la prueba.
    test_predictions, future_predictions = [], []
    test_origins = np.arange(val_end, len(y)-config.horizon+1)
    for stage, fit_end in (("prueba", val_end), ("futuro", len(y))):
        x, targets, origins, anchors, center, scale = scaled_windows(y, fit_end, selected["lookback"], config.horizon, selected["representation"])
        train = origins+config.horizon <= fit_end
        test = origins >= val_end
        xf, af = future_input(y, selected["lookback"], selected["representation"], center, scale)
        for member in range(config.members):
            notify(f"Ajuste LSTM para {stage}: entrenamiento {member+1}/{config.members}.")
            tf.keras.backend.clear_session()
            model = build_network(selected, config, config.seed+member)
            model.fit(x[train], targets[train], sample_weight=sample_weights(anchors[train], config.recent_weight),
                      epochs=selected["epochs"], batch_size=min(config.batch_size, int(train.sum())),
                      shuffle=False, verbose=0, callbacks=[tf.keras.callbacks.TerminateOnNaN()])
            if stage == "prueba":
                test_predictions.append(inverse_prediction(model, x[test], anchors[test], center, scale))
            else:
                future_predictions.append(inverse_prediction(model, xf, [af], center, scale)[0])
    pred = np.mean(test_predictions, axis=0)
    future = np.mean(future_predictions, axis=0)
    real = np.array([y[t:t+config.horizon] for t in test_origins])
    score = metrics(real, pred)
    detail = [{"Origen": int(t), "Periodo": str(data.iloc[t+h]["periodo"]), "Horizonte": h+1,
               "Real": float(real[i, h]), "LSTM": float(pred[i, h])}
              for i, t in enumerate(test_origins) for h in range(config.horizon)]
    labels = future_periods if future_periods is not None else [f"Período +{h}" for h in range(1, config.horizon+1)]
    if len(labels) != config.horizon:
        raise ValueError("Cantidad de etiquetas futuras incompatible con el horizonte.")
    notes = []
    if len(folds) == 1:
        notes.append("El histórico solo permite un corte de validación con esta ventana y horizonte.")
    if np.mean(y == 0) >= .3:
        notes.append("Demanda con muchos ceros: revise los errores de los picos y no interprete MAPE como exactitud.")
    if np.std(y) < 1e-8:
        notes.append("Serie constante: R² y métricas porcentuales con denominador cero pueden no estar definidos.")
    if len(test_origins) < 10:
        notes.append("Hay menos de diez orígenes de prueba; la estimación del error es limitada.")
    if np.isfinite(score["R2"]) and score["R2"] <= 0:
        notes.append("R² no positivo en prueba: este ajuste no ha demostrado explicar la variación de la demanda.")
    profile = pd.DataFrame([{"Bloque": name, "Periodos": len(values), "Promedio": float(values.mean()),
                             "Desviacion": float(values.std()), "Minimo": float(values.min()), "Maximo": float(values.max())}
                            for name, values in (("Entrenamiento", y[:train_end]), ("Validación", y[train_end:val_end]), ("Prueba", y[val_end:]))])
    shift = float(y[val_end:].mean()-y[:train_end].mean())
    if abs(shift) > max(.5*float(y[:train_end].std()), .2*abs(float(y[:train_end].mean())), 1e-8):
        notes.append(f"Cambio de nivel: promedio de entrenamiento {y[:train_end].mean():.2f}, prueba {y[val_end:].mean():.2f}. El comportamiento reciente difiere del inicial.")
    tf.keras.backend.clear_session()
    return {"engine_version": ENGINE_VERSION, "config": asdict(config), "selected": selected, "search": pd.DataFrame(rows),
            "validation_folds": pd.DataFrame(fold_rows), "data_profile": profile,
            "metrics": pd.DataFrame([{"Modelo": "LSTM", **score}]),
            "horizon_metrics": pd.DataFrame([{"Horizonte": h+1, **metrics(real[:, h], pred[:, h])} for h in range(config.horizon)]),
            "detail": pd.DataFrame(detail), "forecast": pd.DataFrame({"Periodo": labels, "Horizonte": np.arange(1, config.horizon+1), "LSTM": future}),
            "history": selected_history, "notes": notes, "train_end": train_end, "val_end": val_end,
            "test_origins": len(test_origins), "train_windows": train_end-selected["lookback"]-config.horizon+1,
            "validation_origins": val_end-train_end-config.horizon+1,
            "data_hash": sha256(data.to_csv(index=False).encode()).hexdigest()}
