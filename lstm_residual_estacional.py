"""LSTM residual estacional para pronóstico semanal de demanda a 12 semanas.

Modelo en log1p: aprende la desviación respecto al valor de hace 52 semanas.
Cada origen se ajusta solo con el prefijo disponible. Tres semillas se promedian.
La selección se congela antes de evaluar TEST. No modifica la base histórica.
"""
from dataclasses import dataclass
from datetime import timedelta
import numpy as np
import pandas as pd


@dataclass
class ResidualCandidate:
    model: object
    lookback: int
    period: int
    units: int
    epochs: int
    seeds: tuple
    kind: str = "LSTM residual estacional log1p"
    smear: float = 1.0


def _train_one(y, origin, lookback, units, epochs, seed, progress=None):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    tf.config.experimental.enable_op_determinism()
    y = np.asarray(y, dtype=float)[:int(origin)]
    if len(y) < 52 + lookback + 12:
        raise ValueError("Historial insuficiente para LSTM residual anual.")
    z = np.log1p(y)
    mu = float(z.mean())
    sd = float(z.std()) or 1.0
    zs = (z - mu) / sd
    residual = zs[52:] - zs[:-52]
    rmu = float(residual.mean())
    rsd = float(residual.std()) or 1.0
    starts = np.arange(52 + lookback, len(y), dtype=int)
    if len(starts) < 12:
        raise ValueError("Menos de 12 ventanas estacionales para entrenar.")
    xs, targets = [], []
    for t in starts:
        recent = zs[t-lookback:t]
        seasonal = zs[t-lookback-52:t-52]
        xs.append(np.column_stack([recent, seasonal, recent-seasonal]))
        targets.append((zs[t] - zs[t-52] - rmu) / rsd)
    x = np.asarray(xs, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    inputs = tf.keras.Input(shape=(lookback, 3))
    h = tf.keras.layers.LSTM(int(units), kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    h = tf.keras.layers.Dropout(0.10)(h)
    outputs = tf.keras.layers.Dense(1)(h)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                  loss=tf.keras.losses.Huber(delta=1.0))
    model.fit(x, target, epochs=int(epochs), batch_size=min(16, len(x)),
              shuffle=False, verbose=0,
              callbacks=[tf.keras.callbacks.TerminateOnNaN()])
    return {"model": model, "mu": mu, "sd": sd, "rmu": rmu, "rsd": rsd,
            "lookback": int(lookback), "seed": int(seed)}


def _predict_member(member, history, steps):
    history = list(map(float, np.asarray(history, dtype=float)))
    out = []
    for h in range(int(steps)):
        t = len(history)
        if t < 52 + member["lookback"]:
            raise ValueError("Historial insuficiente para predecir.")
        recent = (np.log1p(np.asarray(history[t-member["lookback"]:t])) - member["mu"]) / member["sd"]
        seasonal = (np.log1p(np.asarray(history[t-member["lookback"]-52:t-52])) - member["mu"]) / member["sd"]
        x = np.column_stack([recent, seasonal, recent-seasonal]).astype("float32")[None, :, :]
        raw = float(member["model"](x, training=False).numpy()[0, 0])
        residual = raw * member["rsd"] + member["rmu"]
        annual = (np.log1p(history[t-52]) - member["mu"]) / member["sd"]
        log_value = (annual + residual) * member["sd"] + member["mu"]
        value = max(0.0, float(np.expm1(np.clip(log_value, -20, 20))))
        history.append(value)
        out.append(value)
    return np.asarray(out, dtype=float)


def _fit_ensemble(y, origin, lookback, units, epochs, seeds=(42, 43, 44)):
    return [_train_one(y, origin, lookback, units, epochs, seed) for seed in seeds]


def _predict_ensemble(members, history, steps):
    return np.mean([_predict_member(m, history, steps) for m in members], axis=0)


def _metrics(real, pred, train):
    from motor_lstm import metrics
    return metrics(np.asarray(real, float), np.asarray(pred, float), np.asarray(train, float))


def _origin_forecast(y, origin, lookback, units, epochs, seeds=(42, 43, 44), steps=12):
    members = _fit_ensemble(y, origin, lookback, units, epochs, seeds)
    return _predict_ensemble(members, y[:origin], steps)


def select_seasonal_residual(series, settings=None, control=None, progress=None, calibrate_bias=False):
    """Selecciona configuración predefinida en VALIDACIÓN; TEST no se consulta."""
    y = series.valor.to_numpy(dtype=float)
    tr = int(len(y) * (settings.train_ratio if settings else 0.70))
    ve = int(len(y) * ((settings.train_ratio + settings.val_ratio) if settings else 0.85))
    if tr < 52 + 12 + 12 or ve-tr < 12 or len(y)-ve < 12:
        raise ValueError("La serie no permite reservar bloques temporales completos.")
    lookback = 8 if len(y) >= 180 else 12
    if len(y) < 180:
        lookback = 12
    units, epochs, seeds = 8, 50, (42, 43, 44)
    origins = list(range(tr, ve-12+1))
    actual, pred, details = [], [], []
    for j, origin in enumerate(origins, 1):
        if progress:
            progress(f"LSTM residual estacional: origen de validación {j}/{len(origins)}")
        estimate = _origin_forecast(y, origin, lookback, units, epochs, seeds, 12)
        actual.extend(y[origin:origin+12])
        pred.extend(estimate)
        for h, (real, forecast) in enumerate(zip(y[origin:origin+12], estimate), 1):
            details.append({"Modelo": "LSTM residual estacional 52", "origen": origin,
                            "h": h, "real": float(real), "prediccion": float(forecast)})
    # Common seasonal benchmarks, same rolling origins and same 12-week horizon.
    rows = [{"Modelo": "LSTM residual estacional 52", "Orígenes": len(origins),
             "Horizonte": 12, **_metrics(actual, pred, y[:tr])}]
    for period in (48, 52):
        if tr < period:
            continue
        baseline = np.concatenate([y[o-period:o-period+12] for o in origins])
        rows.append({"Modelo": f"Naive estacional {period}", "Orígenes": len(origins),
                     "Horizonte": 12, **_metrics(actual, baseline, y[:tr])})
    table = pd.DataFrame(rows).sort_values(["RMSE", "MAE", "Modelo"]).reset_index(drop=True)
    selected = "LSTM residual estacional 52"
    candidate = ResidualCandidate(None, lookback, 52, units, epochs, seeds)
    return {"selected": selected, "best_comparison": str(table.iloc[0]["Modelo"]),
            "selection_policy": "LSTM residual estacional predefinida; comparación rolling-origin en VALIDACIÓN",
            "candidates": {selected: candidate}, "validation": table,
            "validation_predictions": pd.DataFrame(details), "train_end": tr,
            "validation_end": ve, "series": series.copy(), "settings": settings,
            "test": None, "fixed_test": pd.DataFrame(), "fixed_test_origin_date": None,
            "bias_correction": 0.0, "horizon_search": True}


def evaluate_seasonal_residual(selection):
    if selection["test"] is not None:
        return selection["test"].copy()
    y = selection["series"].valor.to_numpy(dtype=float)
    tr, ve = selection["train_end"], selection["validation_end"]
    c = selection["candidates"][selection["selected"]]
    origins = list(range(ve, len(y)-12+1))
    actual, pred, details = [], [], []
    for origin in origins:
        estimate = _origin_forecast(y, origin, c.lookback, c.units, c.epochs, c.seeds, 12)
        actual.extend(y[origin:origin+12]); pred.extend(estimate)
        for h, (real, forecast) in enumerate(zip(y[origin:origin+12], estimate), 1):
            details.append({"Modelo": selection["selected"], "origen": origin, "h": h,
                            "real": float(real), "prediccion": float(forecast)})
    rows = [{"Modelo": selection["selected"], "Orígenes": len(origins), "Horizonte": 12,
             **_metrics(actual, pred, y[:tr])}]
    for period in (48, 52):
        baseline = np.concatenate([y[o-period:o-period+12] for o in origins])
        rows.append({"Modelo": f"Naive estacional {period}", "Orígenes": len(origins),
                     "Horizonte": 12, **_metrics(actual, baseline, y[:tr])})
    selection["test"] = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    selection["test_predictions"] = pd.DataFrame(details)
    if len(y)-ve >= 12:
        origin = ve
        estimate = _origin_forecast(y, origin, c.lookback, c.units, c.epochs, c.seeds, 12)
        selection["fixed_test"] = pd.DataFrame([{
            "Modelo": selection["selected"], "Orígenes": 1, "Horizonte": 12,
            **_metrics(y[origin:origin+12], estimate, y[:tr])}])
        selection["fixed_test_origin_date"] = pd.Timestamp(selection["series"].fecha_inicio.iloc[origin]).date()
    return selection["test"].copy()


def future_seasonal_residual(selection, first_plan, weeks=12, refit=True):
    if selection["test"] is None:
        raise ValueError("Evalúe la prueba antes de generar el pronóstico.")
    s = selection["series"]
    first_plan = pd.Timestamp(first_plan)
    next_date = pd.Timestamp(s.fecha_inicio.iloc[-1]) + pd.Timedelta(days=7)
    days = (first_plan-next_date).days
    if first_plan.weekday() != 0 or days < 0 or days % 7 or days > 112 or not 1 <= int(weeks) <= 26:
        raise ValueError("Fecha u horizonte inválido.")
    c = selection["candidates"][selection["selected"]]
    y = s.valor.to_numpy(dtype=float)
    bridge = days // 7
    # Train once with all observations after the frozen retrospective evaluation.
    members = _fit_ensemble(y, len(y), c.lookback, c.units, c.epochs, c.seeds)
    estimate = _predict_ensemble(members, y, bridge + int(weeks))
    return pd.DataFrame({"fecha_inicio": pd.date_range(first_plan, periods=int(weeks), freq="7D"),
                         "pronostico": estimate[bridge:]})
