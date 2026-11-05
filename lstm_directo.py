"""LSTM directa multi-horizonte para demanda semanal de jeans.

Cada predicción emite simultáneamente los siguientes 12 períodos. Las ventanas
de entrenamiento tienen todos sus objetivos dentro del bloque TRAIN; la
selección usa exclusivamente orígenes de VALIDACIÓN. TEST nunca participa
en escalado, entrenamiento, selección ni calibración.
"""
from dataclasses import replace
import numpy as np
import pandas as pd

from motor_lstm import Settings, metrics
from protocolo_final import Candidate, compare, choose_lstm, origins

HORIZON = 12
LOOKBACKS = (8, 12, 16)
EPOCHS = (10, 30, 60, 100)
LONG_LOOKBACKS = (26, 52)
LONG_EPOCHS = (30, 60)


def _scale(y):
    log = np.log1p(np.asarray(y, dtype=float))
    mean = float(log.mean())
    std = float(log.std()) or 1.0
    return mean, std


def _windows(z, end, lookback, horizon):
    """Ventanas supervisadas cuyos objetivos permanecen antes de end."""
    end = int(end)
    starts = np.arange(int(lookback), end-int(horizon)+1, dtype=int)
    if len(starts) == 0:
        raise ValueError("No existen ventanas directas completas en TRAIN.")
    x = np.stack([z[i-lookback:i] for i in starts]).astype("float32")[..., None]
    y = np.stack([z[i:i+horizon] for i in starts]).astype("float32")
    return x, y


def train_direct(y, settings, lookback, epochs, horizon=HORIZON):
    """Entrena solo con y recibido; el llamador controla el corte temporal."""
    import tensorflow as tf
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) < lookback+horizon+10 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Historial insuficiente o inválido para LSTM directa.")
    tf.keras.utils.set_random_seed(settings.seed)
    tf.config.experimental.enable_op_determinism()
    mean, std = _scale(y)
    z = (np.log1p(y)-mean)/std
    x, target = _windows(z, len(y), lookback, horizon)
    inputs = tf.keras.Input(shape=(lookback, 1))
    h = tf.keras.layers.LSTM(settings.units, name="memoria_lstm_directa")(inputs)
    h = tf.keras.layers.Dropout(settings.dropout)(h)
    h = tf.keras.layers.Dense(max(8, settings.units//2), activation="relu")(h)
    outputs = tf.keras.layers.Dense(horizon, name="salida_12_semanas")(h)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001, clipnorm=1.0),
                  loss=tf.keras.losses.Huber())
    ds = tf.data.Dataset.from_tensor_slices((x, target)).batch(settings.batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    model.fit(ds.with_options(options), epochs=int(epochs), verbose=0, shuffle=False)
    return Candidate("direct", model, mean, std, lookback, int(epochs),
                     settings=settings, strategy="direct", output_horizon=horizon)


def select_direct(series, settings=None, control=None, progress=None, horizon=HORIZON, calibrate_bias=False):
    """Compara LSTM directas y recursivas sobre idénticos orígenes de VALIDACIÓN."""
    settings = settings or Settings()
    if int(horizon) != HORIZON:
        raise ValueError("La versión experimental se valida con horizonte fijo de 12 semanas.")
    y = series.valor.to_numpy(dtype=float)
    tr = int(len(y)*settings.train_ratio)
    ve = int(len(y)*(settings.train_ratio+settings.val_ratio))
    origins(tr, ve, horizon)
    candidates = dict(control["candidates"]) if control else {}
    tables, details = [], []
    if candidates:
        table, detail = compare(candidates, y, tr, ve, y[:tr], horizon)
        tables.append(table); details.append(detail)
    for lookback in LOOKBACKS:
        config = replace(settings, lookback=lookback)
        for epoch in EPOCHS:
            name = f"H12 LSTM directa log1p w{lookback} u{config.units} e{epoch}"
            if progress:
                progress(name)
            # Only TRAIN values are passed to the fitting/scaling function.
            candidate = train_direct(y[:tr], config, lookback, epoch, horizon)
            candidate_name = name
            candidates[candidate_name] = candidate
            table, detail = compare({candidate_name: candidate}, y, tr, ve, y[:tr], horizon)
            tables.append(table); details.append(detail)
    # Ventanas semestral y anual para capturar patrones más largos.
    for lookback in LONG_LOOKBACKS:
        config = replace(settings, lookback=lookback)
        for epoch in LONG_EPOCHS:
            name = f"H12 LSTM directa log1p w{lookback} u{config.units} e{epoch}"
            if progress:
                progress(name)
            candidate = train_direct(y[:tr], config, lookback, epoch, horizon)
            candidates[name] = candidate
            table, detail = compare({name: candidate}, y, tr, ve, y[:tr], horizon)
            tables.append(table); details.append(detail)
    table = pd.concat(tables, ignore_index=True).sort_values(
        ["RMSE", "MAE", "Modelo"], kind="stable").reset_index(drop=True)
    selected = choose_lstm(table, candidates)
    bias_correction = 0.0
    if calibrate_bias:
        selected_rows = pd.concat(details, ignore_index=True)
        selected_rows = selected_rows.loc[selected_rows["Modelo"] == selected]
        residual = (selected_rows["real"]-selected_rows["prediccion"]).to_numpy(dtype=float)
        if not len(residual) or not np.isfinite(residual).all():
            raise ValueError("No hay residuos finitos de VALIDACIÓN para calibrar el sesgo.")
        bias_correction = float(candidates[selected].bias_correction) + float(np.mean(residual))
        candidates[selected].bias_correction = bias_correction
    kept = {k:v for k,v in candidates.items()
            if k == selected or not (k.startswith("H12 ") and k != selected)}
    return {
        "selected": selected,
        "best_comparison": str(table.iloc[0].Modelo),
        "selection_policy": "LSTM recursiva/directa con ventanas 8, 12, 16, 26 y 52; RMSE de VALIDACIÓN a 12 semanas",
        "candidates": kept,
        "validation": table,
        "validation_predictions": pd.concat(details, ignore_index=True),
        "train_end": tr,
        "validation_end": ve,
        "series": series.copy(),
        "settings": settings,
        "test": None,
        "horizon_search": True,
        "direct_horizon": horizon,
        "bias_correction": bias_correction,
    }


def refit_direct_forecast(selection, first_plan, weeks=12):
    """Reajusta el ganador directo con toda la historia y épocas congeladas."""
    from lstm_validacion_horizonte import train_fixed
    from protocolo_final import recursive
    import pandas as pd
    s = selection["series"]
    first_plan = pd.Timestamp(first_plan)
    next_date = pd.Timestamp(s.fecha_inicio.iloc[-1])+pd.Timedelta(days=7)
    days = (first_plan-next_date).days
    if first_plan.weekday() != 0 or days < 0 or days % 7 or days > 112 or not 1 <= int(weeks) <= 26:
        raise ValueError("Fecha u horizonte inválido.")
    chosen = selection["candidates"][selection["selected"]]
    if chosen.strategy != "direct":
        raise ValueError("El candidato seleccionado no es LSTM directa.")
    y = s.valor.to_numpy(dtype=float)
    # Refit with the same architecture, window, horizon and epochs, now using all
    # observed history. No TEST metric or residual is used to alter the model.
    model = train_direct(y, chosen.settings, chosen.lookback, chosen.epochs, chosen.output_horizon)
    pred = recursive(model, y, int(weeks)+days//7)[days//7:]
    return pd.DataFrame({"fecha_inicio":pd.date_range(first_plan,periods=int(weeks),freq="7D"),
                         "pronostico":pred})
