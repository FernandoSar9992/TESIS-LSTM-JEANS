"""Motor V9: LSTM directa multihorizonte con contexto temporal y residuo estacional.

- Pronostica directamente los 12 horizontes; no propaga errores por recursión semanal.
- Usa una referencia estacional de 52 semanas y características de nivel, tendencia y calendario.
- Escalamiento y entrenamiento respetan el orden temporal.
- Early stopping usa un bloque interno anterior a VALIDACIÓN.
- La arquitectura se selecciona exclusivamente con VALIDACIÓN.
- La corrección aditiva se estima en VALIDACIÓN y se congela antes de TEST.
- TEST se informa en dos modalidades: un pronóstico fijo de 12 semanas y rolling-origin
  con pesos congelados y actualización del contexto con observaciones ya disponibles.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

HORIZON = 12
PERIOD = 52
SEQUENCE_FEATURES = 6
CONTEXT_FEATURES = 15
MODEL_VERSION = "v9-lstm-directa-contexto"


@dataclass
class ResidualCandidate:
    model: object
    lookback: int
    period: int
    units: int
    epochs: int
    seeds: tuple
    kind: str = "LSTM directa residual estacional con contexto"
    smear: float = 1.0
    bias_correction: float = 0.0
    dropout: float = 0.15
    l2_value: float = 1e-4
    member_epochs: tuple = ()
    horizon: int = HORIZON


def _scale_from(y, fit_end):
    values = np.asarray(y, dtype=float)[:int(fit_end)]
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("La serie debe contener valores finitos y no negativos.")
    log_values = np.log1p(values)
    center = float(np.mean(log_values))
    scale = float(np.std(log_values))
    return center, scale if scale > 1e-8 else 1.0


def _build_features(y, origin, step, lookback, center, scale,
                    period=PERIOD, horizon=HORIZON):
    """Construye entradas usando exclusivamente observaciones anteriores al origen."""
    values = np.asarray(y, dtype=float)
    origin, step = int(origin), int(step)
    lookback, period, horizon = int(lookback), int(period), int(horizon)
    if origin < lookback or origin > len(values):
        raise ValueError("Origen inválido o historial menor que la ventana.")
    if not 1 <= step <= horizon:
        raise ValueError("El horizonte solicitado está fuera del rango entrenado.")
    history = values[:origin]
    if not np.isfinite(history).all() or np.any(history < 0):
        raise ValueError("El historial contiene valores no válidos.")

    log_history = np.log1p(history)
    z_history = (log_history - center) / scale
    indices = np.arange(origin - lookback, origin, dtype=int)
    recent = z_history[indices]
    differences = np.r_[0.0, np.diff(recent)]

    annual_residual = np.zeros(lookback, dtype=float)
    annual_lag = np.zeros(lookback, dtype=float)
    available_indices = indices >= period
    if np.any(available_indices):
        annual_residual[available_indices] = (
            z_history[indices[available_indices]]
            - z_history[indices[available_indices] - period]
        )
        annual_lag[available_indices] = z_history[indices[available_indices] - period]

    phase = 2.0 * np.pi * (indices % period) / period
    sequence = np.column_stack([
        recent,
        differences,
        annual_residual,
        annual_lag,
        np.sin(phase),
        np.cos(phase),
    ]).astype(np.float32)

    recent4 = z_history[max(0, origin - 4):origin]
    recent8 = z_history[max(0, origin - 8):origin]
    recent24 = z_history[max(0, origin - 24):origin]
    target = origin + step - 1
    target_phase = 2.0 * np.pi * (target % period) / period

    seasonal_index = target - period
    annual_available = seasonal_index >= 0 and seasonal_index < origin
    if annual_available:
        base_log = float(log_history[seasonal_index])
    else:
        base_log = float(np.mean(log_history[max(0, origin - 8):origin]))
    base_z = (base_log - center) / scale

    recent_annual = annual_residual[-min(8, lookback):]
    context = np.asarray([
        base_z,
        float(np.mean(recent4)),
        float(np.mean(recent8)),
        float(np.mean(recent24)),
        float(recent[-1]),
        float((recent8[-1] - recent8[0]) / max(1, len(recent8) - 1)),
        float(np.std(recent8)),
        float(np.mean(recent_annual)),
        float(recent_annual[-1]),
        float(np.sin(target_phase)),
        float(np.cos(target_phase)),
        float(np.sin(2.0 * target_phase)),
        float(np.cos(2.0 * target_phase)),
        float(step / horizon),
        float(annual_available),
    ], dtype=np.float32)

    return sequence, context, base_log


def _training_pairs(y, fit_end, lookback, center, scale,
                    start_origin=None, period=PERIOD, horizon=HORIZON):
    """Crea pares directos; todos los objetivos quedan estrictamente antes de fit_end."""
    values = np.asarray(y, dtype=float)
    fit_end = int(fit_end)
    first_origin = max(int(lookback), int(start_origin or lookback))
    last_origin = fit_end - int(horizon)
    if last_origin < first_origin:
        raise ValueError("No hay suficientes orígenes completos para entrenar la LSTM.")

    sequences, contexts, bases, targets, metadata = [], [], [], [], []
    for origin in range(first_origin, last_origin + 1):
        for step in range(1, int(horizon) + 1):
            target_index = origin + step - 1
            sequence, context, base_log = _build_features(
                values, origin, step, lookback, center, scale, period, horizon
            )
            target_log = float(np.log1p(values[target_index]))
            sequences.append(sequence)
            contexts.append(context)
            bases.append(base_log)
            targets.append((target_log - base_log) / scale)
            metadata.append((origin, step, target_index, target_log))

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(contexts, dtype=np.float32),
        np.asarray(bases, dtype=float),
        np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        metadata,
    )


def _build_model(lookback, units, dropout, l2_value):
    import keras

    keras.backend.clear_session()
    sequence_input = keras.Input(
        shape=(int(lookback), SEQUENCE_FEATURES), name="secuencia_temporal"
    )
    context_input = keras.Input(
        shape=(CONTEXT_FEATURES,), name="contexto_horizonte"
    )
    recurrent = keras.layers.LSTM(
        int(units),
        kernel_regularizer=keras.regularizers.l2(float(l2_value)),
        recurrent_regularizer=keras.regularizers.l2(float(l2_value)),
    )(sequence_input)
    combined = keras.layers.Concatenate()([recurrent, context_input])
    hidden = keras.layers.Dense(
        8, activation="tanh",
        kernel_regularizer=keras.regularizers.l2(float(l2_value)),
    )(combined)
    hidden = keras.layers.Dropout(float(dropout))(hidden)
    output = keras.layers.Dense(
        1, kernel_initializer="zeros", bias_initializer="zeros",
        name="residuo_log",
    )(hidden)
    model = keras.Model([sequence_input, context_input], output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=keras.losses.Huber(),
    )
    return model


def _fit_member(y, fit_end, lookback, units, seed, max_epochs, patience,
                dropout, l2_value, batch_size, progress=None):
    import keras

    values = np.asarray(y, dtype=float)
    fit_end = int(fit_end)
    inner_width = max(HORIZON + 1, int(math.ceil(fit_end * 0.20)))
    inner_end = fit_end - inner_width
    if inner_end <= int(lookback) + HORIZON + 8:
        raise ValueError("Historial insuficiente para la validación interna de la LSTM.")

    # El escalamiento del piloto se ajusta solo con el bloque interno de entrenamiento.
    pilot_center, pilot_scale = _scale_from(values, inner_end)
    x_train, c_train, _, y_train, _ = _training_pairs(
        values, inner_end, lookback, pilot_center, pilot_scale
    )
    x_valid, c_valid, b_valid, y_valid, valid_meta = _training_pairs(
        values, fit_end, lookback, pilot_center, pilot_scale,
        start_origin=inner_end,
    )
    if len(x_train) < 30 or len(x_valid) < HORIZON:
        raise ValueError("No hay suficientes ventanas para entrenar y validar internamente.")

    keras.utils.set_random_seed(int(seed))
    pilot = _build_model(lookback, units, dropout, l2_value)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        min_delta=1e-5,
        restore_best_weights=True,
    )
    history = pilot.fit(
        [x_train, c_train],
        y_train,
        validation_data=([x_valid, c_valid], y_valid),
        epochs=int(max_epochs),
        batch_size=min(int(batch_size), len(x_train)),
        shuffle=False,
        verbose=0,
        callbacks=[early_stop, keras.callbacks.TerminateOnNaN()],
    )
    val_losses = np.asarray(history.history.get("val_loss", []), dtype=float)
    if len(val_losses) == 0 or not np.isfinite(val_losses).any():
        raise ValueError("La LSTM no produjo una pérdida de validación finita.")
    best_epoch = int(np.nanargmin(val_losses) + 1)

    # Corrección de retrans formación de log1p, calculada dentro del bloque TRAIN.
    pilot_residual = pilot.predict([x_valid, c_valid], verbose=0).reshape(-1)
    valid_target_log = b_valid + y_valid.reshape(-1) * pilot_scale
    valid_pred_log = b_valid + pilot_residual * pilot_scale
    smear = float(np.mean(np.exp(np.clip(valid_target_log - valid_pred_log, -3.0, 3.0))))
    if not np.isfinite(smear):
        smear = 1.0
    smear = float(np.clip(smear, 0.85, 1.25))

    # Reajuste sobre todo TRAIN con el número de épocas elegido internamente.
    center, scale = _scale_from(values, fit_end)
    x_all, c_all, _, y_all, _ = _training_pairs(
        values, fit_end, lookback, center, scale
    )
    keras.utils.set_random_seed(int(seed))
    final_model = _build_model(lookback, units, dropout, l2_value)
    final_model.fit(
        [x_all, c_all],
        y_all,
        epochs=best_epoch,
        batch_size=min(int(batch_size), len(x_all)),
        shuffle=False,
        verbose=0,
    )
    return {
        "model": final_model,
        "center": center,
        "scale": scale,
        "lookback": int(lookback),
        "units": int(units),
        "epochs": best_epoch,
        "seed": int(seed),
        "smear": smear,
    }


def _fit_fixed_member(y, fit_end, lookback, units, seed, epochs,
                      smear=1.0, dropout=0.15, l2_value=1e-4, batch_size=16):
    import keras

    values = np.asarray(y, dtype=float)
    center, scale = _scale_from(values, fit_end)
    x_all, c_all, _, y_all, _ = _training_pairs(
        values, fit_end, lookback, center, scale
    )
    keras.utils.set_random_seed(int(seed))
    model = _build_model(lookback, units, dropout, l2_value)
    model.fit(
        [x_all, c_all],
        y_all,
        epochs=max(1, int(epochs)),
        batch_size=min(int(batch_size), len(x_all)),
        shuffle=False,
        verbose=0,
    )
    return {
        "model": model,
        "center": center,
        "scale": scale,
        "lookback": int(lookback),
        "units": int(units),
        "epochs": max(1, int(epochs)),
        "seed": int(seed),
        "smear": float(smear),
    }


def _fit_ensemble(y, fit_end, lookback, units, seeds, max_epochs, patience,
                  dropout, l2_value, batch_size, progress=None, label="LSTM"):
    members = []
    for index, seed in enumerate(seeds, start=1):
        if progress:
            progress(f"{label}: semilla {index}/{len(seeds)} ({seed})")
        members.append(_fit_member(
            y, fit_end, lookback, units, seed, max_epochs, patience,
            dropout, l2_value, batch_size,
        ))
    return members


def _fit_fixed_ensemble(y, fit_end, candidate, batch_size=16):
    members = []
    epochs_by_seed = candidate.member_epochs or tuple(
        candidate.epochs for _ in candidate.seeds
    )
    smears = [
        member.get("smear", candidate.smear)
        for member in (candidate.model or [])
    ]
    for index, seed in enumerate(candidate.seeds):
        epochs = epochs_by_seed[index] if index < len(epochs_by_seed) else candidate.epochs
        smear = smears[index] if index < len(smears) else candidate.smear
        members.append(_fit_fixed_member(
            y, fit_end, candidate.lookback, candidate.units, seed, epochs,
            smear=smear, dropout=candidate.dropout,
            l2_value=candidate.l2_value, batch_size=batch_size,
        ))
    return members


def _predict_members(members, y, origins, steps, bias=0.0):
    values = np.asarray(y, dtype=float)
    origins = np.asarray(origins, dtype=int).reshape(-1)
    steps = np.asarray(steps, dtype=int).reshape(-1)
    if len(origins) != len(steps):
        raise ValueError("Orígenes y horizontes deben tener la misma longitud.")
    predictions = []
    for member in members:
        sequences, contexts, bases = [], [], []
        for origin, step in zip(origins, steps):
            sequence, context, base_log = _build_features(
                values, origin, step, member["lookback"],
                member["center"], member["scale"],
            )
            sequences.append(sequence)
            contexts.append(context)
            bases.append(base_log)
        raw_residual = member["model"].predict(
            [np.asarray(sequences, dtype=np.float32),
             np.asarray(contexts, dtype=np.float32)],
            verbose=0,
        ).reshape(-1)
        predicted_log = np.asarray(bases, dtype=float) + raw_residual * member["scale"]
        predicted = np.exp(np.clip(predicted_log, -20.0, 20.0)) * member["smear"] - 1.0
        predictions.append(np.maximum(0.0, predicted))
    if not predictions:
        raise ValueError("No hay miembros entrenados para generar el pronóstico.")
    return np.maximum(0.0, np.mean(predictions, axis=0) + float(bias))


def _baseline_predictions(y, origins, steps, period):
    values = np.asarray(y, dtype=float)
    out = []
    for origin, step in zip(np.asarray(origins, int), np.asarray(steps, int)):
        target = int(origin + step - 1)
        ref = target - int(period)
        if 0 <= ref < int(origin):
            out.append(float(values[ref]))
        else:
            out.append(float(np.mean(values[max(0, int(origin) - 8):int(origin)])))
    return np.asarray(out, dtype=float)


def _metrics(real, pred, train):
    from motor_lstm import metrics
    return metrics(
        np.asarray(real, dtype=float),
        np.asarray(pred, dtype=float),
        np.asarray(train, dtype=float),
    )


def _candidate_grid(settings):
    base_lookback = int(getattr(settings, "lookback", 12) if settings else 12)
    base_units = int(getattr(settings, "units", 16) if settings else 16)
    base_lookback = max(4, min(16, base_lookback))
    base_units = max(8, min(32, base_units))
    short_lookback = max(4, min(8, base_lookback // 3))
    long_lookback = max(8, min(12, (2 * base_lookback) // 3))
    if long_lookback <= short_lookback:
        long_lookback = min(12, short_lookback + 4)
    small_units = max(4, min(8, base_units // 4))
    large_units = max(small_units, min(8, base_units // 2))
    return list(dict.fromkeys([
        (short_lookback, small_units),
        (long_lookback, large_units),
    ]))


def select_seasonal_residual(series, settings=None, control=None,
                             progress=None, calibrate_bias=True):
    """Selecciona la arquitectura LSTM usando VALIDACIÓN; TEST permanece intacto."""
    y = series.valor.to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("La serie de demanda contiene valores no finitos o negativos.")

    train_ratio = float(getattr(settings, "train_ratio", 0.70) if settings else 0.70)
    val_ratio = float(getattr(settings, "val_ratio", 0.15) if settings else 0.15)
    tr = int(len(y) * train_ratio)
    ve = int(len(y) * (train_ratio + val_ratio))
    if tr < 52 + HORIZON + 20 or ve - tr < HORIZON or len(y) - ve < HORIZON:
        raise ValueError("La serie no permite reservar bloques temporales completos.")

    max_epochs = int(getattr(settings, "epochs", 100) if settings else 100)
    patience = int(getattr(settings, "patience", 12) if settings else 12)
    batch_size = int(getattr(settings, "batch_size", 16) if settings else 16)
    dropout = float(getattr(settings, "dropout", 0.15) if settings else 0.15)
    seed = int(getattr(settings, "seed", 42) if settings else 42)
    seeds = (seed, seed + 1, seed + 2)
    grid = _candidate_grid(settings)

    validation_origins = np.arange(tr, ve - HORIZON + 1, dtype=int)
    origins = np.repeat(validation_origins, HORIZON)
    steps = np.tile(np.arange(1, HORIZON + 1, dtype=int), len(validation_origins))
    target_indices = origins + steps - 1
    actual = y[target_indices]

    rows = []
    best_candidate = None
    best_name = None
    best_raw = None
    best_score = (float("inf"), float("inf"))
    validation_details = []

    for config_index, (lookback, units) in enumerate(grid, start=1):
        if progress:
            progress(
                f"Evaluando arquitectura {config_index}/{len(grid)}: "
                f"ventana {lookback}, unidades {units}"
            )
        members = _fit_ensemble(
            y, tr, lookback, units, seeds, max_epochs, patience,
            dropout, 1e-4, batch_size, progress=progress,
            label=f"LSTM w{lookback} u{units}",
        )
        raw = _predict_members(members, y, origins, steps, bias=0.0)
        score = _metrics(actual, raw, y[:tr])
        member_epochs = tuple(int(m["epochs"]) for m in members)
        selected_epochs = int(round(float(np.median(member_epochs))))
        name = f"LSTM directa residual 52 w{lookback} u{units} e{selected_epochs}"
        rows.append({
            "Modelo": name,
            "Orígenes": len(validation_origins),
            "Horizonte": HORIZON,
            **score,
        })
        for origin, step, target_index, real, forecast in zip(
            origins, steps, target_indices, actual, raw
        ):
            validation_details.append({
                "Modelo": name,
                "origen": int(origin),
                "h": int(step),
                "indice_objetivo": int(target_index),
                "real": float(real),
                "prediccion": float(forecast),
            })

        candidate = ResidualCandidate(
            model=members,
            lookback=int(lookback),
            period=PERIOD,
            units=int(units),
            epochs=selected_epochs,
            seeds=seeds,
            kind="LSTM directa multihorizonte con residuo estacional y contexto",
            smear=float(np.mean([m["smear"] for m in members])),
            dropout=dropout,
            l2_value=1e-4,
            member_epochs=member_epochs,
            horizon=HORIZON,
        )
        candidate_score = (float(score["RMSE"]), float(score["MAE"]))
        if candidate_score < best_score:
            best_score = candidate_score
            best_candidate = candidate
            best_name = name
            best_raw = raw.copy()

    # Referencias estacionales calculadas en los mismos objetivos de VALIDACIÓN.
    for period in (48, 52):
        baseline = _baseline_predictions(y, origins, steps, period)
        rows.append({
            "Modelo": f"Naive estacional {period}",
            "Orígenes": len(validation_origins),
            "Horizonte": HORIZON,
            **_metrics(actual, baseline, y[:tr]),
        })

    if best_candidate is None or best_raw is None:
        raise ValueError("No se pudo seleccionar una arquitectura LSTM.")

    bias = float(np.mean(actual - best_raw)) if calibrate_bias else 0.0
    if not np.isfinite(bias):
        bias = 0.0
    # Evita una corrección extrema que exceda el 30 % del nivel medio de entrenamiento.
    max_bias = 0.30 * max(1.0, float(np.mean(y[:tr])))
    bias = float(np.clip(bias, -max_bias, max_bias))
    best_candidate.bias_correction = bias

    table = pd.DataFrame(rows).sort_values(
        ["RMSE", "MAE", "Modelo"]
    ).reset_index(drop=True)
    best_comparison = str(table.iloc[0]["Modelo"])
    calibrated_validation = np.maximum(0.0, best_raw + bias)
    validation_calibrated = pd.DataFrame([{
        "Modelo": best_name,
        "Orígenes": len(validation_origins),
        "Horizonte": HORIZON,
        **_metrics(actual, calibrated_validation, y[:tr]),
    }])

    selected_details = []
    for origin, step, target_index, real, raw, calibrated in zip(
        origins, steps, target_indices, actual, best_raw, calibrated_validation
    ):
        selected_details.append({
            "Modelo": best_name,
            "origen": int(origin),
            "h": int(step),
            "indice_objetivo": int(target_index),
            "real": float(real),
            "prediccion": float(raw),
            "prediccion_calibrada": float(calibrated),
        })

    return {
        "selected": best_name,
        "best_comparison": best_comparison,
        "selection_policy": (
            "Menor RMSE de VALIDACIÓN entre arquitecturas LSTM directas; "
            "calibración aditiva estimada después de seleccionar y congelada antes de TEST."
        ),
        "candidates": {best_name: best_candidate},
        "validation": table,
        "validation_calibrated": validation_calibrated,
        "validation_predictions": pd.DataFrame(selected_details),
        "candidate_diagnostics": pd.DataFrame(validation_details),
        "candidate_grid": grid,
        "train_end": tr,
        "validation_end": ve,
        "series": series.copy(),
        "settings": settings,
        "test": None,
        "test_predictions": pd.DataFrame(),
        "fixed_test": pd.DataFrame(),
        "fixed_test_predictions": pd.DataFrame(),
        "fixed_test_origin_date": None,
        "bias_correction": bias,
        "calibration_source": "VALIDACIÓN rolling-origin, media(real - pronóstico), congelada antes de TEST",
        "model_version": MODEL_VERSION,
        "horizon_search": True,
    }


def evaluate_seasonal_residual(selection):
    """Evalúa TEST con arquitectura y épocas congeladas; no vuelve a seleccionar."""
    if selection["test"] is not None:
        return selection["test"].copy()

    y = selection["series"].valor.to_numpy(dtype=float)
    tr, ve = int(selection["train_end"]), int(selection["validation_end"])
    candidate = selection["candidates"][selection["selected"]]
    settings = selection.get("settings")
    batch_size = int(getattr(settings, "batch_size", 16) if settings else 16)

    # Se incorporan TRAIN + VALIDACIÓN para ajustar pesos, pero no se cambian
    # arquitectura, épocas, semillas, escala de corrección ni hiperparámetros.
    members = _fit_fixed_ensemble(y, ve, candidate, batch_size=batch_size)
    test_origins = np.arange(ve, len(y) - HORIZON + 1, dtype=int)
    if len(test_origins) == 0:
        raise ValueError("TEST no contiene un horizonte completo de 12 semanas.")
    origins = np.repeat(test_origins, HORIZON)
    steps = np.tile(np.arange(1, HORIZON + 1, dtype=int), len(test_origins))
    target_indices = origins + steps - 1
    actual = y[target_indices]

    raw = _predict_members(members, y, origins, steps, bias=0.0)
    calibrated = np.maximum(0.0, raw + candidate.bias_correction)
    rows = [{
        "Modelo": selection["selected"],
        "Orígenes": len(test_origins),
        "Horizonte": HORIZON,
        **_metrics(actual, calibrated, y[:tr]),
    }]
    for period in (48, 52):
        baseline = _baseline_predictions(y, origins, steps, period)
        rows.append({
            "Modelo": f"Naive estacional {period}",
            "Orígenes": len(test_origins),
            "Horizonte": HORIZON,
            **_metrics(actual, baseline, y[:tr]),
        })
    selection["test"] = pd.DataFrame(rows).sort_values(
        ["RMSE", "MAE", "Modelo"]
    ).reset_index(drop=True)

    selection["test_predictions"] = pd.DataFrame([
        {
            "Modelo": selection["selected"],
            "origen": int(origin),
            "h": int(step),
            "real": float(real),
            "prediccion_sin_calibrar": float(raw_value),
            "prediccion": float(calibrated_value),
            "error": float(real - calibrated_value),
        }
        for origin, step, real, raw_value, calibrated_value in zip(
            origins, steps, actual, raw, calibrated
        )
    ])

    # Un único pronóstico de 12 semanas desde el primer origen de TEST.
    fixed_origin = ve
    fixed_origins = np.full(HORIZON, fixed_origin, dtype=int)
    fixed_steps = np.arange(1, HORIZON + 1, dtype=int)
    fixed_raw = _predict_members(members, y, fixed_origins, fixed_steps, bias=0.0)
    fixed_calibrated = np.maximum(0.0, fixed_raw + candidate.bias_correction)
    fixed_actual = y[fixed_origin:fixed_origin + HORIZON]
    fixed_rows = [{
        "Modelo": selection["selected"],
        "Orígenes": 1,
        "Horizonte": HORIZON,
        **_metrics(fixed_actual, fixed_calibrated, y[:tr]),
    }]
    for period in (48, 52):
        baseline = _baseline_predictions(
            y, fixed_origins, fixed_steps, period
        )
        fixed_rows.append({
            "Modelo": f"Naive estacional {period}",
            "Orígenes": 1,
            "Horizonte": HORIZON,
            **_metrics(fixed_actual, baseline, y[:tr]),
        })
    selection["fixed_test"] = pd.DataFrame(fixed_rows).sort_values(
        ["RMSE", "MAE", "Modelo"]
    ).reset_index(drop=True)
    selection["fixed_test_predictions"] = pd.DataFrame([
        {
            "Modelo": selection["selected"],
            "h": int(step),
            "real": float(real),
            "prediccion_sin_calibrar": float(raw_value),
            "prediccion": float(calibrated_value),
            "error": float(real - calibrated_value),
        }
        for step, real, raw_value, calibrated_value in zip(
            fixed_steps, fixed_actual, fixed_raw, fixed_calibrated
        )
    ])
    selection["fixed_test_origin_date"] = pd.Timestamp(
        selection["series"].fecha_inicio.iloc[fixed_origin]
    ).date()
    selection["test_members"] = members
    return selection["test"].copy()


def future_seasonal_residual(selection, first_plan, weeks=12, refit=True):
    """Reentrena con todo el historial y genera el plan directo por bloques de 12."""
    if selection["test"] is None:
        raise ValueError("Evalúe la prueba antes de generar el pronóstico.")

    series = selection["series"]
    first_plan = pd.Timestamp(first_plan)
    next_date = pd.Timestamp(series.fecha_inicio.iloc[-1]) + pd.Timedelta(days=7)
    days = (first_plan - next_date).days
    if (
        first_plan.weekday() != 0
        or days < 0
        or days % 7
        or days > 112
        or not 1 <= int(weeks) <= 26
    ):
        raise ValueError("Fecha u horizonte inválido.")

    candidate = selection["candidates"][selection["selected"]]
    y = series.valor.to_numpy(dtype=float)
    settings = selection.get("settings")
    batch_size = int(getattr(settings, "batch_size", 16) if settings else 16)
    if refit:
        members = _fit_fixed_ensemble(
            y, len(y), candidate, batch_size=batch_size
        )
    else:
        members = candidate.model

    bridge = days // 7
    total = bridge + int(weeks)
    history = list(map(float, y))
    generated = []
    while len(generated) < total:
        block = min(HORIZON, total - len(generated))
        origins = np.full(block, len(history), dtype=int)
        steps = np.arange(1, block + 1, dtype=int)
        block_forecast = _predict_members(
            members,
            np.asarray(history, dtype=float),
            origins,
            steps,
            bias=candidate.bias_correction,
        )
        history.extend(block_forecast.tolist())
        generated.extend(block_forecast.tolist())

    forecast = np.asarray(generated[bridge:bridge + int(weeks)], dtype=float)
    return pd.DataFrame({
        "fecha_inicio": pd.date_range(first_plan, periods=int(weeks), freq="7D"),
        "pronostico": forecast,
    })
