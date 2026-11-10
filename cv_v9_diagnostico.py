"""Auditoría temporal V9 aislada del motor de pronóstico principal.

VALIDACIÓN CRUZADA CON VENTANA EXPANSIVA Y REENTRENAMIENTO REAL.
- Solo utiliza el tramo de desarrollo anterior a TEST.
- Cada fold tiene un objetivo futuro fijo de 12 semanas, sin observar semanas
  intermedias durante la predicción.
- Dentro de cada fold, la arquitectura se elige con las 12 semanas ANTERIORES
  al origen de evaluación; las épocas se eligen antes mediante parada temprana.
- Nunca guarda ni modifica pesos, parámetros, pronósticos o estado de app.py.
- Las curvas de pérdida pertenecen al entrenamiento piloto de ESTE módulo.
- Las métricas calculadas sobre reconstrucciones no prueban demanda observada.
"""
from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import pandas as pd

from lstm_residual_estacional import (
    HORIZON, _candidate_grid, _scale_from, _training_pairs,
    _build_model, _predict_members, _fit_fixed_member, _baseline_predictions,
)
from motor_lstm import metrics


@dataclass(frozen=True)
class AuditConfig:
    train_ratio: float = .70
    val_ratio: float = .15
    lookback: int = 12
    units: int = 16
    dropout: float = .15
    epochs: int = 100
    patience: int = 12
    batch_size: int = 16
    seed: int = 42
    max_folds: int = 3
    horizon: int = HORIZON

    def validate(self):
        if self.horizon != HORIZON:
            raise ValueError("Esta auditoría utiliza horizonte fijo de 12 semanas.")
        if not (.55 <= self.train_ratio <= .80 and
                .10 <= self.val_ratio <= .25 and
                self.train_ratio + self.val_ratio <= .90):
            raise ValueError("Partición temporal no válida.")
        if not (1 <= self.max_folds <= 5):
            raise ValueError("max_folds debe estar entre 1 y 5.")
        if not (0 <= self.dropout < 1.0):
            raise ValueError("Dropout debe estar entre 0 y 1.")
        if min(self.epochs, self.patience, self.batch_size, self.lookback, self.units) < 1:
            raise ValueError("Los hiperparámetros deben ser positivos.")


def expanding_folds(n, train_ratio=.70, val_ratio=.15, horizon=12,
                    max_folds=3, min_train=84):
    """Ventanas de evaluación completas, contiguas, sin incluir TEST.

    Devuelve pares (origen, fin_exclusivo). Cada objetivo es disjunto.
    La última ventana termina exactamente al comienzo del bloque TEST.
    """
    n = int(n)
    horizon = int(horizon)
    cutoff = int(n * (train_ratio + val_ratio))
    starts = list(range(cutoff - horizon, int(min_train) - 1, -horizon))
    starts = sorted(starts[:int(max_folds)])
    if not starts:
        raise ValueError(
            "Histórico insuficiente para reentrenar V9 y reservar una validación "
            "externa de 12 semanas sin tocar TEST. No se fabricarán folds."
        )
    folds = [(start, start + horizon) for start in starts]
    if any(start < min_train or end > cutoff or end - start != horizon
           for start, end in folds):
        raise AssertionError("Cortes temporales incorrectos.")
    return folds


def _train_member_with_curve(y, fit_end, lookback, units, seed, config):
    """Reproduce el piloto V9 y conserva loss/val_loss sin cambiar V9."""
    import keras

    values = np.asarray(y, dtype=float)
    fit_end = int(fit_end)
    inner_width = max(HORIZON + 1, int(math.ceil(fit_end * .20)))
    inner_end = fit_end - inner_width
    if inner_end <= int(lookback) + HORIZON + 8:
        raise ValueError("TRAIN insuficiente para validación interna.")

    pilot_center, pilot_scale = _scale_from(values, inner_end)
    x_train, c_train, _, y_train, _ = _training_pairs(
        values, inner_end, lookback, pilot_center, pilot_scale
    )
    x_valid, c_valid, b_valid, y_valid, _ = _training_pairs(
        values, fit_end, lookback, pilot_center, pilot_scale,
        start_origin=inner_end
    )
    if len(x_train) < 30 or len(x_valid) < HORIZON:
        raise ValueError("La LSTM no tiene suficientes ejemplos completos.")

    keras.utils.set_random_seed(int(seed))
    pilot = _build_model(lookback, units, config.dropout, 1e-4)
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        min_delta=1e-5,
        restore_best_weights=True,
    )
    history = pilot.fit(
        [x_train, c_train], y_train,
        validation_data=([x_valid, c_valid], y_valid),
        epochs=config.epochs,
        batch_size=min(config.batch_size, len(x_train)),
        shuffle=False,
        verbose=0,
        callbacks=[callback, keras.callbacks.TerminateOnNaN()]
    ).history
    losses = np.asarray(history["loss"], dtype=float)
    val_losses = np.asarray(history["val_loss"], dtype=float)
    if not np.isfinite(losses).all() or not np.isfinite(val_losses).all():
        raise ValueError("Pérdida no finita durante entrenamiento piloto.")
    best_epoch = int(np.argmin(val_losses) + 1)

    # Smearing usando exclusivamente la cola interna de TRAIN.
    pilot_pred = pilot.predict([x_valid, c_valid], verbose=0).ravel()
    actual_log = b_valid + y_valid.ravel() * pilot_scale
    predicted_log = b_valid + pilot_pred * pilot_scale
    smear = float(np.mean(
        np.exp(np.clip(actual_log - predicted_log, -3.0, 3.0))
    ))
    smear = float(np.clip(smear, .85, 1.25))

    # Reajuste con número de épocas ya seleccionado dentro del prefijo.
    return_member = _fit_fixed_member(
        values, fit_end, lookback, units, seed, best_epoch,
        smear=smear, dropout=config.dropout, l2_value=1e-4,
        batch_size=config.batch_size
    )
    curve = pd.DataFrame({
        "Época": np.arange(1, len(losses) + 1),
        "loss_train": losses,
        "loss_validacion_interna": val_losses,
        "Época_óptima": best_epoch,
    })
    return return_member, curve


def _score(y_true, y_pred, train_reference):
    return metrics(
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
        np.asarray(train_reference, dtype=float),
    )


def evaluate_cross_validation(series, config: Optional[AuditConfig] = None,
                              progress=None):
    """Ejecuta folds temporales con entrenamiento real y selección anidada.

    No usa ni evalúa el TEST del modelo principal. No cambia el modelo V9.
    """
    config = config or AuditConfig()
    config.validate()
    data = series.copy().reset_index(drop=True)
    if not {"fecha_inicio", "valor"}.issubset(data.columns):
        raise ValueError("Se necesitan fecha_inicio y valor.")
    y = data["valor"].to_numpy(dtype=float)
    dates = pd.to_datetime(data["fecha_inicio"], errors="coerce")
    if dates.isna().any() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("La serie contiene fechas o demandas inválidas.")
    if (dates.diff().dropna().dt.days != 7).any():
        raise ValueError("Las fechas deben ser semanas consecutivas.")

    folds = expanding_folds(
        len(y), config.train_ratio, config.val_ratio,
        HORIZON, config.max_folds
    )
    test_start = int(len(y) * (config.train_ratio + config.val_ratio))
    candidate_grid = _candidate_grid(config)
    seeds = (config.seed, config.seed + 1, config.seed + 2)

    fold_rows, candidate_rows, prediction_rows, curve_rows = [], [], [], []
    for fold_number, (origin, stop) in enumerate(folds, start=1):
        inner_train_end = origin - HORIZON
        inner_val_origins = np.full(HORIZON, inner_train_end, dtype=int)
        outer_origins = np.full(HORIZON, origin, dtype=int)
        steps = np.arange(1, HORIZON + 1, dtype=int)
        inner_true = y[inner_train_end:origin]
        outer_true = y[origin:stop]

        if progress:
            progress(
                f"Fold {fold_number}/{len(folds)}: entrenando hasta "
                f"{dates.iloc[origin - 1].date()}"
            )
        best = None
        best_key = (float("inf"), float("inf"))
        best_members = None
        for lookback, units in candidate_grid:
            members = []
            for seed in seeds:
                member, curve = _train_member_with_curve(
                    y, inner_train_end, lookback, units, seed, config
                )
                members.append(member)
                curve["Fold"] = fold_number
                curve["Ventana"] = lookback
                curve["Unidades"] = units
                curve["Semilla"] = seed
                curve_rows.append(curve)

            inner_pred = _predict_members(
                members, y, inner_val_origins, steps, bias=0.0
            )
            score = _score(inner_true, inner_pred, y[:inner_train_end])
            candidate_rows.append({
                "Fold": fold_number,
                "Ventana": lookback,
                "Unidades": units,
                "Semillas": len(seeds),
                "RMSE_validación_anidada": score["RMSE"],
                "MAE_validación_anidada": score["MAE"],
                "Fecha_fin_entrenamiento_anidado":
                    str(dates.iloc[inner_train_end - 1].date()),
            })
            key = (score["RMSE"], score["MAE"])
            if key < best_key:
                best_key = key
                best = (lookback, units)
                best_members = members

        # El sesgo se ajusta SOLO en la validación interior a este fold.
        best_inner_pred = _predict_members(
            best_members, y, inner_val_origins, steps, bias=0.0
        )
        bias = float(np.mean(inner_true - best_inner_pred))
        bound = .30 * max(1.0, float(np.mean(y[:inner_train_end])))
        bias = float(np.clip(bias, -bound, bound))

        # REENTRENAMIENTO del candidato ganador con el prefijo completo.
        # Las épocas y el smearing provienen exclusivamente del prefijo anterior.
        refitted = []
        for member in best_members:
            refitted.append(_fit_fixed_member(
                y, origin, best[0], best[1], member["seed"],
                member["epochs"], smear=member["smear"],
                dropout=config.dropout, l2_value=1e-4,
                batch_size=config.batch_size
            ))
        predictions = _predict_members(
            refitted, y, outer_origins, steps, bias=bias
        )
        score = _score(outer_true, predictions, y[:origin])
        naive48 = _score(
            outer_true, _baseline_predictions(y, outer_origins, steps, 48),
            y[:origin]
        )
        naive52 = _score(
            outer_true, _baseline_predictions(y, outer_origins, steps, 52),
            y[:origin]
        )
        fold_rows.append({
            "Fold": fold_number,
            "Entrenamiento_hasta": str(dates.iloc[origin - 1].date()),
            "Validación_desde": str(dates.iloc[origin].date()),
            "Validación_hasta": str(dates.iloc[stop - 1].date()),
            "Semanas_evaluadas": HORIZON,
            "Ventana_elegida": best[0],
            "Unidades_elegidas": best[1],
            "Bias_interno": bias,
            "MAE_LSTM": score["MAE"],
            "RMSE_LSTM": score["RMSE"],
            "R2_LSTM": score["R²"],
            "Sesgo_LSTM": score["Sesgo"],
            "RMSE_Naive48": naive48["RMSE"],
            "RMSE_Naive52": naive52["RMSE"],
            "MAE_Naive48": naive48["MAE"],
            "MAE_Naive52": naive52["MAE"],
        })
        for i, step in enumerate(steps):
            prediction_rows.append({
                "Fold": fold_number,
                "Fecha_objetivo": str(dates.iloc[origin + i].date()),
                "Horizonte": int(step),
                "Real": float(outer_true[i]),
                "LSTM": float(predictions[i]),
                "Naive48": float(_baseline_predictions(
                    y, [origin], [step], 48
                )[0]),
                "Naive52": float(_baseline_predictions(
                    y, [origin], [step], 52
                )[0]),
            })

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(candidate_rows)
    curves = pd.concat(curve_rows, ignore_index=True)
    rmse = folds_df.RMSE_LSTM.to_numpy(float)
    rmse_naive = folds_df[["RMSE_Naive48", "RMSE_Naive52"]].min(axis=1)
    summary = pd.DataFrame([{
        "Folds": len(folds_df),
        "Horizonte_semanas": HORIZON,
        "TEST_inicio_indice": test_start,
        "TEST_consultado": False,
        "Promedio_MAE_LSTM": float(folds_df.MAE_LSTM.mean()),
        "Promedio_RMSE_LSTM": float(np.mean(rmse)),
        "Desviacion_RMSE_folds": float(np.std(rmse, ddof=0)),
        "Mejor_referencia_por_fold": int((rmse < rmse_naive).sum()),
        "Folds_superados_por_LSTM": int((rmse < rmse_naive).sum()),
    }])
    return {
        "folds": folds_df,
        "predictions": preds_df,
        "candidates": selected,
        "curves": curves,
        "summary": summary,
        "test_start": test_start,
    }
