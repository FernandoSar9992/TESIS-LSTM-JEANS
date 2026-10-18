"""Evaluación cronológica de un paso, trazabilidad y TEST final congelado."""
from dataclasses import dataclass, asdict, replace
from hashlib import sha256
import random

import numpy as np
import pandas as pd

from forecast_engine import metrics, _ridge_fit, _ridge_predict, tensorflow_status
from lstm_engine import scaled_windows, future_input, inverse_prediction, build_network, sample_weights

ENGINE_VERSION = "tesis-lstm-v6-anual"


@dataclass(frozen=True)
class ThesisConfig:
    horizon: int = 8
    automatic: bool = True
    lookback: int = 8
    units: int = 16
    representation: str = "local"
    auto_representation: bool = True
    dropout: float = .1
    learning_rate: float = .001
    epochs: int = 300
    patience: int = 20
    batch_size: int = 16
    members: int = 1
    seed: int = 42
    train_ratio: float = .70
    validation_ratio: float = .15
    extended_windows: bool = False
    reduce_lr: bool = True
    periodicity: int = 48
    seasonal_features: bool = True
    refit_future: bool = True
    calendar_experiment: bool = False

    def validate(self):
        if not .5 <= self.train_ratio <= .85 or not .05 <= self.validation_ratio <= .3 or self.train_ratio+self.validation_ratio > .95:
            raise ValueError("Revise los porcentajes: conserve al menos 5 % para TEST.")
        for field in ("horizon", "lookback", "units", "epochs", "patience", "batch_size", "members"):
            if not isinstance(getattr(self, field), int) or getattr(self, field) < 1:
                raise ValueError(f"{field} debe ser un entero positivo.")
        if self.members not in (1, 3) or self.units > 32 or not 2 <= self.lookback <= 24:
            raise ValueError("Use 1 o 3 redes, hasta 32 unidades y ventanas de 2–24 períodos.")
        if self.representation not in ("nivel", "cambio", "local", "estacional") or not 0 <= self.dropout < 1 or not 0 < self.learning_rate <= .1:
            raise ValueError("Representación, dropout o tasa de aprendizaje inválidos.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32 or not isinstance(self.periodicity, int) or self.periodicity < 2:
            raise ValueError("Semilla o periodicidad inválida.")


def split_points(n, horizon=1, train_ratio=.7, validation_ratio=.15):
    tr, va = int(n*train_ratio), int(n*(train_ratio+validation_ratio))
    if tr < 25 or va-tr < 3 or n-va < 2:
        raise ValueError("No hay suficientes datos para TRAIN, VALIDATION y TEST. Agregue períodos o revise la partición.")
    return tr, va


def normalize_data(data):
    data = data.copy().reset_index(drop=True)
    data["demanda"] = pd.to_numeric(data["demanda"], errors="coerce")
    if not np.isfinite(data.demanda).all() or (data.demanda < 0).any():
        raise ValueError("Demandas no finitas o negativas.")
    if data.periodo.isna().any() or data.periodo.astype(str).str.strip().eq("").any() or data.periodo.duplicated().any():
        raise ValueError("Períodos vacíos o duplicados.")
    if "origen" not in data:
        data["origen"] = "SIN_DECLARAR"
    data["origen"] = data.origen.fillna("SIN_DECLARAR").str.upper().str.strip()
    if not data.origen.isin(["REAL", "IMPUTADO", "MIXTO", "SIN_DECLARAR"]).all():
        raise ValueError("Origen no reconocido.")
    if "demanda_original" not in data:
        data["demanda_original"] = data.demanda
    if "tipo_estimacion" not in data:
        data["tipo_estimacion"] = ""
    return data


def metric_values(real, prediction):
    real, prediction = np.asarray(real, float).ravel(), np.asarray(prediction, float).ravel()
    assert len(real) == len(prediction), "Real y predicción desalineados"
    assert np.isfinite(real).all() and np.isfinite(prediction).all(), "Valores no finitos"
    if len(real) == 0:
        return {k: np.nan for k in ("MAE", "RMSE", "sMAPE_%", "R2", "WAPE_%", "MAPE_%", "MSE", "Sesgo")}
    return metrics(real, prediction)


def score_mask(data, origins):
    real = data.iloc[origins].origen.eq("REAL").to_numpy()
    if real.sum() >= 2:
        return real, "SOLO_REAL", True
    if data.iloc[:int(np.max(origins))+1].origen.eq("SIN_DECLARAR").all():
        return np.ones(len(origins), bool), "SIN_TRAZABILIDAD", True
    return np.ones(len(origins), bool), "TODOS_DESCRIPTIVO", False


def table_metrics(data, origins, predictions, group):
    real = data.iloc[origins].demanda.to_numpy(float)
    observed = data.iloc[origins].origen.eq("REAL").to_numpy()
    mask, scope, _ = score_mask(data, origins)
    primary, all_rows, real_rows = [], [], []
    for name, pred in predictions.items():
        pred = np.asarray(pred).ravel()
        assert len(real) == len(pred)
        common = {"Modelo": name, "Grupo": group}
        primary.append({**common, "Alcance": scope, "N": int(mask.sum()), **metric_values(real[mask], pred[mask])})
        all_rows.append({**common, "Alcance": "TODOS_DESCRIPTIVO", "N": len(real), **metric_values(real, pred)})
        real_rows.append({**common, "Alcance": "SOLO_REAL", "N": int(observed.sum()), **metric_values(real[observed], pred[observed])})
    return tuple(pd.DataFrame(rows).sort_values(["RMSE", "MAE"], na_position="last").reset_index(drop=True) for rows in (primary, all_rows, real_rows))


def temporal_folds(tr, va, min_train):
    width = va-tr
    return [(end, end+width) for end in (tr-2*width, tr-width, tr) if end >= min_train]


def calendar_start_offset(data):
    """Posición del primer período dentro del año, sin consultar demanda."""
    import re
    months = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
              "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    label = str(data.periodo.iloc[0]).strip().lower()
    match = re.fullmatch(r"([a-záéíóúñ]+)\s+\d{4}-se([1-4])", label)
    if not match or match.group(1) not in months:
        raise ValueError(f"Período inicial desconocido: {label}")
    return months.index(match.group(1)) * 4 + int(match.group(2)) - 1


def candidates(config, earliest, calendar_offset=0):
    if not config.automatic:
        if earliest-config.lookback < 20:
            raise ValueError("La ventana deja menos de 20 secuencias de entrenamiento.")
        return [{"lookback": config.lookback, "units": config.units, "dropout": config.dropout, "representation": config.representation,
                 "periodicity": config.periodicity if config.seasonal_features else None,
                 "calendar_offset": calendar_offset, "calendar_extra": config.calendar_experiment and config.seasonal_features}]
    windows = [4, 8, 12] + ([16, 24] if config.extended_windows else [])
    modes = ["local", "nivel", "cambio"] if config.auto_representation else [config.representation]
    options = [{"lookback": lb, "units": units, "dropout": dropout, "representation": mode, "periodicity": None}
            for mode in modes for lb in windows if earliest-lb >= 20
            for units, dropout in ((8, 0.), (16, .1), (32, .2))]
    if config.seasonal_features and earliest >= config.periodicity+20:
        options += [{"lookback": lb, "units": units, "dropout": .1, "representation": "local", "periodicity": config.periodicity}
                    for lb in (4, 8, 12) if earliest-lb >= 20 for units in (8, 16)]
    for candidate in options:
        if candidate.get("periodicity"):
            candidate["calendar_offset"] = calendar_offset
    if config.calendar_experiment and config.seasonal_features and config.periodicity == 48:
        options += [{"lookback": lb, "units": 8, "dropout": .1, "representation": "local",
                     "periodicity": 48, "calendar_extra": True, "calendar_offset": calendar_offset}
                    for lb in (4, 8, 12) if earliest-lb >= 20]
    if config.seasonal_features and earliest >= config.periodicity + 12:
        options += [
            {"lookback": lb, "units": units, "dropout": .1,
             "representation": "estacional", "periodicity": config.periodicity,
             "calendar_offset": calendar_offset, "calendar_extra": False}
            for lb in (4, 8, 12) for units in (8, 16)
        ]
    return options


def fit_candidate(y, data, fit_end, stop, candidate, config, seed):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    x, targets, origins, anchors, center, scale = scaled_windows(y[:stop], fit_end, candidate["lookback"], 1, candidate["representation"], candidate.get("periodicity"),
                              candidate.get("calendar_offset", 0), candidate.get("calendar_extra", False))
    train = origins < fit_end
    val = origins >= fit_end
    valid_origins = origins[val]
    mask, scope, eligible = score_mask(data, valid_origins)
    val_indices = np.flatnonzero(val)[mask]
    actual_config = replace(config, horizon=1, dropout=candidate["dropout"])
    model = build_network(candidate, actual_config, seed)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True), tf.keras.callbacks.TerminateOnNaN()]
    if config.reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=8, min_lr=1e-6))
    history = fit_arrays(model, x[train], targets[train], sample_weights(anchors[train]),
                         config, callbacks, validation=(x[val_indices], targets[val_indices], sample_weights(anchors[val_indices])))
    if not np.isfinite(history["loss"]).all() or not np.isfinite(history["val_loss"]).all():
        raise ValueError("El entrenamiento produjo una pérdida no finita.")
    pred = inverse_prediction(model, x[val], anchors[val], center, scale).ravel()
    result = metric_values(y[valid_origins][mask], pred[mask])
    return {"model": model, "center": center, "scale": scale, "history": history,
            "epochs": int(np.argmin(history["val_loss"])+1), "scope": scope, "eligible": eligible,
            "origins": valid_origins, "pred": pred, "scores": result, "samples": int(train.sum())}


def fit_arrays(model, x, targets, weights, config, callbacks, validation=None):
    """Dataset pequeño y determinista; evita crear grandes pools cada época."""
    import tensorflow as tf
    if hasattr(tf, 'data'):
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        def dataset(parts):
            return tf.data.Dataset.from_tensor_slices(parts).batch(min(config.batch_size, len(parts[0]))).with_options(options).prefetch(1)
        return model.fit(dataset((x, targets, weights)), validation_data=dataset(validation) if validation is not None else None,
                         epochs=config.epochs, shuffle=False, verbose=0, callbacks=callbacks).history
    # Redes simuladas en las pruebas de contratos.
    return model.fit(x, targets, sample_weight=weights, validation_data=validation, epochs=config.epochs,
                     batch_size=min(config.batch_size, len(x)), shuffle=False, verbose=0, callbacks=callbacks).history


def predict_observed(models, y, fit_end, candidate, origins):
    x, _, all_origins, anchors, center, scale = scaled_windows(y, fit_end, candidate["lookback"], 1, candidate["representation"], candidate.get("periodicity"),
                              candidate.get("calendar_offset", 0), candidate.get("calendar_extra", False))
    indices = np.searchsorted(all_origins, origins)
    assert np.array_equal(all_origins[indices], origins)
    return np.mean([inverse_prediction(model, x[indices], anchors[indices], center, scale).ravel() for model in models], axis=0)


def ridge_x(y, origins, lags):
    return np.array([[y[t-lag] for lag in lags] for t in origins])


def base_predictions(y, origins, ridge, lags, periodicity=48):
    result = {"Naive": y[origins-1]}
    for window in (4, 8, 12, 24):
        result[f"Media móvil {window}"] = np.array([y[max(0,t-window):t].mean() for t in origins])
    result["Ridge"] = np.maximum(_ridge_predict(ridge, ridge_x(y, origins, lags)).ravel(), 0)
    result["Naive estacional"] = np.array([y[t-periodicity] if t >= periodicity else y[:t].mean() for t in origins])
    return result


def profile_and_split(data, tr, va, include_test=True):
    ranges = [("TRAIN", 0, tr), ("VALIDATION", tr, va), ("TEST", va, len(data))]
    splits, profiles = [], []
    for name, start, stop in ranges:
        segment = data.iloc[start:stop]
        splits.append({"Grupo": name, "Inicio": segment.iloc[0].periodo, "Fin": segment.iloc[-1].periodo, "Observaciones": len(segment),
                       "Reales": int(segment.origen.eq("REAL").sum()), "Imputados_o_mixtos": int(segment.origen.isin(["IMPUTADO", "MIXTO"]).sum())})
        if include_test or name != "TEST":
            profiles.append({"Grupo": name, "Media": segment.demanda.mean(), "Mediana": segment.demanda.median(),
                             "Desviacion": segment.demanda.std(), "Minimo": segment.demanda.min(), "Maximo": segment.demanda.max()})
    return pd.DataFrame(splits), pd.DataFrame(profiles)


def diagnose(result, data, final=False):
    rows = []
    def add(indicator, outcome, interpretation):
        rows.append({"Indicador": indicator, "Resultado": outcome, "Interpretacion": interpretation})
    real_n = int(data.origen.eq("REAL").sum())
    imputed = int(data.origen.isin(["IMPUTADO", "MIXTO"]).sum())
    pct = imputed/len(data)*100
    add("Trazabilidad", f"{real_n} REAL; {imputed} imputados/mixtos ({pct:.2f} %)",
        "Limitación importante: más del 50 % reconstruido; puede aprender patrones de imputación." if pct > 50 else
        "Más del 30 % reconstruido: interprete las métricas con cautela." if pct > 30 else
        "Sin Origen declarado: no es posible certificar observaciones reales." if data.origen.eq("SIN_DECLARAR").any() else "Se conservó el origen de cada objetivo.")
    add("Imputación causal", "POR VERIFICAR" if imputed else "Sin imputaciones declaradas",
        "El archivo no permite demostrar cómo se imputó: confirme que no se usaron datos posteriores para reconstruir el pasado.")
    add("Secuencias TRAIN", str(result["train_windows"]), "Poca información efectiva para una red neuronal." if result["train_windows"] < 100 else "Revisar estabilidad temporal; cantidad no garantiza señal.")
    add("Escalado e inversión", "Verificado", "Media y escala ajustadas solo en TRAIN; transformaciones locales usan solo la ventana pasada. Inversión y alineación verificadas con pruebas automáticas.")
    add("Selección", f"{len(result['search'])} configuraciones; {result['fold_count']} cortes", "Selección por RMSE promedio de validación; MAE desempata. TEST no interviene. Las métricas de validación se usaron para seleccionar y no son prueba independiente.")
    train = data.demanda.iloc[:result['train_end']].to_numpy(float)
    period = result['config']['periodicity']
    corr = float(np.corrcoef(train[:-period], train[period:])[0, 1]) if len(train) >= period+10 and np.std(train[:-period]) > 0 and np.std(train[period:]) > 0 else np.nan
    add("Repetición estacional en TRAIN", f"Correlación a {period} períodos: {corr:.3f}", "Correlación descriptiva, no prueba de estacionalidad. El calendario no garantiza repetición. Se comparan entradas estacionales y no estacionales cuando hay historia suficiente.")
    if result.get("selection_scope") == "TODOS_DESCRIPTIVO":
        add("Validación real insuficiente", "REVISAR", "No hubo cortes con al menos dos objetivos REAL. La selección usa reconstrucciones y no demuestra precisión histórica observada.")
    if final:
        scores = result["metrics"].set_index("Modelo")
        lstm = scores.loc["LSTM"]
        train = result["split_metrics"].query("Grupo == 'TRAIN' and Modelo == 'LSTM'").iloc[0]
        test_real = int(data.iloc[result["val_end"]:].origen.eq("REAL").sum())
        if test_real < 10:
            add("Objetivos reales en TEST", str(test_real), "Evidencia real escasa o sin declarar. Los errores sobre imputaciones son descriptivos, no validación real.")
        if lstm.R2 < 0:
            add("R² negativo", f"{lstm.R2:.4f}", "El modelo presenta un desempeño inferior a utilizar la media del conjunto de prueba como predictor de referencia. Esa media es retrospectiva.")
        if train.R2 > .8 and lstm.R2 < 0:
            add("Generalización", "Posible sobreajuste", "El modelo aprende TRAIN pero presenta baja capacidad de generalización. Un cambio estructural también puede explicar la brecha.")
        elif train.R2 <= .2 and lstm.R2 <= .2:
            add("Aprendizaje", "R² bajo en TRAIN y TEST", "El patrón temporal puede presentar baja capacidad predictiva o la configuración puede ser insuficiente.")
        if all(scores.loc[m, "R2"] < 0 for m in ("LSTM", "Ridge", "Naive")):
            add("Modelos comparativos", "Dificultad compartida", "Varios modelos no generalizan en el mismo TEST; revise cambios de comportamiento, información disponible e imputación.")
        if lstm.R2 < 0 and scores.loc["Ridge", "R2"] > 0:
            add("Configuración LSTM", "Revisar", "Ridge supera la referencia retrospectiva y LSTM no. La dificultad no parece deberse únicamente a los datos.")
        tr_mean = data.demanda.iloc[:result["train_end"]].mean()
        te_mean = data.demanda.iloc[result["val_end"]:].mean()
        change = (te_mean-tr_mean)/tr_mean*100 if tr_mean else np.nan
        add("Cambio de nivel", f"{change:.2f} %" if np.isfinite(change) else "No definido: media TRAIN cero",
            "Posible cambio de nivel superior al 15 %; puede afectar la generalización. Es una regla descriptiva, no una prueba de significancia." if abs(change) > 15 or (tr_mean == 0 and te_mean != 0) else "No se supera el umbral descriptivo de 15 %.")
        add("Comparación LSTM", f"RMSE={lstm.RMSE:.3f}", "Mejor RMSE de TEST en esta comparación." if result["metrics"].iloc[0].Modelo == "LSTM" else "LSTM no tiene el menor RMSE de TEST. No se declara superior.")
    return pd.DataFrame(rows)


def run_thesis(data, config, future_periods=None, progress=None, evaluate_test=True):
    config.validate()
    data = normalize_data(data)
    y = data.demanda.to_numpy(float)
    tr, va = split_points(len(y), 1, config.train_ratio, config.validation_ratio)
    max_lb = (24 if config.extended_windows else 12) if config.automatic else config.lookback
    folds = temporal_folds(tr, va, max_lb+20)
    if not folds:
        raise ValueError("No hay suficientes secuencias para esta ventana; reduzca lookback o agregue datos.")
    options = candidates(config, folds[0][0], calendar_start_offset(data))
    ok, message = tensorflow_status()
    if not ok:
        raise RuntimeError(message)
    import tensorflow as tf
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)
    tf.keras.backend.clear_session()
    notify = progress or (lambda message: None)
    rows, fold_rows, best, best_key, selected = [], [], None, None, None
    for number, candidate in enumerate(options, 1):
        scores = []
        for cut, (fit_end, stop) in enumerate(folds, 1):
            notify(f"LSTM {number}/{len(options)} · {candidate['representation']} · ventana {candidate['lookback']} · {candidate['units']} unidades · corte {cut}/{len(folds)}")
            fitted = fit_candidate(y, data, fit_end, stop, candidate, config, config.seed)
            scores.append(fitted)
            fold_rows.append({"Variante": number, "Corte": cut, "Train_fin": fit_end, "Validacion_fin": stop,
                              "Alcance": fitted["scope"], "Epocas": fitted["epochs"], **fitted["scores"]})
        usable = [s for s in scores if s["eligible"]] or scores
        mae = float(np.mean([s["scores"]["MAE"] for s in usable]))
        rmse = float(np.mean([s["scores"]["RMSE"] for s in usable]))
        scopes = {s["scope"] for s in usable}
        scope = next(iter(scopes)) if len(scopes) == 1 else "REAL_Y_SIN_TRAZABILIDAD"
        rows.append({"Variante": number, **candidate, "epochs": fitted["epochs"], "MAE_validacion": mae, "RMSE_validacion": rmse,
                     "Cortes_utilizados": len(usable), "Alcance": scope})
        if best_key is None or (rmse, mae) < best_key:
            best_key, best = (rmse, mae), None
            selected = {**candidate, "epochs": fitted["epochs"], "Variante": number, "selection_scope": scope}
    # El mejor candidato pudo provenir de otro fold: reajustar sobre TRAIN,
    # antes de reportar validacion; ningun objetivo TEST se utiliza.
    best = fit_candidate(y, data, tr, va, selected, replace(config, epochs=selected["epochs"]), config.seed)
    # Modelo del último corte ya ajustado exclusivamente en TRAIN. No se reajusta
    # ni el scaler ni los pesos con VALIDATION/TEST.
    models = [best["model"]]
    for member in range(1, config.members):
        notify(f"Entrenamiento adicional LSTM {member+1}/{config.members}, solo TRAIN")
        member_fit = fit_candidate(y, data, tr, va, selected, replace(config, epochs=selected["epochs"]), config.seed+member)
        models.append(member_fit["model"])
    common_start = max(selected["lookback"], 12)
    train_origins = np.arange(common_start, tr)
    val_origins = np.arange(tr, va)
    lags = [1, 2, 3, 4, 8, 12]
    x_train, y_train = ridge_x(y, train_origins, lags), y[train_origins, None]
    mask, _, _ = score_mask(data, val_origins)
    ridge_options = []
    for alpha in (.1, 1., 10., 100.):
        ridge = _ridge_fit(x_train, y_train, alpha)
        pred = np.maximum(_ridge_predict(ridge, ridge_x(y, val_origins, lags)).ravel(), 0)
        score = metric_values(y[val_origins][mask], pred[mask])
        ridge_options.append((score["MAE"], score["RMSE"], alpha, ridge))
    _, _, alpha, ridge = min(ridge_options, key=lambda v: (v[1], v[0]))
    splits = []
    for name, origins in (("TRAIN", train_origins), ("VALIDATION", val_origins)):
        predictions = base_predictions(y, origins, ridge, lags, config.periodicity)
        predictions["LSTM"] = predict_observed(models, y[:va], tr, selected, origins)
        splits.append(table_metrics(data, origins, predictions, name)[0])
    split_table, profile = profile_and_split(data, tr, va, include_test=False)
    processed = data.rename(columns={"periodo": "Periodo", "demanda": "Demanda_utilizada", "demanda_original": "Demanda_original", "origen": "Origen", "tipo_estimacion": "Tipo_estimacion"})
    processed["Grupo"] = ["TRAIN"]*tr + ["VALIDATION"]*(va-tr) + ["TEST"]*(len(data)-va)
    h = best["history"]
    result = {"engine_version": ENGINE_VERSION, "config": asdict(config), "selected": selected,
              "search": pd.DataFrame(rows).sort_values(["RMSE_validacion", "MAE_validacion"]), "validation_folds": pd.DataFrame(fold_rows),
              "split_metrics": pd.concat(splits, ignore_index=True), "split_table": split_table, "data_profile": profile,
              "processed": processed, "history": {"loss": h["loss"], "val_loss": h["val_loss"], "parameters": best["model"].count_params()},
              "training_summary": {"Epocas_ejecutadas": len(h["loss"]), "Mejor_epoca": best["epochs"], "Mejor_val_loss": min(h["val_loss"]),
                                   "Train_loss_final": h["loss"][-1], "Val_loss_final": h["val_loss"][-1]},
              "train_end": tr, "val_end": va, "train_windows": tr-selected["lookback"], "validation_origins": va-tr,
              "test_origins": len(data)-va, "fold_count": len(folds), "selection_scope": selected["selection_scope"],
              "ridge_alpha": alpha, "scaler": {"fit_end": tr, "center": best["center"], "scale": best["scale"]},
              "data_hash": sha256(data.to_csv(index=False).encode()).hexdigest(), "evaluated": False,
              "_artifacts": {"models": models, "ridge": ridge, "lags": lags, "data": data, "labels": future_periods}}
    # Referencia elegida antes de abrir TEST. No se sustituye ni se renombra como LSTM.
    result["reference_model"] = splits[-1].query("Modelo != 'LSTM'").iloc[0].Modelo
    result["validation_detail"] = pd.DataFrame({"Periodo": data.periodo.iloc[val_origins].to_numpy(), "Real": y[val_origins], **predictions})
    result["diagnostic"] = diagnose(result, data)
    return finalize_test(result, progress) if evaluate_test else result


def finalize_test(result, progress=None):
    if result["evaluated"]:
        raise ValueError("TEST ya fue evaluado para este experimento.")
    a = result["_artifacts"]
    data, config = a["data"], ThesisConfig(**result["config"])
    y = data.demanda.to_numpy(float)
    origins = np.arange(result["val_end"], len(y))
    predictions = base_predictions(y, origins, a["ridge"], a["lags"], config.periodicity)
    predictions["LSTM"] = predict_observed(a["models"], y, result["train_end"], result["selected"], origins)
    primary, all_rows, real_rows = table_metrics(data, origins, predictions, "TEST")
    result.update(metrics=primary, metrics_all=all_rows, metrics_real=real_rows)
    result["split_metrics"] = pd.concat([result["split_metrics"], primary], ignore_index=True)
    result["detail"] = pd.DataFrame({"Periodo": data.iloc[origins].periodo.to_numpy(), "Origen": data.iloc[origins].origen.to_numpy(),
                                      "Demanda_real": y[origins], "Prediccion_LSTM": predictions["LSTM"],
                                      "Error": y[origins]-predictions["LSTM"], "Error_absoluto": np.abs(y[origins]-predictions["LSTM"])})
    result["detail"]["Interpretacion_objetivo"] = np.where(result["detail"].Origen.eq("REAL"), "Observado", "Reconstruido o no declarado")
    result["baseline_predictions"] = pd.DataFrame({"Periodo": data.iloc[origins].periodo.to_numpy(), **predictions})
    _, result["data_profile"] = profile_and_split(data, result["train_end"], result["val_end"])
    # Evaluación del horizonte con los mismos pesos congelados. Cada trayectoria
    # recibe solo datos anteriores a su origen, nunca los objetivos siguientes.
    result["backtest_horizon"] = min(config.horizon, result["val_end"]-result["train_end"], len(y)-result["val_end"])
    result["recursive_detail"], result["horizon_metrics"] = recursive_evaluation(result)
    result["evaluated"] = True
    result["diagnostic"] = diagnose(result, data, final=True)
    return generate_future(result, progress)


def recursive_predictions(models, histories, candidate, scaler, horizon):
    """Predicción en lote de trayectorias causales, sin observaciones intermedias."""
    from lstm_engine import anchor_parts
    histories = [list(history) for history in histories]
    output = np.empty((len(histories), horizon))
    for step in range(horizon):
        inputs = [future_input(np.asarray(seq), candidate['lookback'], candidate['representation'],
                               scaler['center'], scaler['scale'], candidate.get('periodicity'), candidate.get('calendar_offset', 0),
                               candidate.get('calendar_extra', False)) for seq in histories]
        x = np.concatenate([v[0] for v in inputs])
        offsets, factors = anchor_parts(np.asarray([v[1] for v in inputs]))
        members = []
        for model in models:
            pred = np.asarray(model(x, training=False), float).ravel()
            pred = (pred*factors+offsets)*scaler['scale']+scaler['center']
            if not np.isfinite(pred).all():
                raise ValueError('Pronóstico no finito.')
            members.append(np.maximum(pred, 0))
        output[:, step] = np.mean(members, axis=0)
        for seq, pred in zip(histories, output[:, step]):
            seq.append(float(pred))
    return output


def recursive_reference(y, starts, horizon, name, ridge, lags, periodicity):
    predictions = []
    for start in starts:
        history, trajectory = list(y[:start]), []
        for _ in range(horizon):
            values = np.asarray(history)
            value = float(base_predictions(values, np.array([len(values)]), ridge, lags, periodicity)[name][0])
            history.append(value)
            trajectory.append(value)
        predictions.append(trajectory)
    return np.asarray(predictions)


def recursive_evaluation(result):
    a = result['_artifacts']
    data, y = a['data'], a['data'].demanda.to_numpy(float)
    horizon = result['backtest_horizon']
    details, rows = [], []
    for group, start, stop in [('VALIDATION', result['train_end'], result['val_end']), ('TEST', result['val_end'], len(y))]:
        starts = np.arange(start, stop-horizon+1)
        predictions = {'LSTM': recursive_predictions(a['models'], [y[:t] for t in starts], result['selected'], result['scaler'], horizon),
                       result['reference_model']: recursive_reference(y, starts, horizon, result['reference_model'], a['ridge'], a['lags'], result['config']['periodicity'])}
        for name, pred in predictions.items():
            for h in range(horizon):
                targets = starts+h
                mask, scope, _ = score_mask(data, targets)
                rows.append({'Grupo': group, 'Modelo': name, 'Horizonte': h+1, 'Alcance': scope, 'N': int(mask.sum()),
                             **metric_values(y[targets][mask], pred[:, h][mask])})
                for i, origin in enumerate(starts):
                    details.append({'Grupo': group, 'Modelo': name, 'Indice_origen': int(origin),
                                    'Periodo_origen': data.periodo.iloc[origin], 'Periodo': data.periodo.iloc[origin+h],
                                    'Horizonte': h+1, 'Origen': data.origen.iloc[origin+h],
                                    'Real': y[origin+h], 'Prediccion': pred[i, h]})
    return pd.DataFrame(details), pd.DataFrame(rows)


def fit_production(result, progress=None):
    """Reajuste posterior a TEST; configuración y épocas fijadas en validación."""
    import tensorflow as tf
    a, candidate = result['_artifacts'], result['selected']
    y = a['data'].demanda.to_numpy(float)
    config = replace(ThesisConfig(**result['config']), horizon=1, epochs=candidate['epochs'], dropout=candidate['dropout'])
    x, targets, _, anchors, center, scale = scaled_windows(y, len(y), candidate['lookback'], 1, candidate['representation'], candidate.get('periodicity'),
                                                          candidate.get('calendar_offset', 0), candidate.get('calendar_extra', False))
    models = []
    for member in range(config.members):
        if progress:
            progress(f'Reajuste para el futuro con {len(y)} datos: red {member+1}/{config.members}, {config.epochs} épocas fijadas en validación.')
        model = build_network(candidate, config, config.seed+member)
        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
        history = fit_arrays(model, x, targets, sample_weights(anchors), config, callbacks)
        if not np.isfinite(history['loss']).all():
            raise ValueError('Reajuste futuro inestable.')
        models.append(model)
    a['production_models'] = models
    result['production_scaler'] = {'fit_end': len(y), 'center': center, 'scale': scale}
    result['production_training'] = {'Observaciones': len(y), 'Epocas': config.epochs, 'Redes': config.members,
                                     'Nota': 'Reajuste tras congelar métricas. Su precisión futura no está medida por el TEST del modelo anterior.'}


def generate_future(result, progress=None):
    """Reajuste una sola vez; cambiar horizonte reutiliza el modelo de producción."""
    if not result.get("evaluated"):
        raise ValueError("Evalúe TEST antes de generar el futuro.")
    a = result["_artifacts"]
    config = ThesisConfig(**result["config"])
    config.validate()
    y = a["data"].demanda.to_numpy(float)
    if config.refit_future and 'production_models' not in a:
        fit_production(result, progress)
    models = a['production_models'] if config.refit_future else a['models']
    scaler = result['production_scaler'] if config.refit_future else result['scaler']
    future = recursive_predictions(models, [y], result['selected'], scaler, config.horizon)[0]
    labels = a["labels"] if a["labels"] is not None else [f"Período +{h}" for h in range(1, config.horizon+1)]
    if len(labels) != config.horizon:
        raise ValueError('La cantidad de etiquetas futuras no coincide con el horizonte.')
    origins = np.arange(12, len(y))
    ridge = _ridge_fit(ridge_x(y, origins, a['lags']), y[origins, None], result['ridge_alpha'])
    reference = recursive_reference(y, [len(y)], config.horizon, result['reference_model'], ridge, a['lags'], config.periodicity)[0]
    result["forecast"] = pd.DataFrame({"Periodo": labels, "Horizonte": np.arange(1, config.horizon+1), "LSTM": future,
                                       'Referencia_validacion': reference})
    return result
