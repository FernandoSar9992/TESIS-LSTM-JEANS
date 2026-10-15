"""Diagnóstico adicional sobre TRAIN/VALIDATION; nunca ajusta con TEST.

Contrasta memoria de la LSTM, generalización, referencias y señal temporal.
No reemplaza el motor ni modifica las observaciones del usuario.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from forecast_engine import metrics, _ridge_fit, _ridge_predict
from lstm_engine import scaled_windows, build_network, sample_weights, anchor_parts
from thesis_engine import ThesisConfig, fit_arrays

OUT = Path('auditoria_2026/modelo_vs_datos')
FOLDS = [(71, 97), (97, 123), (123, 149)]


def design(y, indices, mode, k=1):
    rows = []
    for t in indices:
        features = []
        if mode in ('calendario', 'calendario_tendencia', 'rezagos_calendario'):
            for harmonic in range(1, k+1):
                phase = 2*np.pi*harmonic*t/48
                features.extend([np.sin(phase), np.cos(phase)])
            # Posición 1..4 dentro del mes, conocida al pronosticar.
            features.extend([float(t % 4 == i) for i in range(3)])
        if mode in ('calendario_tendencia', 'tendencia'):
            features.append(t/48)
        if mode in ('rezagos', 'rezagos_calendario'):
            features.extend(y[t-lag] for lag in (1, 2, 3, 4, 8, 12))
            features.extend([np.mean(y[t-8:t]), np.std(y[t-8:t])])
        rows.append(features)
    return np.asarray(rows)


def alternatives(y):
    rows, details = [], []
    for fold, (fit, stop) in enumerate(FOLDS, 1):
        origins = np.arange(fit, stop)
        predictions = {'Media TRAIN': np.full(len(origins), y[:fit].mean()),
                       'Media expansiva': np.array([y[:t].mean() for t in origins]),
                       'Naive': y[origins-1], 'Naive estacional 48': y[origins-48]}
        for window in (4, 8, 12, 24, 48):
            predictions[f'Media movil {window}'] = [y[t-window:t].mean() for t in origins]
            predictions[f'Mediana movil {window}'] = [np.median(y[t-window:t]) for t in origins]
        for alpha in (.05, .1, .2, .4):
            level, values = y[0], []
            for t in range(1, stop):
                if t >= fit:
                    values.append(level)
                level = alpha*y[t]+(1-alpha)*level
            predictions[f'Suavizado exponencial {alpha}'] = values
        predictions['Estacional con ajuste de nivel'] = [y[t-48]+np.mean(y[t-8:t]-y[t-56:t-48]) for t in origins]
        for mode in ('calendario', 'calendario_tendencia', 'rezagos', 'rezagos_calendario', 'tendencia'):
            for harmonics in ((1, 2, 4) if 'calendario' in mode else (1,)):
                for alpha in (1., 10., 100.):
                    train_origins = np.arange(12, fit)
                    model = _ridge_fit(design(y, train_origins, mode, harmonics), y[train_origins, None], alpha)
                    predictions[f'Ridge {mode} K{harmonics} alpha{alpha:g}'] = np.maximum(_ridge_predict(model, design(y, origins, mode, harmonics)).ravel(), 0)
        # Vecinos de ventanas estandarizadas: referencia no lineal independiente.
        train_origins = np.arange(12, fit)
        x = design(y, train_origins, 'rezagos')
        xv = design(y, origins, 'rezagos')
        scale = np.maximum(x.std(0), 1e-8)
        distance = (((xv[:, None]-x[None])/scale)**2).mean(2)
        for k in (3, 8, 16, 32):
            near = np.argsort(distance, axis=1)[:, :k]
            predictions[f'Vecinos {k}'] = np.mean(y[train_origins][near], axis=1)
        for name, pred in predictions.items():
            pred = np.asarray(pred)
            rows.append({'Modelo': name, 'Corte': fold, 'Train_fin': fit, 'Validacion_fin': stop, **metrics(y[origins], pred)})
            details.extend({'Modelo': name, 'Corte': fold, 'Indice': int(t), 'Real': y[t], 'Prediccion': float(p)} for t, p in zip(origins, pred))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'referencias_por_corte.csv', index=False)
    pd.DataFrame(details).to_csv(OUT / 'predicciones_validacion.csv', index=False)
    summary = frame.groupby('Modelo').agg(RMSE_promedio=('RMSE', 'mean'), MAE_promedio=('MAE', 'mean'), R2_minimo=('R2', 'min'), Cortes_R2_positivo=('R2', lambda x: int((x > 0).sum())))
    last = frame.query('Corte == 3').set_index('Modelo')[['RMSE', 'R2']].rename(columns={'RMSE': 'RMSE_validacion_final', 'R2': 'R2_validacion_final'})
    summary = summary.join(last).sort_values('RMSE_promedio')
    summary.to_csv(OUT / 'comparacion_referencias.csv')
    print('REFERENCIAS; sin TEST', summary.head(12).to_string(), flush=True)


def signal_checks(train):
    rng = np.random.default_rng(20261005)
    n = len(train)
    x = np.arange(n)
    residual = train-np.polyval(np.polyfit(x, train, 1), x)
    lags = np.arange(1, 49)
    def correlations(v):
        return np.array([np.corrcoef(v[:-lag], v[lag:])[0, 1] for lag in lags])
    observed = correlations(residual)
    simulated = np.array([correlations(rng.permutation(residual)) for _ in range(1999)])
    maximum = np.max(np.abs(observed))
    p_max = (1+np.sum(np.max(np.abs(simulated), axis=1) >= maximum))/2000
    p_year = (1+np.sum(np.abs(simulated[:, 47]) >= abs(observed[47])))/2000
    pd.DataFrame({'Rezago': lags, 'Correlacion_detrendida': observed}).to_csv(OUT / 'correlaciones_train.csv', index=False)
    result = {'N_train': n, 'correlacion_anual_detrendida': float(observed[47]),
              'max_correlacion_absoluta_1_48': float(maximum), 'rezago_maximo': int(lags[np.argmax(abs(observed))]),
              'p_permutacion_anual': p_year, 'p_maximo_48_rezagos': p_max,
              'Nota': 'Prueba exploratoria con permutación de residuos detrendidos. Supone intercambiabilidad; no demuestra independencia, aleatoriedad ni ausencia de señal no lineal.'}
    (OUT / 'senal_train.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('SEÑAL TRAIN', result, flush=True)


def capacity_check(y):
    import tensorflow as tf
    rows = []
    plans = [dict(name='Local seleccionada, entrenamiento prolongado', lookback=8, units=8, representation='local', periodicity=48, epochs=300, lr=.001),
             dict(name='Mayor capacidad, nivel y sin dropout', lookback=12, units=32, representation='nivel', periodicity=None, epochs=600, lr=.003)]
    for plan in plans:
        tf.keras.backend.clear_session()
        config = ThesisConfig(automatic=False, horizon=1, lookback=plan['lookback'], units=plan['units'],
                              representation=plan['representation'], dropout=0., learning_rate=plan['lr'], epochs=plan['epochs'], reduce_lr=False)
        candidate = {**plan, 'dropout': 0.}
        x, target, origins, anchors, center, scale = scaled_windows(y, 123, plan['lookback'], 1, plan['representation'], plan['periodicity'])
        train = origins < 123
        model = build_network(candidate, config, 42)
        class Snapshot(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch+1 not in (1, 25, 50, 100, 150, 300, 600):
                    return
                for group, mask in [('TRAIN', (origins >= 12) & train), ('VALIDATION', ~train)]:
                    raw = np.asarray(self.model(x[mask], training=False), float).ravel()
                    offset, factor = anchor_parts(anchors[mask])
                    pred = np.maximum((raw*factor+offset)*scale+center, 0)
                    row = {'Experimento': plan['name'], 'Epoca': epoch+1, 'Grupo': group, **metrics(y[origins[mask]], pred)}
                    rows.append(row)
                    print(plan['name'], epoch+1, group, 'R2', round(row['R2'], 4), flush=True)
        fit_arrays(model, x[train], target[train], sample_weights(anchors[train]), config,
                   [Snapshot(), tf.keras.callbacks.TerminateOnNaN()])
    pd.DataFrame(rows).to_csv(OUT / 'capacidad_lstm.csv', index=False)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    # Entrada ya trazada en la auditoría previa. Se descarta TEST antes del análisis.
    data = pd.read_csv('auditoria_2026/despues/processed.csv').iloc[:149].copy()
    y = data.Demanda_utilizada.to_numpy(float)
    assert len(y) == 149
    (OUT / 'protocolo.json').write_text(json.dumps({'cortes': FOLDS, 'objetivo_maximo': 148,
        'test_utilizado': False, 'seleccion_produccion': False, 'control_capacidad_epocas': [300, 600]}, indent=2), encoding='utf-8')
    alternatives(y)
    signal_checks(y[:123])
    capacity_check(y)
