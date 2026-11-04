"""Protocolo temporal común. Ninguna selección consulta el bloque de prueba."""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from motor_lstm import metrics, _make_lstm, _xy, Settings
from produccion_intermitente import predict_one


def confirm_pre_sales(history, confirmed=False):
    """Ceros declarados por usuario, solo antes de primeras ventas y desde producción."""
    data = history.copy(deep=True)
    trace = pd.DataFrame(columns=['fecha_inicio', 'campo', 'original', 'preparado', 'procedencia'])
    pos = np.flatnonzero(data.produccion_cp.to_numpy(float) > 0)
    observed = np.flatnonzero(data.demanda_cp.notna().to_numpy())
    if confirmed and len(pos) and len(observed):
        idx = np.arange(pos[0], observed[0])
        idx = idx[data.demanda_cp.iloc[idx].isna().to_numpy()]
        trace = pd.DataFrame({'fecha_inicio': data.fecha_inicio.iloc[idx], 'campo': 'demanda_cp',
                              'original': np.nan, 'preparado': 0.,
                              'procedencia': 'Usuario confirma ausencia de ventas; 2026-11-03'})
        data.loc[data.index[idx], 'demanda_cp'] = 0.
    return data, trace.reset_index(drop=True)


def origins(start, end, horizon=12):
    if horizon < 1 or end-start < horizon:
        raise ValueError('Bloque insuficiente para un horizonte completo; no reducirlo silenciosamente.')
    return list(range(int(start), int(end)-int(horizon)+1))


def baseline(history, method, horizon, alpha=.1):
    y = np.asarray(history, float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Historial inválido')
    if method in ('TSB', 'SBA'):
        value = predict_one(y, method, alpha)
    elif method == 'Persistencia':
        value = y[-1]
    elif method.startswith('Media móvil'):
        value = y[-int(method.split()[-1]):].mean()
    elif method == 'Media histórica':
        value = y.mean()
    elif method == 'Dos etapas':
        # Bernoulli MLE y media condicional: modelo hurdle intercept-only.
        # Con pesos uniformes coincide con media histórica; se informa explícitamente.
        value = (y > 0).mean() * (y[y > 0].mean() if (y > 0).any() else 0.)
    else:
        raise ValueError(method)
    return np.repeat(float(value), int(horizon))


@dataclass
class Candidate:
    kind: str
    model: object
    mean: float
    std: float
    lookback: int
    epochs: int
    smear: float = 1.
    alpha: float = .1
    settings: object = None


def encode(y, c):
    a = np.log1p(y) if c.kind == 'log' else np.asarray(y, float)
    return (a-c.mean)/c.std


def decode(z, c):
    a = np.asarray(z, float)*c.std+c.mean
    if c.kind == 'log':
        a = np.exp(np.clip(a, -20, 20))*c.smear-1
    return np.maximum(0., a)


def recursive(c, history, horizon):
    if c.model is None:
        return baseline(history, c.kind, horizon, c.alpha)
    window = list(encode(np.asarray(history, float)[-c.lookback:], c))
    result = []
    for _ in range(int(horizon)):
        x = np.array(window[-c.lookback:], dtype='float32').reshape(1, c.lookback, 1)
        z = float(np.asarray(c.model(x, training=False))[0, 0])
        value = float(decode(z, c))
        if not np.isfinite(value):
            raise ValueError('Pronóstico no finito')
        result.append(value)
        # Feed the forecast actually emitted (including clipping/retransformation).
        window.append(float(encode(value, c)))
    return np.asarray(result)


def _network(kind, settings):
    if kind == 'log':
        return _make_lstm(settings.units, settings.lookback, settings.dropout, settings.seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(settings.seed)
    net = tf.keras.Sequential([
        tf.keras.layers.Input((settings.lookback, 1)),
        tf.keras.layers.LSTM(settings.units), tf.keras.layers.Dropout(settings.dropout),
        tf.keras.layers.Dense(max(8, settings.units//2), activation='relu'),
        tf.keras.layers.Dense(1, activation='softplus')])
    net.compile(optimizer=tf.keras.optimizers.Adam(.001, clipnorm=1.), loss='mse')
    return net


def _dataset(x, y, batch):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch)
    opt = tf.data.Options()
    opt.threading.private_threadpool_size = 1
    opt.threading.max_intra_op_parallelism = 1
    return ds.with_options(opt)


def train_candidate(y, kind, settings, epochs=None):
    """Si epochs=None, cola interna de TRAIN elige épocas; no recibe TEST."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(settings.seed)
    tf.config.experimental.enable_op_determinism()
    y = np.asarray(y, float)
    # First 70% fit, next 15% epoch validation, last 15% independent calibration.
    cut = int(len(y)*.70) if epochs is None else len(y)
    cal = int(len(y)*.85)
    a = np.log1p(y[:cut]) if kind == 'log' else y[:cut]
    mean = float(a.mean()) if kind == 'log' else 0.
    std = (float(a.std()) if kind == 'log' else float(np.sqrt(np.mean(a*a)))) or 1.
    c = Candidate(kind, _network(kind, settings), mean, std, settings.lookback,
                  epochs or settings.epochs, settings=settings)
    z = encode(y, c)
    x, yy = _xy(z, np.arange(settings.lookback, cut), settings.lookback)
    kw = {}
    if epochs is None:
        xv, yv = _xy(z, np.arange(cut, cal), settings.lookback)
        kw = {'validation_data': _dataset(xv, yv, settings.batch_size),
              'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                  patience=settings.patience, restore_best_weights=True)]}
    hist = c.model.fit(_dataset(x, yy, settings.batch_size), epochs=c.epochs, verbose=0, shuffle=False, **kw)
    if epochs is None:
        c.epochs = int(np.argmin(hist.history['val_loss'])+1)
        smear = 1.
        if kind == 'log':
            xc, _ = _xy(z, np.arange(cal, len(y)), settings.lookback)
            loghat = np.asarray(c.model(xc, training=False)).ravel()*std+mean
            smear = float(np.mean(np.exp(np.log1p(y[cal:])-loghat)))
        refit = train_candidate(y, kind, settings, c.epochs)
        return refit, smear
    return c


def compare(candidates, y, start, end, train_reference, horizon=12):
    oo = origins(start, end, horizon)
    rows, details = [], []
    actual = np.concatenate([y[o:o+horizon] for o in oo])
    for name, c in candidates.items():
        pred = np.concatenate([recursive(c, y[:o], horizon) for o in oo])
        rows.append({'Modelo': name, 'Orígenes': len(oo), 'Horizonte': horizon,
                     **metrics(actual, pred, train_reference)})
        for j, o in enumerate(oo):
            for h in range(horizon):
                details.append({'Modelo': name, 'origen': o, 'h': h+1,
                                'indice': o+h, 'real': float(y[o+h]),
                                'prediccion': float(pred[j*horizon+h])})
    return pd.DataFrame(rows), pd.DataFrame(details)


def choose_lstm(validation, candidates):
    """Restricción de tesis: referencias se comparan, nunca sustituyen la LSTM."""
    eligible=[name for name,c in candidates.items() if c.model is not None]
    ranked=validation.loc[validation.Modelo.isin(eligible)].sort_values(
        ['RMSE','MAE','Modelo'],kind='stable')
    if ranked.empty:
        raise ValueError('No hay una LSTM elegible para el pronóstico principal.')
    return str(ranked.iloc[0].Modelo)


def select(series, production=False, settings=None):
    settings = settings or Settings()
    y = series.valor.to_numpy(float)
    tr = int(len(y)*settings.train_ratio)
    ve = int(len(y)*(settings.train_ratio+settings.val_ratio))
    origins(tr, ve); origins(ve, len(y))
    log, smear = train_candidate(y[:tr], 'log', settings)
    candidates = {'LSTM log1p Huber': log}
    if production:
        candidates['LSTM log1p Duan'] = replace(log, smear=smear)
        original, _ = train_candidate(y[:tr], 'original', settings)
        candidates['LSTM original MSE'] = original
        for method in ('TSB', 'SBA'):
            for alpha in (.05, .1, .2, .3):
                candidates[f'{method} {alpha}'] = Candidate(method, None, 0, 1, 0, 0, alpha=alpha)
        candidates['Dos etapas'] = Candidate('Dos etapas', None, 0, 1, 0, 0)
    for method in ('Persistencia', 'Media móvil 4', 'Media móvil 8', 'Media histórica'):
        candidates[method] = Candidate(method, None, 0, 1, 0, 0)
    table, detail = compare(candidates, y, tr, ve, y[:tr])
    table = table.sort_values(['RMSE', 'MAE', 'Modelo'], kind='stable').reset_index(drop=True)
    return {'selected': choose_lstm(table,candidates),
            'best_comparison':str(table.iloc[0].Modelo),'selection_policy':'LSTM obligatoria',
            'candidates': candidates,
            'validation': table, 'validation_predictions': detail,
            'train_end': tr, 'validation_end': ve, 'series': series.copy(),
            'settings': settings, 'smear': smear, 'test': None}


def evaluate(selection):
    if selection['test'] is None:
        y = selection['series'].valor.to_numpy(float)
        selection['test'], selection['test_predictions'] = compare(
            selection['candidates'], y, selection['validation_end'], len(y),
            y[:selection['train_end']])
    return selection['test'].copy()


def future(selection, first_plan, weeks=12, refit=True):
    if selection['test'] is None:
        raise ValueError('Evaluar prueba antes del ajuste final')
    s = selection['series']
    first_plan = pd.Timestamp(first_plan)
    next_date = pd.Timestamp(s.fecha_inicio.iloc[-1])+pd.Timedelta(days=7)
    days = (first_plan-next_date).days
    if first_plan.weekday() != 0 or days < 0 or days % 7 or days > 112 or not 1 <= weeks <= 26:
        raise ValueError('Fecha u horizonte inválido')
    y = s.valor.to_numpy(float)
    c = selection['candidates'][selection['selected']]
    if c.model is None:
        raise ValueError('El pronóstico principal debe ser LSTM; las referencias son solo comparación.')
    if refit and c.model is not None:
        # New object: evaluation network, scaling and metrics remain untouched.
        c = replace(train_candidate(y, c.kind, c.settings, c.epochs), smear=c.smear)
    bridge = days//7
    pred = recursive(c, y, weeks+bridge)[bridge:]
    return pd.DataFrame({'fecha_inicio': pd.date_range(first_plan, periods=weeks, freq='7D'),
                         'pronostico': pred})
