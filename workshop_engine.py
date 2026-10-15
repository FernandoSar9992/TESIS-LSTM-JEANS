"""Pronóstico directo multivariable: cliente, precios pasados y calendario causal.

El modo histórico no consulta cliente/precio/costo del objetivo. El modo plan
solo es válido cuando se conocían antes del origen; exige un plan para el futuro.
Ridge, LSTM y LSTM + Ridge mantienen sus nombres y compiten por validación.
"""
from dataclasses import dataclass, asdict, replace
from hashlib import sha256
import numpy as np
import pandas as pd
from forecast_engine import metrics, _ridge_fit, _ridge_predict
from thesis_engine import split_points, temporal_folds, score_mask
from workshop_data import feature_audit

ENGINE_VERSION = 'taller-lstm-v3-nested'


@dataclass(frozen=True)
class WorkshopConfig:
    horizon: int = 8
    mode: str = 'historico'
    periodicity: int = 48
    epochs: int = 180
    patience: int = 20
    seed: int = 42
    train_ratio: float = .70
    validation_ratio: float = .15
    use_business: bool = True
    members: int = 3

    def validate(self):
        if self.mode not in ('historico', 'plan'):
            raise ValueError('Modo inválido.')
        for name in ('horizon', 'periodicity', 'epochs', 'patience'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} debe ser entero positivo.')
        if self.horizon > 48 or self.periodicity < 2 or not 0 <= self.seed < 2**32:
            raise ValueError('Horizonte, ciclo o semilla inválidos.')
        if self.members not in (1, 3) or self.seed+self.members >= 2**32:
            raise ValueError('Use una o tres semillas válidas.')
        if not .5 <= self.train_ratio <= .85 or not .05 <= self.validation_ratio <= .3 or self.train_ratio+self.validation_ratio > .95:
            raise ValueError('Partición temporal inválida.')


def normalize(data):
    data = data.copy().reset_index(drop=True)
    required = ['periodo', 'demanda', 'cliente', 'precio_unitario', 'costo_unitario']
    if any(c not in data for c in required):
        raise ValueError('Faltan columnas del taller.')
    for c in ('demanda', 'precio_unitario', 'costo_unitario'):
        data[c] = pd.to_numeric(data[c], errors='coerce')
        if not np.isfinite(data[c]).all() or (data[c] < 0).any():
            raise ValueError(f'{c} contiene valores inválidos.')
    if data.periodo.isna().any() or data.periodo.duplicated().any():
        raise ValueError('Use una fila por período. Agregar varias transacciones requiere definir su agrupación.')
    if data.cliente.isna().any() or data.cliente.astype(str).str.strip().eq('').any():
        raise ValueError('Hay clientes vacíos.')
    if 'origen' not in data:
        data['origen'] = 'SIN_DECLARAR'
    return data


class Encoder:
    """Escalas y vocabulario congelados exclusivamente en el prefijo de ajuste."""
    def __init__(self, data, fit_end, config):
        train = data.iloc[:fit_end]
        self.fit_end, self.config = fit_end, config
        self.customers = sorted(train.cliente.unique().tolist())
        self.center = float(train.demanda.mean())
        self.scale = max(float(train.demanda.std(ddof=0)), 1.)
        self.mu = train[['precio_unitario', 'costo_unitario']].mean().to_numpy(float)
        self.sd = train[['precio_unitario', 'costo_unitario']].std(ddof=0).to_numpy(float)
        self.active_numeric = (self.sd >= 1e-8).astype(float)
        self.sd[self.sd < 1e-8] = 1.

    def onehot(self, customers):
        lookup = {c: i for i, c in enumerate(self.customers)}
        result = np.zeros((len(customers), len(lookup)+1))
        for j, c in enumerate(customers):
            result[j, lookup.get(c, len(lookup))] = 1.
        return result

    def calendar(self, indices):
        t = np.asarray(indices)
        return np.column_stack([fn(2*np.pi*k*t/self.config.periodicity) for k in (1, 2, 3) for fn in (np.sin, np.cos)] +
                               [np.sin(2*np.pi*t/4), np.cos(2*np.pi*t/4)])

    def encode(self, data, origins, steps, lookback=12, plan=None):
        """Objetivo = origen + paso - 1. Solo la demanda previa al origen entra en X."""
        y = data.demanda.to_numpy(float)
        categories = self.onehot(data.cliente.to_numpy())
        prices = (data[['precio_unitario', 'costo_unitario']].to_numpy(float)-self.mu)/self.sd*self.active_numeric
        if not self.config.use_business:
            categories = np.zeros_like(categories)
            prices = np.zeros_like(prices)
        sequences, context, baselines = [], [], []
        period = self.config.periodicity
        for origin, step in zip(origins, steps):
            origin, step = int(origin), int(step)
            target = origin+step-1
            if origin < lookback or origin > len(data) or step < 1:
                raise ValueError('Origen o horizonte inválido.')
            past = y[origin-lookback:origin]
            recent = float(y[origin-min(8, origin):origin].mean())
            mean24 = float(y[max(0, origin-24):origin].mean())
            seq = np.column_stack(((past-self.center)/self.scale, categories[origin-lookback:origin],
                                   prices[origin-lookback:origin], self.calendar(np.arange(origin-lookback, origin))))
            annual_index = target-period
            available = 0 <= annual_index < origin
            annual = y[annual_index] if available else recent
            annual_category = categories[annual_index] if available else np.zeros(categories.shape[1])
            ctx = [*self.calendar([target])[0], step/period, (recent-self.center)/self.scale,
                   (mean24-self.center)/self.scale, (y[origin-1]-self.center)/self.scale,
                   np.std(past)/self.scale, (annual-self.center)/self.scale, float(available),
                   *categories[max(0, origin-24):origin].mean(0), *categories[origin-1], *annual_category,
                   *prices[origin-1]]
            if self.config.mode == 'plan':
                row = plan.iloc[target-len(data)] if target >= len(data) and plan is not None else data.iloc[target]
                ctx.extend(self.onehot([row.cliente])[0] if self.config.use_business else np.zeros(categories.shape[1]))
                ctx.extend((np.asarray([row.precio_unitario, row.costo_unitario])-self.mu)/self.sd*self.active_numeric if self.config.use_business else np.zeros(2))
            sequences.append(seq)
            context.append(ctx)
            baselines.append(recent)
        seq, context = np.asarray(sequences, np.float32), np.asarray(context, np.float32)
        if not np.isfinite(seq).all() or not np.isfinite(context).all():
            raise ValueError('Entradas no finitas.')
        return seq, context, np.asarray(baselines)


def pairs(start, stop, horizon):
    """Trayectorias completas: ninguna etiqueta cruza la frontera del bloque."""
    starts = np.arange(start, stop-horizon+1)
    if len(starts) < 2:
        raise ValueError('El horizonte deja menos de dos trayectorias completas. Redúzcalo o agregue datos.')
    origins = np.repeat(starts, horizon)
    steps = np.tile(np.arange(1, horizon+1), len(starts))
    return origins, steps


def score(data, targets, pred, fit_end=None, period=48):
    mask, scope, eligible = score_mask(data, targets)
    values = metrics(data.demanda.to_numpy()[targets][mask], np.asarray(pred)[mask])
    prefix = data.demanda.iloc[:fit_end if fit_end is not None else int(np.min(targets))].to_numpy(float)
    scale = float(np.mean(np.abs(prefix[period:]-prefix[:-period]))) if len(prefix) > period else np.nan
    values['MASE'] = values['MAE']/scale if np.isfinite(scale) and scale > 1e-12 else np.nan
    squared_scale = float(np.mean((prefix[period:]-prefix[:-period])**2)) if len(prefix) > period else np.nan
    values['RMSSE'] = np.sqrt(values['MSE']/squared_scale) if np.isfinite(squared_scale) and squared_scale > 1e-12 else np.nan
    return {'N': int(mask.sum()), 'Alcance': scope, 'Escala_MASE': scale, 'Escala_RMSSE_MSE': squared_scale, **values}


def residual_base(data, origins, steps, enc, candidate, recent):
    if candidate.get('residual', 'recent') != 'seasonal':
        return recent
    y = data.demanda.to_numpy(float)
    return np.asarray([y[o+h-1-enc.config.periodicity]
                       if 0 <= o+h-1-enc.config.periodicity < o else fallback
                       for o, h, fallback in zip(origins, steps, recent)])


def ridge_inputs(seq, context, use_history=True):
    # Sin historial equivale a una regresión de calendario (y plan si existe)
    # con regularización; ambas variantes se comparan por validación.
    return np.column_stack((context, seq[:, -1, 0])) if use_history else context[:, :8]


def network(seq_shape, context_size, units, dropout, seed, loss='mse'):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    sequence = tf.keras.Input(seq_shape, name='historia')
    context = tf.keras.Input((context_size,), name='contexto_disponible')
    recurrent = tf.keras.layers.LSTM(units)(sequence)
    hidden = tf.keras.layers.Concatenate()([recurrent, context])
    hidden = tf.keras.layers.Dense(16, activation='tanh')(hidden)
    hidden = tf.keras.layers.Dropout(dropout)(hidden)
    residual = tf.keras.layers.Dense(1, kernel_initializer='zeros')(hidden)
    model = tf.keras.Model([sequence, context], residual)
    if loss not in ('mse', 'mae', 'huber'):
        raise ValueError('Pérdida no admitida.')
    objective = tf.keras.losses.Huber(delta=1.) if loss == 'huber' else loss
    model.compile(optimizer=tf.keras.optimizers.Adam(.001, clipnorm=1.), loss=objective)
    return model


def fit_neural(data, fit_end, stop, config, candidate, ridge_candidate, fixed_epochs=None):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    config = replace(config, use_business=candidate.get('use_business', config.use_business))
    enc = Encoder(data, fit_end, config)
    lookback = candidate.get('lookback', 12)
    origins, steps = pairs(max(lookback, candidate.get('train_start', lookback)), fit_end, config.horizon)
    seq, ctx, base = enc.encode(data, origins, steps, lookback=lookback)
    base = residual_base(data, origins, steps, enc, candidate, base)
    target_indices = origins+steps-1
    ridge = None
    if candidate['family'] == 'LSTM + Ridge':
        ridge = _ridge_fit(ridge_inputs(seq, ctx, ridge_candidate['history']), data.demanda.to_numpy()[target_indices, None], ridge_candidate['alpha'])
        base = np.maximum(_ridge_predict(ridge, ridge_inputs(seq, ctx, ridge_candidate['history'])).ravel(), 0)
    targets = ((data.demanda.to_numpy()[target_indices]-base)/enc.scale).astype(np.float32)[:, None]
    model = network(seq.shape[1:], ctx.shape[1], candidate['units'], candidate['dropout'], config.seed, candidate.get('loss', 'mse'))
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    validation = None
    if fixed_epochs is None:
        vo, vs = pairs(fit_end, stop, config.horizon)
        vseq, vctx, vb = enc.encode(data, vo, vs, lookback=lookback)
        vb = residual_base(data, vo, vs, enc, candidate, vb)
        if ridge is not None:
            vb = np.maximum(_ridge_predict(ridge, ridge_inputs(vseq, vctx, ridge_candidate['history'])).ravel(), 0)
        mask, _, _ = score_mask(data, vo+vs-1)
        vt = ((data.demanda.to_numpy()[vo+vs-1]-vb)/enc.scale).astype(np.float32)[:, None]
        validation = ((vseq[mask], vctx[mask]), vt[mask], np.ones(int(mask.sum()), np.float32))
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    def dataset(x, y, weights):
        return tf.data.Dataset.from_tensor_slices((x, y, weights)).batch(32).with_options(options).prefetch(1)
    history = model.fit(dataset((seq, ctx), targets, np.ones(len(targets), np.float32)),
                        validation_data=dataset(*validation) if validation is not None else None,
                        epochs=fixed_epochs or config.epochs, verbose=0, shuffle=False, callbacks=callbacks).history
    if not all(np.isfinite(v).all() for v in history.values()):
        raise ValueError('Entrenamiento no finito.')
    epoch = fixed_epochs or int(np.argmin(history['val_loss'])+1)
    return {'encoder': enc, 'network': model, 'ridge': ridge, 'ridge_candidate': ridge_candidate,
            'candidate': candidate, 'epochs': epoch, 'history': history}


def fit_temporal_neural(data, fit_end, stop, config, candidate, ridge_candidate):
    """Elige épocas dentro de TRAIN y puntúa después en un bloque externo intacto.

    ``stop`` identifica el final del bloque externo, pero ninguna fila desde
    ``fit_end`` interviene en pesos, escalas ni parada temprana.
    """
    if not fit_end < stop <= len(data):
        raise ValueError('Fronteras de validación externa inválidas.')
    width = max(config.horizon+1, int(np.ceil(fit_end*.2)))
    inner_end = fit_end-width
    start = max(candidate.get('lookback', 12), candidate.get('train_start', 0))
    if inner_end-start-config.horizon+1 < 2:
        raise ValueError('Historia insuficiente para entrenamiento interno y validación temporal.')
    prefix = data.iloc[:fit_end]
    pilot = fit_neural(prefix, inner_end, fit_end, config, candidate, ridge_candidate)
    model = fit_neural(prefix, fit_end, fit_end, config, candidate, ridge_candidate,
                       fixed_epochs=pilot['epochs'])
    model['inner_fit_end'] = inner_end
    return model


def fit_ridge(data, fit_end, config, candidate):
    enc = Encoder(data, fit_end, config)
    origins, steps = pairs(12, fit_end, config.horizon)
    seq, ctx, _ = enc.encode(data, origins, steps)
    model = _ridge_fit(ridge_inputs(seq, ctx, candidate['history']), data.demanda.to_numpy()[origins+steps-1, None], candidate['alpha'])
    return {'encoder': enc, 'ridge': model, 'ridge_candidate': candidate, 'network': None,
            'candidate': {'family': 'Ridge'}, 'epochs': 0}


def predict(artifact, data, origins, steps, plan=None):
    if artifact.get('members'):
        return np.mean([predict(member, data, origins, steps, plan) for member in artifact['members']], axis=0)
    enc = artifact['encoder']
    seq, ctx, base = enc.encode(data, origins, steps, lookback=artifact['candidate'].get('lookback', 12), plan=plan)
    base = residual_base(data, origins, steps, enc, artifact['candidate'], base)
    if artifact.get('baseline'):
        y = data.demanda.to_numpy(float)
        if artifact['baseline'] == 'seasonal':
            base = np.asarray([y[o+h-1-enc.config.periodicity] if 0 <= o+h-1-enc.config.periodicity < o else y[max(0, o-8):o].mean() for o, h in zip(origins, steps)])
        else:
            base = np.asarray([y[max(0, o-artifact['baseline']):o].mean() for o in origins])
    if artifact.get('ridge') is not None:
        base = np.maximum(_ridge_predict(artifact['ridge'], ridge_inputs(seq, ctx, artifact['ridge_candidate']['history'])).ravel(), 0)
    if artifact.get('network') is not None:
        base = base+np.asarray(artifact['network']([seq, ctx], training=False), float).ravel()*enc.scale
    if not np.isfinite(base).all():
        raise ValueError('Predicción no finita.')
    return np.maximum(base, 0)


def select_neural_variant(folds, control):
    """Regla fijada antes de TEST: mejora media, varios cortes y peor corte."""
    reference = folds[folds.Variante.eq(control)].sort_values('Corte')
    eligible = []
    for name, group in folds.groupby('Variante'):
        group = group.sort_values('Corte')
        if len(group) != len(reference):
            raise ValueError('Las variantes deben evaluarse en los mismos cortes.')
        if (group.RMSE.mean() <= reference.RMSE.mean() and group.MAE.mean() <= reference.MAE.mean()
                and group.RMSE.max() <= reference.RMSE.max()
                and (group.RMSE.to_numpy() < reference.RMSE.to_numpy()).sum() >= min(2, len(reference))):
            eligible.append((group.RMSE.mean(), name))
    return min(eligible)[1] if eligible else control


def train_workshop(data, config, progress=None):
    config.validate()
    data = normalize(data)
    tr, va = split_points(len(data), train_ratio=config.train_ratio, validation_ratio=config.validation_ratio)
    folds = temporal_folds(tr, va, 12+config.horizon+20)
    if not folds:
        raise ValueError('Historia insuficiente para entrenar y validar este horizonte.')
    pairs(tr, va, config.horizon)
    pairs(va, len(data), config.horizon)
    notify = progress or (lambda m: None)
    rows, cut_rows, artifacts, seed_rows = [], [], {}, []
    for baseline in (1, 8, 24, 'seasonal'):
        name = 'Último valor' if baseline == 1 else f'Media móvil {baseline}' if isinstance(baseline, int) else 'Naive estacional'
        errors = []
        for cut, (fit, stop) in enumerate(folds, 1):
            model = {'encoder': Encoder(data, fit, config), 'baseline': baseline, 'network': None, 'ridge': None,
                     'candidate': {'family': 'Referencia'}, 'epochs': 0}
            origins, steps = pairs(fit, stop, config.horizon)
            values = score(data, origins+steps-1, predict(model, data.iloc[:stop], origins, steps), fit, config.periodicity)
            errors.append(values['RMSE'])
            cut_rows.append({'Variante': name, 'Corte': cut, 'Train_fin': fit, 'Validacion_fin': stop, **values})
        rows.append({'Variante': name, 'Modelo': 'Referencia', 'RMSE_validacion': float(np.mean(errors)), 'Epocas': 0})
        artifacts[name] = model
    ridge_candidates = [{'alpha': a, 'history': history} for history in (False, True) for a in (1., 10., 100.)]
    best_ridge, best_ridge_score = None, np.inf
    for i, candidate in enumerate(ridge_candidates):
        name = f'Ridge {i+1}'
        scores = []
        for cut, (fit, stop) in enumerate(folds, 1):
            model = fit_ridge(data.iloc[:stop], fit, config, candidate)
            origins, steps = pairs(fit, stop, config.horizon)
            values = score(data, origins+steps-1, predict(model, data.iloc[:stop], origins, steps), fit, config.periodicity)
            scores.append(values['RMSE'])
            cut_rows.append({'Variante': name, 'Corte': cut, 'Train_fin': fit, 'Validacion_fin': stop, **values})
        avg = float(np.mean(scores))
        rows.append({'Variante': name, 'Modelo': 'Ridge', 'RMSE_validacion': avg, 'Epocas': 0, **candidate})
        artifacts[name] = model
        if avg < best_ridge_score:
            best_ridge_score, best_ridge = avg, candidate
    # Ablations predefinidas: un solo cambio frente a la LSTM estacional de control.
    options = [
        {'family': 'LSTM', 'units': 8, 'dropout': .1, 'lookback': 24, 'residual': 'seasonal'},
        {'family': 'LSTM', 'units': 4, 'dropout': .1, 'lookback': 24, 'residual': 'seasonal'},
        {'family': 'LSTM', 'units': 8, 'dropout': .1, 'lookback': 12, 'residual': 'seasonal'},
        {'family': 'LSTM', 'units': 8, 'dropout': .1, 'lookback': 24, 'residual': 'seasonal', 'loss': 'mae'},
        {'family': 'LSTM', 'units': 8, 'dropout': .1, 'lookback': 24, 'residual': 'seasonal', 'loss': 'huber'},
        {'family': 'LSTM', 'units': 8, 'dropout': .1, 'lookback': 24, 'residual': 'seasonal', 'use_business': False},
    ]
    for i, candidate in enumerate(options, 1):
        name = f"LSTM {candidate['units']} ventana {candidate['lookback']} residual {candidate['residual']}"
        if candidate.get('loss', 'mse') != 'mse':
            name += f" loss {candidate['loss']}"
        if candidate.get('use_business') is False:
            name += ' sin comerciales'
        scores = []
        for cut, (fit, stop) in enumerate(folds, 1):
            notify(f"{name}: corte {cut}/{len(folds)}; {i}/{len(options)} configuraciones neuronales")
            origins, steps = pairs(fit, stop, config.horizon)
            members = []
            for offset in range(config.members):
                member_config = replace(config, seed=config.seed+offset)
                member = fit_temporal_neural(data.iloc[:stop], fit, stop, member_config, candidate, best_ridge)
                member_score = score(data, origins+steps-1, predict(member, data.iloc[:stop], origins, steps), fit, config.periodicity)
                seed_rows.append({'Variante': name, 'Corte': cut, 'Semilla': member_config.seed,
                                  'Epocas': member['epochs'], **member_score})
                members.append(member)
            model = {**members[0], 'members': members, 'epochs': [m['epochs'] for m in members]}
            values = score(data, origins+steps-1, predict(model, data.iloc[:stop], origins, steps), fit, config.periodicity)
            scores.append(values['RMSE'])
            cut_rows.append({'Variante': name, 'Corte': cut, 'Train_fin': fit, 'Validacion_fin': stop,
                             'Epocas': str(model['epochs']), **values})
        rows.append({'Variante': name, 'Modelo': 'LSTM', 'RMSE_validacion': float(np.mean(scores)),
                     'Epocas': str(model['epochs']), **candidate, 'Semillas': config.members})
        artifacts[name] = model
    search = pd.DataFrame(rows).sort_values('RMSE_validacion').reset_index(drop=True)
    search['Epocas'] = search.Epocas.astype(str)
    selected = search.iloc[0].Variante
    lstm_names = search[search.Modelo.eq('LSTM')].Variante
    lstm_folds = pd.DataFrame(cut_rows)
    lstm_name = select_neural_variant(lstm_folds[lstm_folds.Variante.isin(lstm_names)],
                                      'LSTM 8 ventana 24 residual seasonal')
    ridge_name = search[search.Modelo.eq('Ridge')].iloc[0].Variante
    kept = {name: artifacts[name] for name in dict.fromkeys([selected, lstm_name,
            'LSTM 8 ventana 24 residual seasonal', ridge_name, 'Último valor', 'Media móvil 8', 'Media móvil 24', 'Naive estacional'])}
    result = {'engine_version': ENGINE_VERSION, 'config': asdict(config), 'selected': selected,
              # La tesis exige presentar la LSTM; selected conserva el ganador
              # estadístico para compararlo como referencia.
              'thesis_selected': lstm_name, 'lstm_name': lstm_name,
              'search': search, 'validation_folds': pd.DataFrame(cut_rows), 'seed_metrics': pd.DataFrame(seed_rows),
              'feature_audit': feature_audit(data, tr),
              'train_end': tr, 'val_end': va, 'data_hash': sha256(data.to_csv(index=False).encode()).hexdigest(),
              'evaluated': False, '_artifacts': {'data': data, 'models': kept}, 'processed': data.copy()}
    result['validation_protocol'] = 'Early stopping interno a TRAIN; puntuación externa por corte.'
    result['split_metrics'], result['detail'], result['horizon_metrics'] = evaluate_groups(result, [('TRAIN', 24, tr), ('VALIDATION', tr, va)])
    return result


def evaluate_groups(result, groups):
    data = result['_artifacts']['data']
    horizon = result['config']['horizon']
    scores, details, horizon_rows = [], [], []
    for group, start, stop in groups:
        one_origins = np.arange(start, stop)
        origins, steps = pairs(start, stop, horizon)
        for name, model in result['_artifacts']['models'].items():
            one = predict(model, data.iloc[:stop], one_origins, np.ones(len(one_origins), int))
            scores.append({'Modelo': name, 'Grupo': group, **score(data, one_origins, one, result['train_end'], result['config']['periodicity'])})
            pred = predict(model, data.iloc[:stop], origins, steps)
            for h in range(1, horizon+1):
                mask = steps == h
                targets = origins[mask]+h-1
                horizon_rows.append({'Modelo': name, 'Grupo': group, 'Horizonte': h, **score(data, targets, pred[mask], result['train_end'], result['config']['periodicity'])})
            for origin, h, value in zip(origins, steps, pred):
                t = origin+h-1
                details.append({'Modelo': name, 'Grupo': group, 'Indice_origen': int(origin), 'Horizonte': int(h),
                                'Periodo': data.periodo.iloc[t], 'Cliente_objetivo': data.cliente.iloc[t],
                                'Real': float(data.demanda.iloc[t]), 'Prediccion': float(value), 'Origen': data.origen.iloc[t]})
    return pd.DataFrame(scores), pd.DataFrame(details), pd.DataFrame(horizon_rows)


def finalize_workshop(result, progress=None):
    if result['evaluated']:
        raise ValueError('TEST ya fue evaluado.')
    metrics_, detail, horizons = evaluate_groups(result, [('TEST', result['val_end'], len(result['_artifacts']['data']))])
    # El reajuste se almacena separado. No altera los modelos/escala de evaluación.
    config = WorkshopConfig(**result['config'])
    data = result['_artifacts']['data']
    production = {}
    for name in dict.fromkeys([result['selected'], result['lstm_name']]):
        if progress:
            progress(f'Reajuste con todo el histórico: {name}')
        old = result['_artifacts']['models'][name]
        if old.get('members'):
            members = [fit_neural(data, len(data), len(data), replace(config, seed=member['encoder'].config.seed),
                                  member['candidate'], member['ridge_candidate'], fixed_epochs=member['epochs'])
                       for member in old['members']]
            new = {**members[0], 'members': members, 'epochs': [m['epochs'] for m in members]}
        elif old.get('baseline'):
            new = {**old, 'encoder': Encoder(data, len(data), config)}
        elif old['network'] is None:
            new = fit_ridge(data, len(data), config, old['ridge_candidate'])
        else:
            new = fit_neural(data, len(data), len(data), config, old['candidate'], old['ridge_candidate'], fixed_epochs=old['epochs'])
        production[name] = new
    # Commit transaccional: si falla un reajuste no se duplica TEST al reintentar.
    result['_artifacts']['production'] = production
    result['split_metrics'] = pd.concat([result['split_metrics'], metrics_], ignore_index=True)
    result['detail'] = pd.concat([result['detail'], detail], ignore_index=True)
    result['horizon_metrics'] = pd.concat([result['horizon_metrics'], horizons], ignore_index=True)
    result['evaluated'] = True
    return result


def forecast_workshop(result, labels, plan=None, model_name=None):
    if not result['evaluated'] or 'production' not in result['_artifacts']:
        raise ValueError('Evalúe TEST y complete el reajuste antes del futuro.')
    data, config = result['_artifacts']['data'], WorkshopConfig(**result['config'])
    if len(labels) != config.horizon:
        raise ValueError('Número de períodos futuros incorrecto.')
    if config.mode == 'plan':
        if plan is None or len(plan) != config.horizon:
            raise ValueError('Este modo requiere cliente, precio y costo unitario conocidos para cada período futuro.')
        if not set(['cliente', 'precio_unitario', 'costo_unitario']).issubset(plan):
            raise ValueError('Plan futuro incompleto.')
        plan = plan.reset_index(drop=True).copy()
        if 'Periodo' in plan and plan.Periodo.astype(str).tolist() != list(map(str, labels)):
            raise ValueError('Los períodos del plan no coinciden con el pronóstico.')
        if plan.cliente.isna().any() or plan.cliente.astype(str).str.strip().eq('').any():
            raise ValueError('Hay clientes vacíos en el plan.')
        for c in ('precio_unitario', 'costo_unitario'):
            plan[c] = pd.to_numeric(plan[c], errors='coerce')
            if not np.isfinite(plan[c]).all() or (plan[c] < 0).any():
                raise ValueError('Precios y costos del plan inválidos.')
    origins = np.full(config.horizon, len(data))
    steps = np.arange(1, config.horizon+1)
    frame = pd.DataFrame({'Periodo': labels, 'Horizonte': steps})
    primary = model_name or result.get('thesis_selected', result.get('lstm_name', result['selected']))
    if primary not in result['_artifacts']['production']:
        raise ValueError(f'Modelo principal no disponible: {primary}.')
    for name, model in result['_artifacts']['production'].items():
        frame[name] = predict(model, data, origins, steps, plan)
    frame['Modelo_principal_tesis'] = primary
    if plan is not None and config.mode == 'plan':
        frame['Cliente_plan'] = plan.cliente.to_numpy()
        frame['Precio_plan'] = plan.precio_unitario.to_numpy()
        frame['Costo_unitario_plan'] = plan.costo_unitario.to_numpy()
        frame['Ingreso_estimado'] = frame[primary]*frame.Precio_plan
        frame['Costo_estimado'] = frame[primary]*frame.Costo_unitario_plan
        frame['Margen_estimado'] = frame.Ingreso_estimado-frame.Costo_estimado
    result['forecast'] = frame
    result['forecast_plan_hash'] = sha256(plan.to_csv(index=False).encode()).hexdigest() if plan is not None and config.mode == 'plan' else None
    return result
