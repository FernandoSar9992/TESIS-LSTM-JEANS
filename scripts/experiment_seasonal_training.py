"""Ensayo acotado; selecciona por validación, conserva el modelo vigente si no mejora."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from dataclasses import replace
from hashlib import sha256
import numpy as np
import pandas as pd
from workshop_engine import WorkshopConfig, fit_neural, predict, pairs, score
from auditar_taller import save_artifact
from verify_workshop_artifacts import load_artifact


def summarize(out):
    """Recalcula trazas guardadas y genera diagnóstico, sin seleccionar ni entrenar."""
    import plotly.express as px
    from forecast_engine import metrics
    out = Path(out)
    old = out.parent/'lstm_v2_2026-10-06/modelo'
    data = pd.read_csv(old/'processed.csv')
    frame = pd.read_csv(out/'predictions.csv')
    name, model = load_artifact(out/'candidate.json', data)
    check = frame[frame.Modelo.eq(name)]
    np.testing.assert_allclose(predict(model, data, check.Origen.to_numpy(int), check.Horizonte.to_numpy(int)),
                               check.Prediccion, atol=3e-4, rtol=1e-5)
    future = data.copy()
    future.loc[153:, ['demanda', 'precio_unitario', 'costo_unitario']] = 1e10
    future.loc[153:, 'cliente'] = 'FUTURO'
    origins, steps = np.full(8, 153), np.arange(1, 9)
    np.testing.assert_array_equal(predict(model, data, origins, steps), predict(model, future, origins, steps))
    rows = []
    for (model_name, h), group in frame[frame.Evaluacion.eq('multihorizonte')].groupby(['Modelo', 'Horizonte']):
        rows.append({'Modelo': model_name, 'Horizonte': h, **metrics(group.Real.to_numpy(), group.Prediccion.to_numpy())})
    pd.DataFrame(rows).to_csv(out/'horizon_metrics.csv', index=False)
    one = frame[frame.Evaluacion.eq('un_paso')].copy()
    one['Error'] = one.Prediccion-one.Real
    one.to_csv(out/'residuals.csv', index=False)
    residual_summary = []
    for model_name, group in one.groupby('Modelo'):
        residual_summary.append({'Modelo': model_name, 'Sesgo': group.Error.mean(),
                                 'Max_error_absoluto': group.Error.abs().max(), 'ACF1': group.Error.autocorr(1)})
    pd.DataFrame(residual_summary).to_csv(out/'residual_summary.csv', index=False)
    chart = one.pivot(index='Periodo', columns='Modelo', values='Prediccion').reindex(one.Periodo.drop_duplicates())
    chart['Demanda real'] = one.drop_duplicates('Periodo').set_index('Periodo').Real
    fig = px.line(chart, markers=True, title='Prueba retrospectiva: demanda real y predicciones a un paso',
                  labels={'value': 'Unidades', 'variable': 'Serie'})
    fig.write_html(out/'real_vs_predicho.html', include_plotlyjs=True)
    (out/'audit.json').write_text(json.dumps({'saved_predictions_reproduced': True,
                                             'future_perturbation_invariant': True,
                                             'promoted': False,
                                             'decision': 'Mejora general no confirmada; conservar modelo vigente.'}, indent=2), encoding='utf-8')
    print('Pesos recargados, predicciones reproducidas y causalidad verificada.', flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    old = root/'results/experiments/lstm_v2_2026-10-06/modelo'
    out = root/'results/experiments/seasonal_training_2026-10-07'
    out.mkdir(exist_ok=False)
    meta = json.loads((old/'metadata.json').read_text(encoding='utf-8'))
    data = pd.read_csv(old/'processed.csv')
    config = WorkshopConfig(**meta['config'])
    original = meta['thesis_selected']
    folds = pd.read_csv(old/'validation_folds.csv')
    reference = folds[folds.Variante.eq(original)].sort_values('Corte')
    candidates = [{'family': 'LSTM', 'units': 8, 'dropout': .1, 'lookback': lb,
                   'residual': 'seasonal', 'train_start': config.periodicity} for lb in (12, 24)]
    protocol = {'candidates': candidates, 'seeds': [42, 43, 44], 'reference': original,
                'folds': reference[['Train_fin', 'Validacion_fin']].to_dict('records'),
                'selection': 'Menor RMSE medio y RMSE menor en al menos 2 de 3 cortes respecto al vigente.',
                'test': 'Retrospectivo ya consultado; no usado para escoger candidatos.',
                'data_sha256': sha256((old/'processed.csv').read_bytes()).hexdigest(),
                'code_sha256': sha256((root/'workshop_engine.py').read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    rows, seeds, models = [], [], {}
    for candidate in candidates:
        name = f"LSTM seasonal completo ventana {candidate['lookback']}"
        for _, fold in reference.iterrows():
            fit, stop = int(fold.Train_fin), int(fold.Validacion_fin)
            origins, steps = pairs(fit, stop, config.horizon)
            members = []
            for offset in range(3):
                print(name, 'corte', fold.Corte, 'semilla', 42+offset, flush=True)
                model = fit_neural(data.iloc[:stop], fit, stop, replace(config, seed=42+offset), candidate, None)
                values = score(data, origins+steps-1, predict(model, data.iloc[:stop], origins, steps), fit, config.periodicity)
                seeds.append({'Modelo': name, 'Corte': fold.Corte, 'Semilla': 42+offset, 'Epocas': model['epochs'], **values})
                members.append(model)
            ensemble = {**members[0], 'members': members, 'epochs': [m['epochs'] for m in members]}
            values = score(data, origins+steps-1, predict(ensemble, data.iloc[:stop], origins, steps), fit, config.periodicity)
            rows.append({'Modelo': name, 'Corte': fold.Corte, **values})
            models[name] = ensemble
            pd.DataFrame(rows).to_csv(out/'validation.csv', index=False)
            pd.DataFrame(seeds).to_csv(out/'seeds.csv', index=False)
    scores = pd.DataFrame(rows)
    eligible = []
    for name, group in scores.groupby('Modelo', sort=False):
        rmse = group.sort_values('Corte').RMSE.to_numpy()
        if (rmse < reference.RMSE.to_numpy()).sum() >= 2 and rmse.mean() < reference.RMSE.mean():
            eligible.append((rmse.mean(), name))
    selected = min(eligible)[1] if eligible else original
    frozen = {'selected': selected, 'reference_RMSE_validation': reference.RMSE.mean(),
              'candidate_RMSE_validation': scores.groupby('Modelo').RMSE.mean().to_dict()}
    (out/'selection_frozen.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    print('Selección fijada:', frozen, flush=True)
    if selected == original:
        print('Mejora no confirmada en validación. Se conserva el modelo vigente.', flush=True)
        return
    selected_model = models[selected]
    save_artifact(selected, selected_model, out/'candidate')
    _, baseline = load_artifact(old/'models_0.json', data)
    rows, detail = [], []
    for name, model in [(original, baseline), (selected, selected_model)]:
        for mode in ('un_paso', 'multihorizonte'):
            origins, steps = pairs(meta['val_end'], len(data), config.horizon) if mode == 'multihorizonte' else (
                np.arange(meta['val_end'], len(data)), np.ones(len(data)-meta['val_end'], int))
            pred = predict(model, data, origins, steps)
            targets = origins+steps-1
            rows.append({'Modelo': name, 'Evaluacion': mode, **score(data, targets, pred, meta['train_end'], config.periodicity)})
            for o, h, t, p in zip(origins, steps, targets, pred):
                detail.append({'Modelo': name, 'Evaluacion': mode, 'Origen': o, 'Horizonte': h,
                               'Periodo': data.periodo.iloc[t], 'Real': data.demanda.iloc[t], 'Prediccion': p})
    pd.DataFrame(rows).to_csv(out/'retrospective_test.csv', index=False)
    pd.DataFrame(detail).to_csv(out/'predictions.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    summarize(out)


if __name__ == '__main__':
    main()
