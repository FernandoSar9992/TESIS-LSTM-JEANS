"""Verifica pesos guardados, predicciones, métricas y fuente sin reentrenar."""
from pathlib import Path
import json
import hashlib
import os
import argparse
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import numpy as np
import pandas as pd
from workshop_engine import Encoder, WorkshopConfig, predict, pairs, score


def load_artifact(path, data):
    description = json.loads(path.read_text(encoding='utf-8'))
    encoder = Encoder(data, description['fit_end'], WorkshopConfig(**description['config']))
    artifact = {'encoder': encoder, 'candidate': description['candidate'], 'epochs': description['epochs'],
                'ridge_candidate': description.get('ridge_candidate'), 'baseline': description.get('baseline'),
                'ridge': None, 'network': None}
    if description.get('members'):
        artifact['members'] = [load_artifact(path.parent/member, data)[1] for member in description['members']]
        return description['name'], artifact
    if path.with_suffix('.npz').exists():
        values = np.load(path.with_suffix('.npz'))
        artifact['ridge'] = values['mu'], values['sd'], values['coef']
    if path.with_suffix('.keras').exists():
        import tensorflow as tf
        artifact['network'] = tf.keras.models.load_model(path.with_suffix('.keras'), compile=False)
    return description['name'], artifact


def evaluation_origins(meta, detail, name, group, size):
    """Recupera el intervalo realmente evaluado y exige trayectorias completas."""
    rows = detail[(detail.Modelo == name) & (detail.Grupo == group)]
    if rows.empty:
        raise ValueError(f'Faltan trayectorias de {name}: {group}.')
    start = int(rows.Indice_origen.min())
    stop = {'TRAIN': meta['train_end'], 'VALIDATION': meta['val_end'], 'TEST': size}[group]
    expected_start = {'TRAIN': start, 'VALIDATION': meta['train_end'], 'TEST': meta['val_end']}[group]
    if start != expected_start or not 0 < start < stop:
        raise ValueError(f'Intervalo inválido: {name}, {group}.')
    origins, steps = pairs(start, stop, meta['config']['horizon'])
    actual = rows.sort_values(['Indice_origen', 'Horizonte'])
    np.testing.assert_array_equal(actual[['Indice_origen', 'Horizonte']].to_numpy(),
                                  np.column_stack((origins, steps)))
    return np.arange(start, stop)


def verify(folder, source=None):
    folder = Path(folder)
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    data = pd.read_csv(folder/'processed.csv')
    summary = pd.read_csv(folder/'split_metrics.csv')
    detail = pd.read_csv(folder/'detail.csv')
    horizon_metrics = pd.read_csv(folder/'horizon_metrics.csv')
    if source is None:
        manifest = folder.parent/'manifest.json'
        if not manifest.exists():
            raise ValueError('Indique el Excel original con --source; no hay manifest.json.')
        source = json.loads(manifest.read_text(encoding='utf-8'))['source']
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != meta['source_hash']:
        raise ValueError('El archivo fuente no coincide con el utilizado en esta ejecución.')
    model_paths = sorted(folder.glob('models_*.json'))
    production_paths = sorted(folder.glob('production_*.json'))
    if not model_paths or not production_paths:
        raise ValueError('Faltan modelos guardados de evaluación o producción.')
    max_difference = 0.
    for path in model_paths:
        name, artifact = load_artifact(path, data)
        for group in ('TRAIN', 'VALIDATION', 'TEST'):
            origins = evaluation_origins(meta, detail, name, group, len(data))
            stop = int(origins[-1])+1
            prediction = predict(artifact, data.iloc[:stop], origins, np.ones(len(origins), int))
            computed = score(data, origins, prediction, meta['train_end'], meta['config']['periodicity'])
            saved = summary[(summary.Modelo == name) & (summary.Grupo == group)].iloc[0]
            for measure in ('RMSE', 'MAE', 'R2'):
                np.testing.assert_allclose(computed[measure], saved[measure], atol=3e-4, rtol=1e-5)
                max_difference = max(max_difference, float(abs(computed[measure]-saved[measure])))
        rows = detail[detail.Modelo == name]
        targets = (rows.Indice_origen + rows.Horizonte - 1).to_numpy(int)
        np.testing.assert_allclose(rows.Real, data.demanda.iloc[targets], atol=1e-9)
        prediction = predict(artifact, data, rows.Indice_origen.to_numpy(), rows.Horizonte.to_numpy())
        np.testing.assert_allclose(prediction, rows.Prediccion, atol=3e-4, rtol=1e-5)
    for (name, group, h), rows in detail.groupby(['Modelo', 'Grupo', 'Horizonte']):
        targets = (rows.Indice_origen + rows.Horizonte - 1).to_numpy(int)
        computed = score(data, targets, rows.Prediccion.to_numpy(), meta['train_end'], meta['config']['periodicity'])
        saved = horizon_metrics[(horizon_metrics.Modelo == name)&(horizon_metrics.Grupo == group)&(horizon_metrics.Horizonte == h)].iloc[0]
        for measure in ('RMSE', 'MAE', 'R2'):
            np.testing.assert_allclose(computed[measure], saved[measure], atol=1e-9)
    forecast = pd.read_csv(folder/'forecast.csv')
    for path in production_paths:
        name, artifact = load_artifact(path, data)
        pred = predict(artifact, data, np.full(len(forecast), len(data)), forecast.Horizonte.to_numpy())
        np.testing.assert_allclose(pred, forecast[name], atol=3e-4, rtol=1e-5)
    print(folder.name, 'Fuente, modelos recargados, métricas y futuro verificados. Diferencia máxima:', max_difference, flush=True)
    return max_difference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Carpeta que contiene metadata.json y los modelos.')
    parser.add_argument('--source', type=Path, help='Excel original; por defecto se lee del manifest.json superior.')
    args = parser.parse_args()
    verify(args.folder, args.source)
