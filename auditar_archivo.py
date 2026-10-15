"""Auditoría reproducible: selección sin TEST y un único TEST posterior.

Ejemplo: python auditar_archivo.py archivo.xlsx --todos-reales
No modifica la fuente. Guarda CSV, configuración, pesos y huellas del código.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import numpy as np
import pandas as pd
from demand_data import prepare_series, detect_column
from thesis_engine import ThesisConfig, run_thesis, finalize_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archivo', type=Path)
    parser.add_argument('--todos-reales', action='store_true', help='Declaración expresa del propietario de los datos.')
    parser.add_argument('--salida', type=Path, default=Path('auditoria_2026/despues'))
    args = parser.parse_args()
    out = args.salida
    if (out / 'metadata.json').exists() or (out / 'seleccion_fijada.json').exists():
        raise SystemExit('Esta auditoría ya existe. No se sobreescribe ni se vuelve a seleccionar mirando TEST.')
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(args.archivo)
    prepared = prepare_series(raw, detect_column(raw.columns, 'Etiqueta de Semana', 'Periodo', 'Fecha'),
                              detect_column(raw.columns, 'Demanda_Final', 'Demanda'), declared_real=args.todos_reales)
    config = ThesisConfig()
    result = run_thesis(prepared.data, config, prepared.future_labels(config.horizon, config.periodicity),
                        progress=lambda m: print(m, flush=True), evaluate_test=False)
    root = Path(__file__).parent
    code_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ['thesis_engine.py', 'lstm_engine.py', 'demand_data.py', 'forecast_engine.py']}
    frozen = {'config': result['config'], 'selected': result['selected'], 'reference_model': result['reference_model'],
              'data_hash': result['data_hash'], 'source_sha256': hashlib.sha256(args.archivo.read_bytes()).hexdigest(), 'code_sha256': code_hashes}
    (out / 'seleccion_fijada.json').write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding='utf-8')
    result['search'].to_csv(out / 'search.csv', index=False)
    print('CONFIGURACION FIJADA ANTES DE TEST', json.dumps(result['selected']), flush=True)
    finalize_test(result, progress=lambda m: print(m, flush=True))
    result['preparation'] = {'archivo': args.archivo.name, 'declaracion_todos_reales': args.todos_reales, 'notas': prepared.notes}
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f'{key}.csv', index=False, encoding='utf-8-sig')
    detail = result['detail']
    error = detail.Demanda_real.to_numpy()-detail.Prediccion_LSTM.to_numpy()
    score = result['metrics_all'].set_index('Modelo').loc['LSTM']
    np.testing.assert_allclose(score.R2, 1-np.sum(error**2)/np.sum((detail.Demanda_real-detail.Demanda_real.mean())**2))
    np.testing.assert_allclose(score.RMSE, np.sqrt(np.mean(error**2)))
    np.testing.assert_allclose(score.MAE, np.mean(np.abs(error)))
    metadata = {k: v for k, v in result.items() if k != '_artifacts' and not isinstance(v, pd.DataFrame)}
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    for kind in ['models', 'production_models']:
        for i, model in enumerate(result['_artifacts'].get(kind, [])):
            model.save(out / f'{kind}_{i+1}.keras')
    print(result['split_metrics'].query("Modelo == 'LSTM'").to_string(index=False), flush=True)
    print(result['metrics'].to_string(index=False), flush=True)
    print(result['forecast'].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
