"""Experimento reproducible: espacio fijo, selección antes de TEST, pesos y trazas."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import numpy as np
import pandas as pd
from auditar_taller import save
from workshop_data import prepare_workshop
from workshop_engine import (ENGINE_VERSION, WorkshopConfig, train_workshop, finalize_workshop,
                             forecast_workshop, predict, score, evaluate_groups, pairs)
from workshop_planning import select_client_model, evaluate_client_model, planning_forecast


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archivo', type=Path)
    parser.add_argument('--salida', type=Path, default=Path('results/experiments/lstm_v2_2026-10-06'))
    parser.add_argument('--epochs', type=int, default=180)
    args = parser.parse_args()
    if args.salida.exists():
        raise SystemExit('El directorio ya existe. No sobrescribir experimentos.')
    args.salida.mkdir(parents=True)
    source_hash = sha256(args.archivo.read_bytes()).hexdigest()
    prepared = prepare_workshop(pd.read_excel(args.archivo, sheet_name='PARA_MODELO'), declared_real=True)
    data = prepared.data
    config = WorkshopConfig(horizon=8, epochs=args.epochs, members=3)
    hashes = {name: sha256(Path(name).read_bytes()).hexdigest() for name in
              ('workshop_engine.py', 'workshop_planning.py', 'workshop_data.py', 'forecast_engine.py', 'thesis_engine.py')}
    manifest = {'engine': ENGINE_VERSION, 'source': str(args.archivo), 'source_sha256': source_hash,
                'rows': len(data), 'first_period': data.periodo.iloc[0], 'last_period': data.periodo.iloc[-1],
                'config': asdict(config), 'seeds': [config.seed+i for i in range(config.members)],
                'status': 'PRERREGISTRADO_ANTES_DE_TEST', 'source_confirmed_real_by_user': True,
                'test_status': 'RETROSPECTIVO: periodos ya consultados en el desarrollo anterior',
                'selection': 'RMSE medio de trayectorias completas por fold; mejor LSTM como principal',
                'code_sha256': hashes, 'python': sys.version, 'platform': platform.platform()}
    write_json(args.salida/'manifest.json', manifest)
    (args.salida/'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    print('Datos:',len(data),'registros. H=8, semillas=42/43/44. TEST retrospectivo.',flush=True)
    result = train_workshop(data, config, progress=lambda message: print(message, flush=True))
    result.update(source_hash=source_hash, source_file=args.archivo.name)
    # Comparador individual de control; no confundir con un experimento histórico.
    previous = result['seed_metrics'].query("Semilla == 42 and Variante.str.endswith('recent')", engine='python')
    if previous.empty:
        previous = result['seed_metrics'].query("Semilla == 42 and Variante == 'LSTM 8 ventana 24 residual seasonal'")
    previous_name = previous.groupby('Variante').RMSE.mean().idxmin()
    old_name = 'Comparador individual: '+previous_name+' semilla 42'
    result['_artifacts']['models'][old_name] = result['_artifacts']['models'][previous_name]['members'][0]
    extra_metrics, extra_detail, extra_horizons = evaluate_groups(
        {**result, '_artifacts': {**result['_artifacts'], 'models': {old_name: result['_artifacts']['models'][old_name]}}},
        [('TRAIN', 24, result['train_end']), ('VALIDATION', result['train_end'], result['val_end'])])
    for key, extra in [('split_metrics', extra_metrics), ('detail', extra_detail), ('horizon_metrics', extra_horizons)]:
        result[key] = pd.concat([result[key], extra], ignore_index=True)
    save(result, args.salida/'modelo')
    frozen = {'principal': result['thesis_selected'], 'comparison_winner': result['selected'],
              'previous_comparator': old_name, 'train_end': result['train_end'], 'val_end': result['val_end'],
              'search': result['search'].to_dict('records'), 'seeds': manifest['seeds']}
    write_json(args.salida/'seleccion_fijada.json', frozen)
    print('Seleccion fijada antes de TEST:',result['thesis_selected'],flush=True)
    print(result['search'][['Variante','RMSE_validacion']].to_string(index=False),flush=True)
    # No volver a modificar candidatos/hiperparámetros después de este punto.
    finalize_workshop(result, progress=lambda message: print(message, flush=True))
    forecast_workshop(result, prepared.future_labels(config.horizon, config.periodicity))
    client = select_client_model(data, result['train_end'], result['val_end'], config.horizon, config.periodicity)
    evaluate_client_model(client, data, result['val_end'])
    result['planning_forecast'] = planning_forecast(result['forecast'], data, result['thesis_selected'], client)
    client['metrics'].to_csv(args.salida/'client_metrics.csv', index=False)
    test_rows = []
    origins, steps = pairs(result['val_end'], len(data), config.horizon)
    for name, model in result['_artifacts']['models'].items():
        if model.get('members'):
            for member in model['members']:
                test_rows.append({'Modelo': name, 'Semilla': member['encoder'].config.seed,
                                  **score(data, origins+steps-1, predict(member, data, origins, steps),
                                           result['train_end'], config.periodicity)})
    result['test_seed_metrics'] = pd.DataFrame(test_rows)
    save(result, args.salida/'modelo')
    assert source_hash == sha256(args.archivo.read_bytes()).hexdigest(), 'La fuente cambió durante la corrida.'
    for name, digest in hashes.items():
        assert sha256(Path(name).read_bytes()).hexdigest() == digest, f'Cambió {name} durante la corrida.'
    manifest['status'] = 'COMPLETO_SIN_RETUNING_TRAS_TEST'
    write_json(args.salida/'manifest.json', manifest)
    print(result['split_metrics'].query("Grupo == 'TEST'").to_string(index=False),flush=True)
    print(result['planning_forecast'].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
