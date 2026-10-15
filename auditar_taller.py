"""Compara multivariable y ablación sobre el MISMO archivo, sin alterar la fuente."""
from pathlib import Path
from dataclasses import asdict, replace
import argparse
import hashlib
import json
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import numpy as np
import pandas as pd
from workshop_data import prepare_workshop
from workshop_engine import WorkshopConfig, train_workshop, finalize_workshop, forecast_workshop


def save_artifact(name, artifact, prefix):
    enc = artifact['encoder']
    description = {'name': name, 'candidate': artifact['candidate'], 'epochs': artifact['epochs'],
                   'center': enc.center, 'scale': enc.scale, 'fit_end': enc.fit_end, 'customers': enc.customers,
                   'mu': enc.mu.tolist(), 'sd': enc.sd.tolist(), 'config': asdict(enc.config),
                   'active_numeric': enc.active_numeric.tolist(),
                   'ridge_candidate': artifact.get('ridge_candidate'), 'baseline': artifact.get('baseline')}
    if artifact.get('members'):
        member_dir = prefix.parent/(prefix.name+'_members')
        member_dir.mkdir(exist_ok=True)
        description['members'] = []
        for i, member in enumerate(artifact['members']):
            member_prefix = member_dir/f'member_{i}'
            save_artifact(name, member, member_prefix)
            description['members'].append(str(member_prefix.with_suffix('.json').relative_to(prefix.parent)))
    else:
        if artifact.get('network') is not None:
            artifact['network'].save(prefix.with_suffix('.keras'))
        if artifact.get('ridge') is not None:
            mu, sd, coef = artifact['ridge']
            np.savez(prefix.with_suffix('.npz'), mu=mu, sd=sd, coef=coef)
    prefix.with_suffix('.json').write_text(json.dumps(description, ensure_ascii=False, indent=2), encoding='utf-8')


def save(result, path):
    path.mkdir(parents=True, exist_ok=True)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(path/f'{key}.csv', index=False, encoding='utf-8-sig')
    metadata = {k: v for k, v in result.items() if k != '_artifacts' and not isinstance(v, pd.DataFrame)}
    (path/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    for stage in ('models', 'production'):
        for i, (name, artifact) in enumerate(result['_artifacts'].get(stage, {}).items()):
            save_artifact(name, artifact, path/f'{stage}_{i}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archivo', type=Path)
    parser.add_argument('--mode', choices=['historico', 'plan'], default='historico')
    parser.add_argument('--todos-reales', action='store_true')
    parser.add_argument('--salida', type=Path, default=Path('auditoria_taller'))
    parser.add_argument('--solo-validacion', action='store_true')
    args = parser.parse_args()
    if (args.salida/'seleccion_fijada.json').exists():
        raise SystemExit('La selección ya está fijada en este directorio. No se sobreescribe TEST.')
    args.salida.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.archivo.read_bytes()).hexdigest()
    prepared = prepare_workshop(pd.read_excel(args.archivo), declared_real=args.todos_reales)
    config = WorkshopConfig(mode=args.mode)
    outcomes = {}
    for title, enabled in [('multivariable', True), ('sin_variables_comerciales', False)]:
        print('INICIO', title, flush=True)
        result = train_workshop(prepared.data, replace(config, use_business=enabled), progress=lambda m: print(m, flush=True))
        result['source_hash'] = source_hash
        result['source_file'] = args.archivo.name
        outcomes[title] = result
        save(result, args.salida/title)
        print(result['search'].head(5).to_string(index=False), flush=True)
    frozen = {name: {'config': result['config'], 'selected': result['selected'], 'search': result['search'].to_dict('records')}
              for name, result in outcomes.items()}
    (args.salida/'seleccion_fijada.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.solo_validacion:
        for name, result in outcomes.items():
            print('EVALUACION FIJADA', name, flush=True)
            finalize_workshop(result, progress=lambda m: print(m, flush=True))
            if config.mode == 'historico':
                forecast_workshop(result, prepared.future_labels(config.horizon, config.periodicity))
            save(result, args.salida/name)
            print(result['split_metrics'].to_string(index=False), flush=True)
            if 'forecast' in result:
                print(result['forecast'].to_string(index=False), flush=True)
    assert hashlib.sha256(args.archivo.read_bytes()).hexdigest() == source_hash
    print('Fuente intacta. Selecciones fijadas antes de TEST.', flush=True)


if __name__ == '__main__':
    main()
