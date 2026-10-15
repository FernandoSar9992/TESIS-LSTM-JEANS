"""Comparación predefinida en validación. No evalúa ni optimiza contra TEST."""
from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd
from demand_data import prepare_series
from thesis_engine import ThesisConfig, run_thesis


def main():
    source = Path(r"C:\Users\ASUS\Downloads\VD\DEMANDA_NUEVA.xlsx")
    prepared = prepare_series(pd.read_excel(source), "Etiqueta de Semana", "Demanda")
    config = ThesisConfig()
    result = run_thesis(prepared.data, config, evaluate_test=False,
                        progress=lambda message: print(message, flush=True))
    assert not result['evaluated'] and 'metrics' not in result and 'forecast' not in result
    out = Path(__file__).resolve().parent / 'auditoria_mejora_validacion'
    out.mkdir(exist_ok=True)
    for key in ('search', 'validation_folds', 'split_metrics'):
        result[key].to_csv(out / f'{key}.csv', index=False, encoding='utf-8-sig')
    original = result['search'].query("representation == 'local'").iloc[0]
    winner = result['search'].iloc[0]
    comparison = pd.DataFrame([
        {'Busqueda': 'Anterior: representación local', 'MAE_validacion': original.MAE_validacion,
         'RMSE_validacion': original.RMSE_validacion},
        {'Busqueda': 'Ampliada: tres representaciones', 'MAE_validacion': winner.MAE_validacion,
         'RMSE_validacion': winner.RMSE_validacion}])
    comparison.to_csv(out / 'comparacion_validacion.csv', index=False, encoding='utf-8-sig')
    metadata = {'engine_version': result['engine_version'], 'config': asdict(config),
                'selected': result['selected'], 'evaluated_test': False,
                'data_hash': result['data_hash'], 'train_end': result['train_end'], 'val_end': result['val_end'],
                'warning': 'Comparación de selección en validación, no evidencia de mejora en TEST ni garantía futura.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(result['selected'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
