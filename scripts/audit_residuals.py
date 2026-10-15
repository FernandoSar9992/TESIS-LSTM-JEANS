"""Diagnóstico descriptivo independiente; no ajusta ni selecciona modelos."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast_engine import metrics


def audit(detail_path, primary):
    detail_path = Path(detail_path)
    detail = pd.read_csv(detail_path)
    data = detail[detail.Modelo.eq(primary)].copy()
    if data.empty:
        raise ValueError(f'Modelo no encontrado: {primary}')
    data['Error'] = data.Prediccion - data.Real
    data['Error_absoluto'] = data.Error.abs()
    rows = []
    for (group, horizon), frame in data.groupby(['Grupo', 'Horizonte']):
        frame = frame.sort_values('Indice_origen')
        error = frame.Error.to_numpy()
        acf = float(np.corrcoef(error[:-1], error[1:])[0, 1]) if len(error) > 3 and np.std(error[:-1]) > 0 and np.std(error[1:]) > 0 else np.nan
        worst = frame.loc[frame.Error_absoluto.idxmax()]
        rows.append({'Modelo': primary, 'Grupo': group, 'Horizonte': horizon,
                     'N': len(frame), **metrics(frame.Real.to_numpy(), frame.Prediccion.to_numpy()),
                     'Correlacion_error_lag1': acf,
                     'Max_error_absoluto': float(worst.Error_absoluto),
                     'Periodo_peor_error': worst.Periodo})
    result = pd.DataFrame(rows)
    result.to_csv(detail_path.parent/'auditoria_residuos.csv', index=False)
    test = data[data.Grupo.eq('TEST')].copy()
    test.sort_values('Error_absoluto', ascending=False).head(16).to_csv(detail_path.parent/'auditoria_errores_mayores.csv', index=False)
    # H1 avoids counting the same observed target at several forecast horizons.
    monthly = test[test.Horizonte.eq(1)].copy()
    monthly['Mes'] = monthly.Periodo.astype(str).str.replace(r'-Se\d+$', '', regex=True)
    monthly.groupby('Mes', sort=False).agg(N=('Error', 'size'), MAE=('Error_absoluto', 'mean'),
                                         Sesgo=('Error', 'mean'), Max_error=('Error_absoluto', 'max')).to_csv(detail_path.parent/'auditoria_error_por_mes.csv')
    print(result[result.Grupo.eq('TEST')].to_string(index=False))
    print('Correlación descriptiva; n pequeño y trayectorias solapadas. No es prueba estadística de independencia.')
    print('H1 incluye solo orígenes con trayectoria completa; difiere del conjunto de un paso usado por split_metrics.csv.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('detail')
    parser.add_argument('model')
    args = parser.parse_args()
    audit(args.detail, args.model)
