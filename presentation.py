"""Etiquetas de presentación; no modifica datos ni trazabilidad del experimento."""
ORIGINS = {'REAL': 'Observado', 'IMPUTADO': 'Reconstruido', 'MIXTO': 'Mixto',
           'SIN_DECLARAR': 'No indicado en el archivo'}
SCOPES = {'SOLO_REAL': 'Solo observaciones reales', 'SIN_TRAZABILIDAD': 'Procedencia no indicada',
          'TODOS_DESCRIPTIVO': 'Evaluación descriptiva', 'REAL_Y_SIN_TRAZABILIDAD': 'Procedencia parcial'}
GROUPS = {'TRAIN': 'Entrenamiento', 'VALIDATION': 'Validación', 'TEST': 'Prueba final'}


def readable_table(frame):
    result = frame.copy()
    for col in ('origen', 'Origen'):
        if col in result:
            result[col] = result[col].replace(ORIGINS)
    for col in ('Alcance', 'selection_scope'):
        if col in result:
            result[col] = result[col].replace(SCOPES)
    if 'Grupo' in result:
        result['Grupo'] = result['Grupo'].replace(GROUPS)
    return result.rename(columns={'periodo': 'Período', 'Periodo': 'Período', 'demanda': 'Demanda',
                                  'demanda_original': 'Valor original', 'origen': 'Procedencia',
                                  'Origen': 'Procedencia', 'tipo_estimacion': 'Método de reconstrucción',
                                  'R2': 'R²', 'sMAPE_%': 'sMAPE (%)', 'WAPE_%': 'WAPE (%)',
                                  'MAPE_%': 'MAPE (%)', 'Alcance': 'Datos evaluados', 'N': 'Períodos'})


def data_preview(data):
    columns = ['periodo', 'demanda']
    if not data.origen.eq('SIN_DECLARAR').all():
        columns.append('origen')
    return readable_table(data[columns])
