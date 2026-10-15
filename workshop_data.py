"""Contrato de entrada del taller: una observación por período, sin totales como X."""
import unicodedata
import numpy as np
import pandas as pd
from demand_data import prepare_series, detect_column


def key(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value)).lower() if c.isalnum())


def workshop_columns(frame):
    names = list(frame.columns)
    def prefix(text):
        return next((c for c in names if key(c).startswith(text)), None)
    return {'period': detect_column(names, 'Semana', 'Etiqueta de Semana', 'Periodo', 'Fecha'),
            'target': prefix('demanda'), 'customer': detect_column(names, 'Cliente'),
            'price': prefix('preciounitario'), 'cost': prefix('costounitario')}


def prepare_workshop(frame, mapping=None, declared_real=False):
    mapping = mapping or workshop_columns(frame)
    if any(mapping.get(k) is None for k in ('period', 'target', 'customer', 'price', 'cost')):
        raise ValueError('Se requieren período, demanda, cliente, precio unitario y costo unitario.')
    # No se reordena por cliente: las filas conservan la secuencia temporal del taller.
    prepared = prepare_series(frame, mapping['period'], mapping['target'], declared_real=declared_real)
    customer = frame[mapping['customer']]
    if customer.isna().any() or customer.astype(str).str.strip().eq('').any():
        raise ValueError('Hay clientes vacíos.')
    prepared.data['cliente'] = customer.astype(str).str.strip().to_numpy()
    for output, source in [('precio_unitario', 'price'), ('costo_unitario', 'cost')]:
        values = pd.to_numeric(frame[mapping[source]], errors='coerce').to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{output}: use números finitos no negativos.')
        prepared.data[output] = values
    prepared.notes.append('Ingreso total, costo total y utilidad no entran como predictores: contienen la cantidad que se quiere estimar.')
    return prepared


def feature_audit(data, train_end):
    """Decisiones de información sobre TRAIN, no sobre objetivos de TEST."""
    train = data.iloc[:train_end]
    rows = [{'Variable': 'Cliente', 'Uso': 'Categórica; disponibilidad según el modo de pronóstico'},
            {'Variable': 'Precio unitario', 'Uso': 'Histórico o plan conocido; no se interpreta como efecto causal independiente del cliente'},
            {'Variable': 'Costo unitario', 'Uso': 'Sin variación en TRAIN: no aporta señal' if train.costo_unitario.nunique() == 1 else 'Histórico o plan conocido'},
            {'Variable': 'Ingreso total / costo total / utilidad', 'Uso': 'Excluidos: derivados de la demanda del mismo período'}]
    if train.groupby('cliente').precio_unitario.nunique().max() == 1:
        rows.append({'Variable': 'Cliente y precio', 'Uso': 'Cada cliente tiene un solo precio en TRAIN; no se puede separar el efecto de ambos.'})
    return pd.DataFrame(rows)
