"""Predicción del cliente con su historial y estacionalidad, sin umbrales de demanda."""
import numpy as np
import pandas as pd

CLIENT_VERSION = 'cliente-estacional-v1'
METHODS = ('Cliente más frecuente', 'Último ciclo', 'Frecuencia por posición estacional')
PLANNING_NOTE = (
    'Cliente es una predicción categórica independiente de la demanda, no un pedido confirmado. '
    'Se comparan cliente más frecuente, último ciclo y frecuencia por posición estacional; '
    'selección por exactitud balanceada en VALIDATION, con exactitud como desempate. '
    'Cada origen usa solo clientes anteriores; se evalúan trayectorias completas del horizonte. '
    'No se asigna A/B mediante un umbral de unidades ni se condiciona la demanda al cliente predicho. '
    'TEST ya consultado es una comprobación retrospectiva; hacen falta datos nuevos para una prueba prospectiva.'
)


def client_predictions(history, origin, targets, period, method):
    """Pronóstico directo: ningún cliente posterior al origen entra al cálculo."""
    past = history.cliente.iloc[:origin].astype(str).str.strip().str.upper().to_numpy()
    if not len(past) or period < 1 or method not in METHODS:
        raise ValueError('Historia, ciclo o método de cliente inválidos.')
    targets = np.asarray(targets, dtype=int)
    if np.any(targets < origin):
        raise ValueError('Los objetivos deben ser posteriores al historial disponible.')

    def majority(values):
        names, counts = np.unique(values, return_counts=True)
        return names[np.argmax(counts)]

    common = majority(past)
    output = []
    for target in targets:
        same_phase = np.flatnonzero(np.arange(len(past)) % period == target % period)
        if method == METHODS[0] or not len(same_phase):
            output.append(common)
        elif method == METHODS[1]:
            output.append(past[same_phase[-1]])
        else:
            output.append(majority(past[same_phase]))
    return output


def score_clients(history, start, stop, horizon, period, method, group):
    truth, predictions = [], []
    for origin in range(start, stop-horizon+1):
        targets = np.arange(origin, origin+horizon)
        predictions.extend(client_predictions(history, origin, targets, period, method))
        truth.extend(history.cliente.iloc[targets].astype(str).str.strip().str.upper())
    if not truth:
        raise ValueError('No hay una trayectoria completa para evaluar el cliente.')
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    recalls = {label: float(np.mean(predictions[truth == label] == label)) for label in np.unique(truth)}
    return {'Grupo': group, 'Modelo_cliente': method, 'N_pares': len(truth),
            'Exactitud': float(np.mean(truth == predictions)),
            'Exactitud_balanceada': float(np.mean(list(recalls.values()))),
            **{f'Recall_{label}': value for label, value in recalls.items()}}


def select_client_model(history, train_end, val_end, horizon, period):
    prefix = history.iloc[:val_end]
    rows = [score_clients(prefix, train_end, val_end, horizon, period, method, 'VALIDATION')
            for method in METHODS]
    best = max(rows, key=lambda row: (row['Exactitud_balanceada'], row['Exactitud']))
    return {'version': CLIENT_VERSION, 'selected': best['Modelo_cliente'],
            'horizon': horizon, 'period': period, 'metrics': pd.DataFrame(rows), 'evaluated': False}


def evaluate_client_model(model, history, val_end):
    if model['evaluated']:
        return model
    methods = dict.fromkeys([model['selected'], METHODS[0]])
    rows = [score_clients(history, val_end, len(history), model['horizon'], model['period'], method, 'TEST')
            for method in methods]
    model['metrics'] = pd.concat([model['metrics'], pd.DataFrame(rows)], ignore_index=True)
    model['evaluated'] = True
    return model


def planning_forecast(forecast, history, selected, client_model):
    """Salida simple; conserva la demanda del modelo seleccionado."""
    targets = np.arange(len(history), len(history)+len(forecast))
    clients = client_predictions(history, len(history), targets, client_model['period'], client_model['selected'])
    return pd.DataFrame({'Periodo': forecast.Periodo.to_numpy(),
                         'Pronostico': forecast[selected].to_numpy(), 'Cliente': clients})
