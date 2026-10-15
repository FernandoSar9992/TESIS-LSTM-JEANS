"""Control positivo con señal conocida; no prueba precisión sobre demanda real."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
from pathlib import Path
import numpy as np
import pandas as pd
from thesis_engine import ThesisConfig, run_thesis


if __name__ == '__main__':
    t = np.arange(192)
    y = 500+100*np.sin(2*np.pi*t/12)+np.random.default_rng(2026).normal(0, 3, len(t))
    data = pd.DataFrame({'periodo': [f'p{i}' for i in t], 'demanda': y, 'origen': 'REAL'})
    config = ThesisConfig(automatic=False, periodicity=12, lookback=8, units=8,
                          dropout=0., epochs=120, patience=15, seasonal_features=True)
    result = run_thesis(data, config, progress=lambda m: print(m, flush=True))
    scores = result['split_metrics'].query("Modelo == 'LSTM'")
    print(scores.to_string(index=False), flush=True)
    assert (scores.R2 > .8).all(), 'La red no recupera la señal estacional conocida.'
    expected = 500+100*np.sin(2*np.pi*np.arange(192, 200)/12)
    rmse = float(np.sqrt(np.mean((result['forecast'].LSTM-expected)**2)))
    print('RMSE futuro contra señal generadora:', rmse, flush=True)
    assert rmse < 30, 'Pronóstico estacional sintético incorrecto.'
    out = Path('auditoria_2026/control_estacional')
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / 'metricas.csv', index=False)
    result['forecast'].assign(Esperado_sin_ruido=expected).to_csv(out / 'pronostico.csv', index=False)
