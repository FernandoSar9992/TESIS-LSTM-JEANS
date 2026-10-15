r"""Prueba mínima reproducible del motor sin iniciar Streamlit.

Ejecutar desde la raíz:
    .\.venv\Scripts\python.exe tests\smoke_test.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast_engine import ForecastConfig, clean_series, diagnostic_report, run_experiment


def main() -> None:
    source = ROOT / "Cronograma_Semanas_2023_2026_v24.xlsx"
    if not source.exists():
        raise FileNotFoundError(f"No se encontró el archivo de prueba: {source}")

    raw = pd.read_excel(source)
    data, notes = clean_series(raw, "Etiqueta de Semana", "Demanda")
    assert len(data) >= 70, "La prueba requiere al menos 70 observaciones."
    assert data["demanda"].notna().all(), "La limpieza dejó demandas vacías."

    config = ForecastConfig(horizon=8, lookback=12, seasonal_period=48, epochs=3)
    models = ["Ridge directa", "Último valor", "Media móvil", "Estacional ingenuo", "Media histórica"]
    result = run_experiment(data, config, models)

    assert not result["metrics"].empty, "No se produjeron métricas."
    assert len(result["forecast"]) == config.horizon, "Horizonte futuro incorrecto."
    assert result["test_origins"] >= 2, "No hay suficientes orígenes de prueba."
    assert np.isfinite(result["metrics"]["RMSE"]).all(), "Existe RMSE no finito."
    assert set(models).issubset(set(result["metrics"]["Modelo"])), "Falta un modelo esperado."

    diagnostic = diagnostic_report(data, result, config)
    assert not diagnostic.empty, "El diagnóstico quedó vacío."

    print("SMOKE TEST: CORRECTO")
    print(f"Observaciones: {len(data)} | Notas de limpieza: {len(notes)}")
    print(result["metrics"][["Modelo", "RMSE", "WAPE_%", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
