"""Auditoría reproducible: no selecciona hiperparámetros mirando la prueba.

Ejecutar: .venv/Scripts/python.exe auditar_pronostico.py
Los resultados nuevos se guardan aparte de los informes históricos.
"""
from pathlib import Path
import json
import platform
from dataclasses import asdict
from importlib.metadata import version

import pandas as pd

from forecast_engine import ForecastConfig, clean_series, run_experiment

ROOT = Path(__file__).resolve().parent
MODELS = ["LSTM", "Ridge directa", "Media histórica", "Último valor", "Media móvil", "Estacional ingenuo"]


def main():
    data, notes = clean_series(pd.read_excel(ROOT / "Cronograma_Semanas_2023_2026_v24.xlsx"), "Etiqueta de Semana", "Demanda")
    out = ROOT / "auditoria"
    out.mkdir(exist_ok=True)
    rows = []
    # Semillas prefijadas para sensibilidad, NO elegir la mejor por error de prueba.
    for seed in (42, 7, 123):
        config = ForecastConfig(seasonal_period=48, seed=seed)
        result = run_experiment(data, config, MODELS)
        result["metrics"].assign(Semilla=seed).to_csv(out / f"metricas_semilla_{seed}.csv", index=False)
        rows.append(result["metrics"].assign(Semilla=seed))
        if seed == 42:
            for key in ("metrics", "validation_metrics", "horizon_metrics", "detail", "forecast", "future_all"):
                result[key].to_csv(out / f"{key}.csv", index=False, encoding="utf-8-sig")
            metadata = {"config": asdict(config), "n": len(data), "notes": notes,
                        "train_end": result["train_end"], "val_end": result["val_end"],
                        "train_windows": result["train_windows"], "validation_origins": result["validation_origins"],
                        "test_origins": result["test_origins"], "winner_validation": result["winner"],
                        "history": result["history"], "python": platform.python_version(),
                        "tensorflow": version("tensorflow"), "streamlit": version("streamlit")}
            (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"SEMILLA {seed} | Seleccionado en validacion: {result['winner']}", flush=True)
        print(result["metrics"][["Modelo", "RMSE", "MAE", "WAPE_%", "R2"]].to_string(index=False), flush=True)
    pd.concat(rows).to_csv(out / "sensibilidad_semillas.csv", index=False, encoding="utf-8-sig")
    temporal_rows = []
    # Tres pruebas de 24 períodos SIN solapamiento entre cortes. Parámetros
    # prefijados; únicamente la época se selecciona con validación de cada corte.
    for n in (108, 132, 156):
        config = ForecastConfig(seasonal_period=48, seed=42,
                                train_ratio=(n-48+0.001)/n, validation_ratio=24/n)
        result = run_experiment(data.iloc[:n].copy(), config, MODELS)
        assert result["train_end"] == n-48 and result["val_end"] == n-24
        frame = result["metrics"].assign(Fin_datos=n, Inicio_prueba=n-24,
                                          Seleccionado_validacion=result["winner"])
        temporal_rows.append(frame)
        print(f"CORTE {n}: {result['winner']}", flush=True)
        print(frame[["Modelo", "RMSE", "WAPE_%", "R2"]].to_string(index=False), flush=True)
    pd.concat(temporal_rows).to_csv(out / "validacion_temporal.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
