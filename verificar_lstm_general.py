"""Entrenamientos reales para comprobar distintos comportamientos, sin retocar por prueba."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pandas as pd

from demand_data import prepare_series
from lstm_engine import LSTMConfig, run_lstm

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT / "auditoria_lstm_general"
    out.mkdir(exist_ok=True)
    raw = pd.read_excel(ROOT / "Cronograma_Semanas_2023_2026_v24.xlsx")
    reference = prepare_series(raw, raw.columns[0], raw.columns[1])
    rng = np.random.default_rng(2026)
    t = np.arange(180)
    scenarios = {
        "excel_referencia": (reference, LSTMConfig()),
        "tendencia": (20+.6*t+rng.normal(0, 1, len(t)), LSTMConfig(horizon=6, epochs=60, members=1)),
        "estacional": (100+30*np.sin(2*np.pi*t/12)+rng.normal(0, 1, len(t)), LSTMConfig(horizon=6, epochs=60, members=1)),
        "intermitente": ((rng.random(len(t)) < .2)*rng.integers(10, 80, len(t)), LSTMConfig(horizon=6, epochs=60, members=1)),
        "constante": (np.full(120, 25.), LSTMConfig(horizon=4, automatic=False, lookback=12, epochs=3, patience=2, members=1)),
        "ceros": (np.zeros(120), LSTMConfig(horizon=4, automatic=False, lookback=12, epochs=3, patience=2, members=1)),
    }
    rows = []
    for name, (source, config) in scenarios.items():
        print(f"INICIO {name}", flush=True)
        if isinstance(source, np.ndarray):
            frame = pd.DataFrame({"fecha": pd.date_range("2020-01-01", periods=len(source), freq="D"), "unidades": source})
            source = prepare_series(frame, "fecha", "unidades", date_mode=True)
        result = run_lstm(source.data, config, source.future_labels(config.horizon), progress=lambda msg: print(msg, flush=True))
        assert np.isfinite(result["forecast"]["LSTM"]).all()
        assert (result["forecast"]["LSTM"] >= 0).all()
        assert len(result["forecast"]) == config.horizon
        if name in ("constante", "ceros"):
            np.testing.assert_allclose(result["forecast"]["LSTM"], source.data.demanda.iloc[-1], atol=1e-5)
        case = out / name
        case.mkdir(exist_ok=True)
        for key in ("metrics", "search", "forecast", "detail", "horizon_metrics"):
            result[key].to_csv(case / f"{key}.csv", index=False, encoding="utf-8-sig")
        metadata = {key: result[key] for key in ("config", "selected", "history", "notes", "train_end", "val_end", "test_origins", "data_hash")}
        (case / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"Caso": name, **result["metrics"].iloc[0].to_dict(), **result["selected"], "members": config.members})
        pd.DataFrame(rows).to_csv(out / "resumen.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame([rows[-1]]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
