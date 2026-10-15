"""Diagnóstico reproducible del archivo indicado por el usuario, sin modificarlo."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from demand_data import prepare_series
from lstm_engine import LSTMConfig, run_lstm, split_points, suggest_cycle

SOURCE = Path(r"C:\Users\ASUS\Downloads\VD\DEMANDA_NUEVA.xlsx")
OUT = Path(__file__).resolve().parent / "diagnostico_demanda_nueva"


def main():
    OUT.mkdir(exist_ok=True)
    raw = pd.read_excel(SOURCE)
    prepared = prepare_series(raw, raw.columns[0], raw.columns[1])
    data = prepared.data
    y = data.demanda.to_numpy()
    tr, va = split_points(len(y), 8)
    stats = {"n": len(y), "train_end": tr, "val_end": va,
             "duplicados": int(data.periodo.duplicated().sum()),
             "ciclo_sugerido_entrenamiento": suggest_cycle(y[:tr]),
             "bloques": {name: {"media": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
                         for name, v in (("train", y[:tr]), ("val", y[tr:va]), ("test", y[va:]))},
             "autocorrelaciones_train": {lag: float(pd.Series(y[:tr]).autocorr(lag)) for lag in (1, 4, 8, 12, 24, 48)},
             "autocorrelaciones_total_descriptivas": {lag: float(pd.Series(y).autocorr(lag)) for lag in (1, 4, 8, 12, 24, 48)}}
    (OUT / "datos_diagnostico.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(data.to_string(index=True), flush=True)
    result = run_lstm(data, LSTMConfig(), prepared.future_labels(8), progress=lambda s: print(s, flush=True))
    for key in ("metrics", "search", "forecast", "detail", "horizon_metrics"):
        result[key].to_csv(OUT / f"{key}.csv", index=False, encoding="utf-8-sig")
    meta = {k: result[k] for k in ("config", "selected", "history", "notes", "train_end", "val_end", "test_origins", "data_hash")}
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["metrics"].to_string(index=False), flush=True)
    print(result["search"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
