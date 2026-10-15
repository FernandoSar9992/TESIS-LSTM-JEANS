"""Una ejecución completa predefinida. No reajusta parámetros mirando TEST."""
from pathlib import Path
import json
import pandas as pd
from demand_data import prepare_series
from thesis_engine import ThesisConfig, run_thesis
from thesis_report import export_thesis


def main():
    source = Path(r"C:\Users\ASUS\Downloads\VD\DEMANDA_NUEVA.xlsx")
    out = Path(__file__).resolve().parent / "auditoria_tesis_v4"
    out.mkdir(exist_ok=True)
    raw = pd.read_excel(source)
    prepared = prepare_series(raw, "Etiqueta de Semana", "Demanda")
    config = ThesisConfig()
    result = run_thesis(prepared.data, config, prepared.future_labels(config.horizon, config.periodicity),
                        progress=lambda message: print(message, flush=True))
    result["preparation"] = {"archivo": source.name, "periodo": "Etiqueta de Semana", "demanda": "Demanda"}
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f"{key}.csv", index=False, encoding="utf-8-sig")
    metadata = {k: v for k, v in result.items() if k != "_artifacts" and not isinstance(v, pd.DataFrame)}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "informe_tesis.xlsx").write_bytes(export_thesis(result))
    print(result["metrics"].to_string(index=False), flush=True)
    print(result["diagnostic"].to_string(index=False), flush=True)


def export_existing():
    """Exportar la ejecución archivada sin entrenar ni consultar TEST otra vez."""
    out = Path(__file__).resolve().parent / "auditoria_tesis"
    result = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    for path in out.glob("*.csv"):
        result[path.stem] = pd.read_csv(path, keep_default_na=False)
    (out / "informe_tesis.xlsx").write_bytes(export_thesis(result))
    detail = result["detail"]
    import numpy as np
    error = detail.Demanda_real.to_numpy(float)-detail.Prediccion_LSTM.to_numpy(float)
    row = result["metrics"].set_index("Modelo").loc["LSTM"]
    np.testing.assert_allclose(row.MAE, np.abs(error).mean())
    np.testing.assert_allclose(row.RMSE, np.sqrt(np.mean(error**2)))
    np.testing.assert_allclose(row.R2, 1-np.sum(error**2)/np.sum((detail.Demanda_real-detail.Demanda_real.mean())**2))
    with pd.ExcelFile(out / "informe_tesis.xlsx") as book:
        assert len(book.sheet_names) == 16
        assert len(pd.read_excel(book, "Predicciones")) == len(detail)
    print("Excel exportado; MAE, RMSE, R2 y filas verificados sin reentrenar.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-existing", action="store_true")
    args = parser.parse_args()
    export_existing() if args.export_existing else main()
