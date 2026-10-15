"""Evaluación reproducible de demanda semanal para una serie corta.

Usa validación cronológica, compara modelos base y una regresión Ridge
directa de 8 salidas. No requiere TensorFlow ni reutiliza información futura.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
HORIZONTE = 8
LOOKBACKS = (4, 8, 12, 16, 26, 52)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
ROOT = Path(__file__).resolve().parent
ARCHIVO = ROOT / "Cronograma_Semanas_2023_2026_v24.xlsx"
SALIDA = ROOT / "resultados_intermedios"


def metricas(real, pred):
    real, pred = np.asarray(real, float).ravel(), np.asarray(pred, float).ravel()
    err = real - pred
    mse = np.mean(err**2)
    denom = np.sum(np.abs(real))
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err / real)) * 100) if np.all(real != 0) else np.nan,
        "WAPE": float(np.sum(np.abs(err)) / denom * 100) if denom else np.nan,
        "sMAPE": float(np.mean(2 * np.abs(err) / (np.abs(real) + np.abs(pred))) * 100),
        "MSE": float(mse),
        "R2": float(1 - np.sum(err**2) / ss_tot) if ss_tot else np.nan,
    }


def matriz_directa(y, lookback, horizonte=HORIZONTE):
    xs, ys, origins = [], [], []
    for t in range(lookback, len(y) - horizonte + 1):
        lag = y[t - lookback:t]
        idx = np.arange(t - lookback, t)
        features = np.column_stack((lag, np.sin(2*np.pi*idx/52), np.cos(2*np.pi*idx/52))).ravel()
        xs.append(features)
        ys.append(y[t:t+horizonte])
        origins.append(t)
    return np.asarray(xs), np.asarray(ys), np.asarray(origins)


def ridge_fit(x, y, alpha):
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    z = (x - mu) / sd
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mu, sd, coef


def ridge_predict(model, x):
    mu, sd, coef = model
    z = (x - mu) / sd
    return np.column_stack((np.ones(len(z)), z)) @ coef


def futuros_etiquetas(ultima, cantidad=8):
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    partes = ultima.split("-Se")
    mes, anio = partes[0].rsplit(" ", 1)
    m, a, s = meses.index(mes), int(anio), int(partes[1])
    salida = []
    for _ in range(cantidad):
        s += 1
        if s == 5:
            s = 1; m += 1
            if m == 12: m = 0; a += 1
        salida.append(f"{meses[m]} {a}-Se{s}")
    return salida


df = pd.read_excel(ARCHIVO).rename(columns={"Etiqueta de Semana": "Semana", "Demanda": "Demanda"})
df["Demanda"] = pd.to_numeric(df["Demanda"], errors="coerce")
df = df.dropna(subset=["Demanda"]).reset_index(drop=True)
y = df["Demanda"].to_numpy(float)
n = len(y)
fin_train, fin_val = int(n * 0.70), int(n * 0.85)

busqueda = []
mejor = None
for lookback in LOOKBACKS:
    x, target, origins = matriz_directa(y, lookback)
    train = origins + HORIZONTE - 1 < fin_train
    val = (origins >= fin_train) & (origins + HORIZONTE - 1 < fin_val)
    if train.sum() < 20 or val.sum() < 2:
        continue
    for alpha in ALPHAS:
        model = ridge_fit(x[train], target[train], alpha)
        pred = np.maximum(ridge_predict(model, x[val]), 0)
        score = metricas(target[val], pred)
        row = {"Modelo": "Ridge directa", "Lookback": lookback, "Alpha": alpha, **score,
               "Muestras_train": int(train.sum()), "Origenes_validacion": int(val.sum())}
        busqueda.append(row)
        if mejor is None or score["RMSE"] < mejor[0]:
            mejor = (score["RMSE"], lookback, alpha)

_, lookback, alpha = mejor
x, target, origins = matriz_directa(y, lookback)
train_val = origins + HORIZONTE - 1 < fin_val
test = (origins >= fin_val) & (origins + HORIZONTE - 1 < n)
modelo = ridge_fit(x[train_val], target[train_val], alpha)
pred_ridge = np.maximum(ridge_predict(modelo, x[test]), 0)
real_test, origen_test = target[test], origins[test]
pred_ultimo = np.asarray([np.repeat(y[t-1], HORIZONTE) for t in origen_test])
pred_media4 = np.asarray([np.repeat(y[t-4:t].mean(), HORIZONTE) for t in origen_test])
pred_estacional = np.asarray([[y[t+h-52] for h in range(HORIZONTE)] for t in origen_test])

predicciones = {
    "Ridge directa": pred_ridge,
    "Último valor": pred_ultimo,
    "Media móvil 4": pred_media4,
    "Estacional ingenuo 52": pred_estacional,
}
resumen = []
por_horizonte = []
detalle = []
for nombre, pred in predicciones.items():
    resumen.append({"Modelo": nombre, **metricas(real_test, pred)})
    for h in range(HORIZONTE):
        por_horizonte.append({"Modelo": nombre, "Horizonte": h+1, **metricas(real_test[:, h], pred[:, h])})
for i, t in enumerate(origen_test):
    for h in range(HORIZONTE):
        detalle.append({"Origen": int(t), "Semana": df.loc[t+h, "Semana"], "Horizonte": h+1,
                        "Real": real_test[i, h], **{k: v[i, h] for k, v in predicciones.items()}})

resumen_df = pd.DataFrame(resumen).sort_values("RMSE").reset_index(drop=True)
ganador = resumen_df.loc[0, "Modelo"]

# Pronóstico final: se reentrena Ridge; si un base ganó test, se usa ese método.
x_all, target_all, origins_all = matriz_directa(y, lookback)
modelo_final = ridge_fit(x_all, target_all, alpha)
t = n
lag = y[t-lookback:t]
idx = np.arange(t-lookback, t)
x_future = np.column_stack((lag, np.sin(2*np.pi*idx/52), np.cos(2*np.pi*idx/52))).ravel()[None, :]
futuros = {
    "Ridge directa": np.maximum(ridge_predict(modelo_final, x_future)[0], 0),
    "Último valor": np.repeat(y[-1], HORIZONTE),
    "Media móvil 4": np.repeat(y[-4:].mean(), HORIZONTE),
    "Estacional ingenuo 52": np.asarray([y[n+h-52] for h in range(HORIZONTE)]),
}

# Intervalo aproximado calibrado con errores absolutos del bloque de prueba.
pred_ganador_test = predicciones[ganador]
q = np.quantile(np.abs(real_test - pred_ganador_test), 0.95, axis=0, method="higher")
forecast = futuros[ganador]
pronostico = pd.DataFrame({"Semana": futuros_etiquetas(str(df.iloc[-1]["Semana"])),
                           "Horizonte": np.arange(1, HORIZONTE+1), "Modelo": ganador,
                           "Pronostico": forecast, "Limite_inferior_95_aprox": np.maximum(forecast-q, 0),
                           "Limite_superior_95_aprox": forecast+q})

SALIDA.mkdir(exist_ok=True)
df.to_csv(SALIDA / "datos_limpios.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(busqueda).sort_values("RMSE").to_csv(SALIDA / "busqueda_ridge.csv", index=False, encoding="utf-8-sig")
resumen_df.to_csv(SALIDA / "metricas_modelos.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(por_horizonte).to_csv(SALIDA / "metricas_horizonte.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(detalle).to_csv(SALIDA / "evaluacion_test.csv", index=False, encoding="utf-8-sig")
pronostico.to_csv(SALIDA / "pronostico_8_semanas.csv", index=False, encoding="utf-8-sig")
metadata = {"n": n, "fin_train": fin_train, "fin_validacion": fin_val, "origenes_test": int(test.sum()),
            "lookback_ridge": lookback, "alpha_ridge": alpha, "ganador_test": ganador}
(SALIDA / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
bundle = {
    "metadata": metadata,
    "datos": df.to_dict(orient="records"),
    "busqueda": pd.DataFrame(busqueda).sort_values("RMSE").to_dict(orient="records"),
    "metricas": resumen_df.to_dict(orient="records"),
    "horizonte": pd.DataFrame(por_horizonte).to_dict(orient="records"),
    "detalle": pd.DataFrame(detalle).to_dict(orient="records"),
    "pronostico": pronostico.to_dict(orient="records"),
}
(SALIDA / "analisis_bundle.json").write_text(
    json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(metadata, ensure_ascii=False, indent=2))
print(resumen_df.to_string(index=False))
print(pronostico.to_string(index=False))
