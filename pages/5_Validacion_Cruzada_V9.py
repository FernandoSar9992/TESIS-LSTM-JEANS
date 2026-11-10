"""Página opcional de auditoría V9: NO altera los pronósticos de app.py."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from hashlib import sha256
from io import BytesIO
from zoneinfo import ZoneInfo
import hmac
import os

import pandas as pd
import streamlit as st

from motor_lstm import prepare_base, extract_series
from protocolo_final import confirm_pre_sales
from cv_v9_diagnostico import AuditConfig, expanding_folds, evaluate_cross_validation
from ui_tesis import install_theme, hero_audit, note, ribbon, kpis


st.set_page_config(page_title="TESIS | Auditoría V9", page_icon="🧪", layout="wide")
install_theme()
hero_audit()
ribbon("Diagnóstico metodológico", "La auditoría vuelve a entrenar copias temporales de V9. No sustituye la LSTM utilizada en planificación.")
st.info(
    "Auditoría independiente: se entrena una copia de la arquitectura V9 en cada "
    "fold. NO se cambian la V9 original, sus hiperparámetros, sus predicciones "
    "ni los archivos de entrada. Esta página NO consulta el bloque TEST."
)

password = os.environ.get("TESIS_ACCESS_PASSWORD", "").strip()
if password and not hmac.compare_digest(
    st.text_input("Clave de acceso", type="password"), password
):
    st.info("Introduce la clave de acceso.")
    st.stop()

uploaded = st.file_uploader(
    "Cargar el mismo Excel de demanda utilizado por V9",
    type=["xlsx", "xlsm"], key="audit_v9_file"
)
if uploaded is None:
    st.write("La auditoría utiliza la hoja **DEMANDA** del archivo original.")
    st.stop()

try:
    payload = uploaded.getvalue()
    book = pd.ExcelFile(BytesIO(payload))
    if "DEMANDA" not in book.sheet_names:
        raise ValueError("No existe la hoja DEMANDA.")
    base, info = prepare_base(
        pd.read_excel(BytesIO(payload), sheet_name="DEMANDA")
    )
    confirm_zeros = st.checkbox(
        "Confirmar que no hubo ventas de corte propio en las cinco semanas "
        "iniciales sin registro (solo cuando corresponda y esté respaldado)",
        value=False,
    )
    base, zero_trace = confirm_pre_sales(base, confirm_zeros)
except Exception as exc:
    st.error(f"No se pudo preparar la base: {exc}")
    st.stop()

left, middle, right = st.columns(3)
left.metric("Semanas cerradas", info["semanas_completas"])
middle.metric("Fin del histórico", str(info["fecha_ultima_semana"]))
right.metric("SHA256 del archivo", sha256(payload).hexdigest()[:12] + "…")
st.caption("La página no escribe en tu Excel ni modifica ningún valor histórico.")

available = {}
for label, target in (
    ("Demanda de maquila", "demanda_mq"),
    ("Demanda de corte propio", "demanda_cp"),
):
    try:
        available[label] = extract_series(base, target)
    except Exception as exc:
        st.warning(f"{label}: {exc}")

if not available:
    st.stop()

with st.sidebar:
    st.header("Auditoría independiente")
    name = st.selectbox("Serie a evaluar", list(available))
    maximum_folds = st.slider("Máximo de folds", min_value=1, max_value=5, value=3)
    st.caption(
        "Se usa la misma arquitectura V9 (LSTM residual estacional a 12 semanas), "
        "3 semillas, dropout 0,15 y L2 1e-4."
    )
    st.caption("TRAIN 70 %, VALIDATION 15 %, TEST 15 % reservado.")
    st.caption("Se eligen solo folds completos, sin superposición del objetivo.")

series = available[name]
configuration = AuditConfig(max_folds=maximum_folds)
try:
    proposed = expanding_folds(
        len(series), configuration.train_ratio,
        configuration.val_ratio, max_folds=maximum_folds
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.write(
    f"**{name}** · {len(series)} semanas · {len(proposed)} fold(s) "
    "de 12 semanas · TEST intocable."
)
if len(proposed) < 2:
    st.warning(
        "Este histórico solo permite un fold completo con las condiciones "
        "actuales. Se puede realizar el diagnóstico, pero NO afirmar que "
        "se comprobó estabilidad entre varios folds independientes."
    )

st.caption(
    "En cada fold: entrenamiento piloto con EarlyStopping en un bloque "
    "interno; selección de arquitectura con 12 semanas interiores; "
    "reentrenamiento con el prefijo disponible; evaluación externa del "
    "pronóstico fijo de 12 semanas. La salida del siguiente mes no se usa "
    "para pronosticar los pasos intermedios."
)

if st.button(
    "Ejecutar auditoría temporal V9 (sin cambiar pronósticos)",
    type="primary", use_container_width=True
):
    status = st.empty()
    try:
        with st.spinner("Entrenando los folds en una copia temporal de V9…"):
            result = evaluate_cross_validation(
                series, configuration, progress=status.info
            )
        st.session_state["cv_v9_result"] = result
        st.session_state["cv_v9_signature"] = (
            sha256(payload).hexdigest(), name,
            confirm_zeros, maximum_folds
        )
    except Exception as exc:
        st.error(f"La validación no se completó: {type(exc).__name__}: {exc}")
    finally:
        status.empty()

signature = (sha256(payload).hexdigest(), name, confirm_zeros, maximum_folds)
if st.session_state.get("cv_v9_signature") != signature:
    st.info("Ejecuta la auditoría para obtener los resultados de esta base y configuración.")
    st.stop()

result = st.session_state["cv_v9_result"]
st.subheader("Indicadores de generalización")
summary_row = result["summary"].iloc[0]
kpis([
    ("Particiones evaluadas", int(summary_row["Folds"]), "Ventana expansiva sin objetivos solapados"),
    ("RMSE medio", f"{float(summary_row['Promedio_RMSE_LSTM']):.2f}", "Unidades de demanda por semana"),
    ("MAE medio", f"{float(summary_row['Promedio_MAE_LSTM']):.2f}", "Error absoluto medio"),
    ("Dispersión RMSE", f"{float(summary_row['Desviacion_RMSE_folds']):.2f}", "Desviación entre folds"),
    ("Folds favorables", f"{int(summary_row['Folds_superados_por_LSTM'])}/{int(summary_row['Folds'])}", "Mejor RMSE que Naive 48 y 52"),
    ("TEST del diagnóstico", "Reservado", "No interviene en este proceso"),
])
st.subheader("Resultados de cada fold (12 semanas sin actualizar objetivos)")
st.dataframe(result["folds"], hide_index=True, use_container_width=True)
st.subheader("Resumen de estabilidad")
st.dataframe(result["summary"], hide_index=True, use_container_width=True)
st.caption(
    "Un RMSE estable entre folds aporta evidencia de generalización, "
    "pero NO demuestra ausencia de sobreajuste. Compara también con "
    "Naive 48/52 y revisa la calidad REAL/IMPUTADO de los objetivos."
)

st.subheader("Curvas loss/val_loss de entrenamiento piloto")
curves = result["curves"]
fold = st.selectbox("Fold", sorted(curves["Fold"].unique()))
models = curves.loc[curves["Fold"] == fold, ["Ventana", "Unidades"]].drop_duplicates()
model_options = [(int(r.Ventana), int(r.Unidades)) for r in models.itertuples()]
model = st.selectbox(
    "Arquitectura evaluada", model_options,
    format_func=lambda v: f"ventana {v[0]} · unidades {v[1]}"
)
seed = st.selectbox(
    "Semilla", sorted(curves.loc[
        (curves.Fold == fold) &
        (curves.Ventana == model[0]) &
        (curves.Unidades == model[1]), "Semilla"
    ].unique())
)
filtered = curves.loc[
    (curves.Fold == fold) &
    (curves.Ventana == model[0]) &
    (curves.Unidades == model[1]) &
    (curves.Semilla == seed)
]
st.line_chart(
    filtered.set_index("Época")[["loss_train", "loss_validacion_interna"]],
    use_container_width=True
)
st.caption(
    f"Mejor época de este piloto: {int(filtered['Época_óptima'].iloc[0])}. "
    "La curva utiliza la pérdida Huber con regularización y es solo "
    "una herramienta diagnóstica del fold."
)

st.subheader("Predicciones externas y selección anidada")
st.dataframe(result["predictions"], hide_index=True, use_container_width=True)
with st.expander("Configuraciones candidatas (evaluación interna)"):
    st.dataframe(result["candidates"], hide_index=True, use_container_width=True)

output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    for sheet, key in (
        ("Resumen", "summary"),
        ("Folds", "folds"),
        ("Predicciones", "predictions"),
        ("Candidatas", "candidates"),
        ("Curvas", "curves"),
    ):
        result[key].to_excel(writer, sheet_name=sheet, index=False)
st.download_button(
    "Descargar auditoría para anexos (Excel)",
    data=output.getvalue(),
    file_name=f"CV_V9_{'MQ' if name == 'Demanda de maquila' else 'CP'}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)
st.warning(
    "IMPORTANTE: estas métricas son históricas y pueden incluir semanas "
    "reconstruidas. Para afirmar precisión sobre demanda observada, "
    "es indispensable filtrar objetivos con evidencia real y disponer "
    "de nuevas semanas externas a las ya utilizadas en el desarrollo."
)
