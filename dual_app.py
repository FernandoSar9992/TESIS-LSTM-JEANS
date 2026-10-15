"""Interfaz de pronóstico dual para la tesis. Ejecutar: streamlit run dual_app.py"""
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dual_forecast import prepare_dual, combined_forecast, combined_test
from forecast_engine import future_labels
from thesis_engine import ThesisConfig, run_thesis, finalize_test

st.set_page_config(page_title="LSTM | Maquila y corte propio", page_icon="📈", layout="wide")
st.title("Pronóstico LSTM: maquila y corte propio")
st.caption("Dos redes independientes · Total = suma · 48 períodos por año · Procedencia auditable")

uploaded = st.file_uploader("Excel de demanda del taller", type=["xlsx"])
if uploaded is None:
    st.info("Carga DEMANDA_JEANS_TESIS.xlsx y selecciona la hoja 01_DEMANDA_LSTM.")
    st.stop()
content = uploaded.getvalue()
try:
    excel = pd.ExcelFile(BytesIO(content))
    sheet = st.selectbox("Hoja de demanda", excel.sheet_names,
                         index=excel.sheet_names.index("01_DEMANDA_LSTM") if "01_DEMANDA_LSTM" in excel.sheet_names else 0)
    raw = pd.read_excel(BytesIO(content), sheet_name=sheet, header=4)
    series = prepare_dual(raw)
except Exception as exc:
    st.error(f"No se pudo preparar la base dual: {exc}")
    st.stop()

a, b = series["Maquila"], series["Corte propio"]
n = len(a)
st.success(f"Base alineada: {n} períodos desde {a.periodo.iloc[0]} hasta {a.periodo.iloc[-1]}.")
c1, c2, c3 = st.columns(3)
c1.metric("Maquila: objetivos simulados", int(a.origen.eq("IMPUTADO").sum()))
c2.metric("Corte propio: objetivos mixtos", int(b.origen.eq("MIXTO").sum()))
c3.metric("Ambas modalidades REAL", int((a.origen.eq("REAL") & b.origen.eq("REAL")).sum()))
st.warning(
    "La maquila histórica incluye escenarios reconstruidos y el corte propio mezcla ventas/pedidos "
    "documentados con períodos incompletos. Un escenario puede servir para probar código y planificación, "
    "pero NO acredita precisión predictiva sobre demanda real. "
    "Los objetivos mixtos NO se contabilizan como observaciones totalmente reales."
)

with st.expander("Datos utilizados y trazabilidad", expanded=False):
    preview = pd.DataFrame({
        "Periodo": a.periodo, "Demanda maquila": a.demanda,
        "Origen maquila": a.origen, "Demanda corte": b.demanda,
        "Origen corte": b.origen, "Demanda total": a.demanda + b.demanda,
    })
    st.dataframe(preview, hide_index=True, use_container_width=True)
    st.plotly_chart(px.line(preview, x="Periodo", y=["Demanda maquila", "Demanda corte"],
                            title="Demanda de escenario por modalidad"),
                    use_container_width=True)

with st.sidebar:
    st.header("Configuración LSTM")
    horizon = st.number_input("Períodos futuros (4 por mes)", 1, 24, 8)
    auto = st.checkbox("Optimización por validación temporal", value=True)
    lookback = st.number_input("Ventana histórica", 2, 24, 8)
    epochs = st.number_input("Épocas máximas", 20, 500, 120, step=20)
    units = st.selectbox("Unidades de la red", [8, 16, 24, 32], index=1)
    seed = st.number_input("Semilla", 0, 2147483647, 42)
    st.caption("TRAIN, validación y TEST comparten el mismo corte temporal por modalidad.")
    train = st.slider("TRAIN (%)", 60, 80, 70)
    val = st.slider("Validación (%)", 10, 25, 15)
    if train + val > 95:
        st.error("Deja al menos 5 % para TEST.")
        st.stop()

config = ThesisConfig(horizon=int(horizon), automatic=auto, lookback=int(lookback), units=int(units),
                      epochs=int(epochs), periodicity=48, seed=int(seed),
                      train_ratio=train/100, validation_ratio=val/100)
signature = sha256(content + str(sheet).encode() + repr(asdict(config)).encode()).hexdigest()
if st.session_state.get("dual_signature") != signature:
    st.session_state["dual_signature"] = signature
    st.session_state.pop("dual_results", None)
    st.session_state.pop("dual_finalized", None)
    st.session_state.pop("dual_test_locked", None)

if st.button("1. Entrenar ambas LSTM y revisar validación", type="primary"):
    st.session_state.pop("dual_results", None)
    st.session_state.pop("dual_finalized", None)
    try:
        outputs = {}
        for name, data in series.items():
            with st.spinner(f"Entrenando {name}; reservando TEST…"):
                labels = future_labels(data, config.horizon)
                outputs[name] = run_thesis(data, config, future_periods=labels, evaluate_test=False)
        st.session_state["dual_results"] = outputs
        st.success("Ambas redes se entrenaron. Revisa la validación antes de evaluar TEST.")
    except Exception as exc:
        st.error(f"Falló el entrenamiento: {exc}")
        st.stop()

results = st.session_state.get("dual_results")
if results:
    st.subheader("Validación temporal (para seleccionar configuración)")
    vrows = []
    for name, result in results.items():
        frame = result["split_metrics"]
        for _, line in frame.loc[(frame.Grupo == "VALIDATION") & (frame.Modelo == "LSTM")].iterrows():
            vrows.append({"Modalidad": name, **line.to_dict()})
    if vrows:
        st.dataframe(pd.DataFrame(vrows), hide_index=True, use_container_width=True)
    st.caption("No selecciones parámetros mirando TEST. La validación se usa para selección y no es independiente.")

    if not st.session_state.get("dual_finalized"):
        if st.session_state.get("dual_test_locked"):
            st.warning("Este TEST ya fue abierto en esta sesión. Para una evaluación independiente, reserva nuevos datos.")
        elif st.button("2. Evaluar TEST final y pronosticar"):
            # Bloqueo antes de calcular: si falla a medias, evitar reutilizar el mismo TEST.
            st.session_state["dual_test_locked"] = True
            try:
                finalized = {}
                for name, result in results.items():
                    with st.spinner(f"Evaluación final y pronóstico: {name}…"):
                        finalized[name] = finalize_test(result)
                st.session_state["dual_results"] = finalized
                st.session_state["dual_finalized"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Falló la evaluación final: {exc}")
                st.stop()

if st.session_state.get("dual_finalized"):
    ares, bres = st.session_state["dual_results"]["Maquila"], st.session_state["dual_results"]["Corte propio"]
    try:
        future = combined_forecast(ares["forecast"], bres["forecast"])
        comparison, total_metrics, observed_metrics, observed_n = combined_test(ares["detail"], bres["detail"])
    except Exception as exc:
        st.error(f"Fallo de conciliación: {exc}")
        st.stop()
    st.subheader("Pronósticos futuros LSTM")
    st.dataframe(future.round(2), hide_index=True, use_container_width=True)
    graph = future.melt(id_vars="Periodo", value_vars=["Maquila", "Corte propio", "Total"],
                        var_name="Modalidad", value_name="Unidades")
    st.plotly_chart(px.line(graph, x="Periodo", y="Unidades", color="Modalidad", markers=True,
                            title="Pronósticos separados y demanda total"), use_container_width=True)
    st.caption("Los resultados pueden ser decimales; para planificar unidades enteras, aplica una política de redondeo después de la suma.")

    st.subheader("Evaluación TEST por modalidad")
    rows = []
    for name, result in st.session_state["dual_results"].items():
        for _, metric in result["metrics_all"].loc[result["metrics_all"].Modelo == "LSTM"].iterrows():
            rows.append({"Modalidad": name, "Alcance": "Todos los objetivos (incluye escenarios)", **metric.to_dict()})
        real = result["metrics_real"]
        if not real.empty:
            for _, metric in real.loc[real.Modelo == "LSTM"].iterrows():
                rows.append({"Modalidad": name, "Alcance": "Solo objetivos REAL", **metric.to_dict()})
    rows.append({"Modalidad": "Total", "Alcance": "Todos los objetivos (incluye escenarios)",
                 "Modelo": "Suma LSTM", **total_metrics})
    if observed_metrics is not None:
        rows.append({"Modalidad": "Total", "Alcance": f"Solo ambas modalidades REAL (N={observed_n})",
                     "Modelo": "Suma LSTM", **observed_metrics})
    else:
        st.warning("No hay al menos dos períodos de TEST con ambas demandas totalmente observadas; "
                   "las métricas del total son DESCRIPTIVAS, no validación sobre demanda real.")
    scores = pd.DataFrame(rows)
    st.dataframe(scores, hide_index=True, use_container_width=True)
    st.subheader("Prueba final del total")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.subheader("Exportar informe")
    st.caption("Incluye pronóstico, prueba conjunta, métricas y fuente de los datos sin convertir simulaciones en registros reales.")
    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            future.to_excel(writer, sheet_name="Pronosticos", index=False)
            scores.to_excel(writer, sheet_name="Metricas", index=False)
            comparison.to_excel(writer, sheet_name="Prueba_total", index=False)
            preview.to_excel(writer, sheet_name="Trazabilidad", index=False)
        payload = buffer.getvalue()
    st.download_button("Descargar resultados duales (.xlsx)", data=payload,
                       file_name="PRONOSTICO_DUAL_LSTM_TESIS.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
