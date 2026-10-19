"""Pronóstico de demanda: dos redes LSTM independientes."""
from io import BytesIO
from hashlib import sha256
import numpy as np
import pandas as pd
import streamlit as st
from forecast_engine import period_positions, future_labels
from thesis_engine import ThesisConfig, run_thesis, finalize_test

st.set_page_config(page_title="Pronóstico de demanda",page_icon="📈",layout="wide")
st.title("Pronóstico de demanda de pantalones jeans")
st.write("Modelos LSTM para maquila y corte propio.")

uploaded=st.file_uploader("Excel de demanda", type=["xlsx"])
if uploaded is None:
    st.info("Carga el archivo de demanda de una hoja para comenzar.")
    st.stop()
try:
    frame=pd.read_excel(BytesIO(uploaded.getvalue()),sheet_name=0,header=3)
    cols=["Periodo (4/mes)","DEMANDA MAQUILA LSTM","DEMANDA CORTE LSTM"]
    if any(c not in frame.columns for c in cols):
        raise ValueError("Faltan columnas del archivo de demanda.")
    frame=frame.dropna(subset=[cols[0]]).reset_index(drop=True)
    periods=frame[cols[0]].astype(str).str.strip()
    positions=period_positions(periods)
    if positions is None or not np.all(np.diff(positions)==1):
        raise ValueError("Los períodos no son consecutivos.")
    series={}
    for title,col in (("Maquila",cols[1]),("Corte propio",cols[2])):
        x=pd.to_numeric(frame[col],errors="coerce")
        start=0
        if title=="Corte propio":
            found=np.flatnonzero(x.notna().to_numpy())
            if not len(found):
                raise ValueError("No hay demanda de corte propio.")
            start=int(found[0])
        x=x.iloc[start:]
        if not np.isfinite(x).all() or (x<0).any():
            raise ValueError(f"{title}: faltan valores o hay demandas inválidas.")
        series[title]=pd.DataFrame({
            "periodo":periods.iloc[start:].to_numpy(),
            "demanda":x.to_numpy(float),
            "origen":"SIN_DECLARAR",
        })
except Exception as exc:
    st.error(f"No se pudo cargar el Excel: {exc}")
    st.stop()

st.success(f"Histórico cargado hasta {periods.iloc[-1]}.")
st.caption("Los resultados reflejan el conjunto cargado; si se utiliza una base sintética, "
           "las proyecciones son escenarios de planificación, no pedidos reales comprobados.")

with st.sidebar:
    st.header("Configuración")
    h=st.number_input("Períodos futuros",1,24,12)
    epochs=st.number_input("Épocas máximas",40,500,120,step=20)
    seed=st.number_input("Semilla",0,2147483647,42)

signature=sha256(uploaded.getvalue()+f"{h}-{epochs}-{seed}".encode()).hexdigest()
if st.session_state.get("firma_pronostico")!=signature:
    st.session_state["firma_pronostico"]=signature
    st.session_state.pop("resultado_pronostico",None)

if st.button("Generar pronóstico",type="primary"):
    config=ThesisConfig(horizon=int(h),epochs=int(epochs),seed=int(seed),
                        periodicity=48,automatic=True,seasonal_features=True,
                        calendar_experiment=True)
    results={}
    try:
        for title,s in series.items():
            with st.spinner(f"Calculando pronóstico de {title}…"):
                labels=future_labels(s,int(h))
                r=run_thesis(s,config,future_periods=labels,evaluate_test=False)
                r=finalize_test(r)
                results[title]=r["forecast"][["Periodo","Horizonte","LSTM"]].copy()
        left=results["Maquila"].rename(columns={"LSTM":"Maquila"})
        right=results["Corte propio"].rename(columns={"LSTM":"Corte propio"})
        both=left.merge(right,on=["Periodo","Horizonte"],validate="one_to_one")
        both["Total"]=both["Maquila"]+both["Corte propio"]
        st.session_state["resultado_pronostico"]=both
    except Exception as exc:
        st.error(f"No se pudo generar el pronóstico: {exc}")

out=st.session_state.get("resultado_pronostico")
if out is not None:
    st.subheader("Demanda pronosticada")
    view=out.copy()
    for col in ("Maquila","Corte propio","Total"):
        view[col]=view[col].round().astype(int)
    st.dataframe(view,hide_index=True,use_container_width=True)
    st.line_chart(out.set_index("Periodo")[["Maquila","Corte propio"]])
    st.caption("Los totales se calculan con decimales y se redondean al presentar.")
    st.download_button("Descargar pronósticos (CSV)",
                       out.to_csv(index=False,sep=";").encode("utf-8-sig"),
                       file_name="PRONOSTICO_LSTM.csv",mime="text/csv")
