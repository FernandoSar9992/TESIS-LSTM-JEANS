"""Pronóstico de demanda: dos redes LSTM independientes."""
from io import BytesIO
from hashlib import sha256
import numpy as np
import pandas as pd
import streamlit as st
from forecast_engine import metrics
from weekly_calendar import load_demand_excel
from thesis_engine import ThesisConfig, run_thesis, finalize_test

st.set_page_config(page_title="Pronóstico de demanda",page_icon="📈",layout="wide")
st.title("Pronóstico de demanda de pantalones jeans")
st.write("Modelos LSTM para maquila y corte propio.")

uploaded=st.file_uploader("Excel de demanda", type=["xlsx"])
if uploaded is None:
    st.info("Carga el Excel conciliado de demanda para comenzar.")
    st.stop()

# Nunca usa semanas en curso o futuras como observaciones históricas.
today=pd.Timestamp.now(tz="America/Guayaquil").date()
cutoff=st.date_input("Fecha de corte de datos confirmados",value=today,max_value=today,
                     help="Solo se entrenará con semanas lunes–viernes finalizadas antes de este día.")
try:
    series,future_builder,metadata=load_demand_excel(uploaded.getvalue(),cutoff)
except Exception as exc:
    st.error(f"No se pudo cargar el Excel: {exc}")
    st.stop()

st.success(f"Histórico válido hasta {metadata['last_period']} "
           f"({metadata['n_historical']} períodos).")
st.caption(f"Calendario: {metadata['calendar']}. Semanas excluidas por fecha "
           f"de corte: {metadata['excluded']}. Última fecha finalizada: "
           f"{metadata['last_complete_date']}.")
st.info("La producción registrada de corte propio se conserva como referencia "
        "operativa, pero NO se suma a la demanda usada para entrenar las LSTM. "
        "El siguiente módulo la utilizará para planificar la fabricación.")
st.caption("Los datos sin trazabilidad documentada no pueden considerarse "
           "demanda real verificada. Las métricas sobre datos estimados o "
           "sintéticos son resultados experimentales.")


with st.sidebar:
    st.header("Configuración")
    h=st.number_input("Períodos futuros",1,24,12)
    epochs=st.number_input("Épocas máximas",40,500,120,step=20)
    seed=st.number_input("Semilla",0,2147483647,42)

signature=sha256(uploaded.getvalue()+f"{h}-{epochs}-{seed}-{cutoff}".encode()).hexdigest()
if st.session_state.get("firma_pronostico")!=signature:
    st.session_state["firma_pronostico"]=signature
    st.session_state.pop("resultado_pronostico",None)
    st.session_state.pop("metricas_finales",None)

if st.button("Generar pronóstico",type="primary"):
    st.session_state.pop("resultado_pronostico",None)
    st.session_state.pop("metricas_finales",None)
    config=ThesisConfig(horizon=int(h),epochs=int(epochs),seed=int(seed),
                        periodicity=metadata["periodicity"],automatic=True,
                        seasonal_features=True,calendar_experiment=False)
    results={}
    test_details={}
    test_metrics={}
    try:
        for title,s in series.items():
            with st.spinner(f"Calculando pronóstico de {title}…"):
                labels=future_builder(int(h))
                r=run_thesis(s,config,future_periods=labels,evaluate_test=False)
                r=finalize_test(r)
                results[title]=r["forecast"][["Periodo","Horizonte","LSTM"]].copy()
                test_details[title]=r["detail"][["Periodo","Demanda_real","Prediccion_LSTM"]].copy()
                score=r["metrics"].loc[r["metrics"]["Modelo"].eq("LSTM")]
                if len(score)!=1:
                    raise ValueError(f"{title}: no se encontró una métrica TEST única para LSTM.")
                test_metrics[title]=score.iloc[0]
        left=results["Maquila"].rename(columns={"LSTM":"Maquila"})
        right=results["Corte propio"].rename(columns={"LSTM":"Corte propio"})
        both=left.merge(right,on=["Periodo","Horizonte"],validate="one_to_one")
        both["Total"]=both["Maquila"]+both["Corte propio"]
        score_rows=[]
        for title in ("Maquila","Corte propio"):
            metric=test_metrics[title]
            score_rows.append({
                "Modalidad":title, "Períodos evaluados":int(metric["N"]),
                "R²":float(metric["R2"]), "MAE":float(metric["MAE"]),
                "RMSE":float(metric["RMSE"]), "MAPE (%)":float(metric["MAPE_%"]),
            })
        left_test=test_details["Maquila"].rename(columns={
            "Demanda_real":"Real_maquila", "Prediccion_LSTM":"Pred_maquila"
        })
        right_test=test_details["Corte propio"].rename(columns={
            "Demanda_real":"Real_corte", "Prediccion_LSTM":"Pred_corte"
        })
        common_test=left_test.merge(right_test,on="Periodo",how="inner",validate="one_to_one")
        if len(common_test)>=2:
            real_total=common_test["Real_maquila"].to_numpy()+common_test["Real_corte"].to_numpy()
            pred_total=common_test["Pred_maquila"].to_numpy()+common_test["Pred_corte"].to_numpy()
            total_score=metrics(real_total,pred_total)
            score_rows.append({
                "Modalidad":"Total (períodos compartidos)", "Períodos evaluados":len(common_test),
                "R²":total_score["R2"],"MAE":total_score["MAE"],
                "RMSE":total_score["RMSE"],"MAPE (%)":total_score["MAPE_%"],
            })
        st.session_state["resultado_pronostico"]=both
        st.session_state["metricas_finales"]=pd.DataFrame(score_rows)
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

    st.divider()
    st.subheader("Métricas finales de evaluación (TEST)")
    scores=st.session_state.get("metricas_finales")
    if scores is not None:
        st.dataframe(scores.round(3),hide_index=True,use_container_width=True)
        st.caption("R², MAE, RMSE y MAPE evalúan predicciones de UN período "
                   "sobre datos reservados de TEST; no miden la exactitud de "
                   "los períodos futuros, cuyos valores aún se desconocen.")
        st.info("La hoja cargada no aporta trazabilidad verificable de pedidos. "
                "Si es sintética, estas métricas son experimentales y no "
                "constituyen validación frente a la demanda real del taller. "
                "La fila Total utiliza solo los períodos TEST comunes.")
