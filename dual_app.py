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
    st.session_state.pop("metricas_etapas",None)
    st.session_state.pop("comparacion_referencias",None)
    st.session_state.pop("configuraciones_elegidas",None)
    st.session_state.pop("decisiones_pronostico",None)

if st.button("Generar pronóstico",type="primary"):
    st.session_state.pop("resultado_pronostico",None)
    st.session_state.pop("metricas_finales",None)
    st.session_state.pop("metricas_etapas",None)
    st.session_state.pop("comparacion_referencias",None)
    st.session_state.pop("configuraciones_elegidas",None)
    st.session_state.pop("decisiones_pronostico",None)
    config=ThesisConfig(horizon=int(h),epochs=int(epochs),seed=int(seed),
                        periodicity=metadata["periodicity"],automatic=True,
                        seasonal_features=True,calendar_experiment=False,
                        adaptive_search=True,recent_validation_weight=True)
    results={}
    test_details={}
    test_metrics={}
    comparisons=[]
    chosen_configs=[]
    decision_rows=[]
    stage_details={}
    stage_rows=[]
    try:
        for title,s in series.items():
            with st.spinner(f"Calculando pronóstico de {title}…"):
                labels=future_builder(int(h))
                r=run_thesis(s,config,future_periods=labels,evaluate_test=False)
                r=finalize_test(r)
                results[title]=r["forecast"][["Periodo","Horizonte","LSTM","Propuesta"]].copy()
                test_details[title]=r["detail"][["Periodo","Demanda_real","Prediccion_LSTM"]].copy()
                stage_details[title]={
                    "TRAIN":r["train_detail"][["Periodo","Real","LSTM"]].copy(),
                    "VALIDATION":r["validation_detail"][["Periodo","Real","LSTM"]].copy(),
                    "TEST":r["detail"][["Periodo","Demanda_real","Prediccion_LSTM"]].rename(
                        columns={"Demanda_real":"Real","Prediccion_LSTM":"LSTM"}).copy(),
                }
                for stage,detail in stage_details[title].items():
                    stage_score=metrics(detail["Real"].to_numpy(float),detail["LSTM"].to_numpy(float))
                    stage_rows.append({
                        "Etapa":stage,"Modalidad":title,"Períodos evaluados":len(detail),
                        "R²":stage_score["R2"],"MAE":stage_score["MAE"],
                        "RMSE":stage_score["RMSE"],"MAPE (%)":stage_score["MAPE_%"],
                    })
                score=r["metrics"].loc[r["metrics"]["Modelo"].eq("LSTM")]
                if len(score)!=1:
                    raise ValueError(f"{title}: no se encontró una métrica TEST única para LSTM.")
                test_metrics[title]=score.iloc[0]
                val=r["split_metrics"]
                val=val.loc[(val["Grupo"]=="VALIDATION") & val["Modelo"].ne("LSTM")]
                baseline=val.sort_values(["RMSE","MAE"]).iloc[0]
                baseline_test=r["metrics"].loc[r["metrics"]["Modelo"].eq(baseline["Modelo"])].iloc[0]
                comparisons.append({
                    "Modalidad":title,"Referencia":baseline["Modelo"],
                    "MAE LSTM":float(score.iloc[0]["MAE"]),
                    "MAE referencia":float(baseline_test["MAE"]),
                    "RMSE LSTM":float(score.iloc[0]["RMSE"]),
                    "RMSE referencia":float(baseline_test["RMSE"]),
                    "LSTM gana en RMSE":bool(score.iloc[0]["RMSE"] < baseline_test["RMSE"]),
                })
                selected=r["selected"]
                mix=r["mixture"]
                proposal=r["metrics"].loc[r["metrics"]["Modelo"].eq("Propuesta (selección VALIDACIÓN)")].iloc[0]
                decision_rows.append({
                    "Modalidad":title,
                    "Peso LSTM":mix["weight_lstm"],
                    "Peso referencia":mix["weight_reference"],
                    "Referencia":mix["reference_model"],
                    "RMSE TEST LSTM":float(score.iloc[0]["RMSE"]),
                    "RMSE TEST propuesta":float(proposal["RMSE"]),
                    "R² TEST propuesta":float(proposal["R2"]),
                })
                chosen_configs.append({
                    "Modalidad":title,"Ventana":selected["lookback"],
                    "Neuronas":selected["units"],"Épocas":selected["epochs"],
                    "Representación":selected["representation"],
                    "Pérdida":selected.get("loss","mse"),
                    "Recencia":bool(selected.get("recent",False)),
                })
        left=results["Maquila"].rename(columns={"LSTM":"LSTM maquila","Propuesta":"Maquila"})
        right=results["Corte propio"].rename(columns={"LSTM":"LSTM corte propio","Propuesta":"Corte propio"})
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
        for stage in ("TRAIN","VALIDATION","TEST"):
            ma=stage_details["Maquila"][stage].rename(columns={"Real":"Real_maquila","LSTM":"Pred_maquila"})
            cp=stage_details["Corte propio"][stage].rename(columns={"Real":"Real_corte","LSTM":"Pred_corte"})
            aligned=ma.merge(cp,on="Periodo",how="inner",validate="one_to_one")
            if len(aligned)>=2:
                actual=aligned["Real_maquila"].to_numpy(float)+aligned["Real_corte"].to_numpy(float)
                predicted=aligned["Pred_maquila"].to_numpy(float)+aligned["Pred_corte"].to_numpy(float)
                stats=metrics(actual,predicted)
                stage_rows.append({
                    "Etapa":stage,"Modalidad":"Total (períodos compartidos)",
                    "Períodos evaluados":len(aligned),
                    "R²":stats["R2"],"MAE":stats["MAE"],"RMSE":stats["RMSE"],
                    "MAPE (%)":stats["MAPE_%"],
                })
        st.session_state["metricas_etapas"]=pd.DataFrame(stage_rows)
        st.session_state["resultado_pronostico"]=both
        st.session_state["metricas_finales"]=pd.DataFrame(score_rows)
        st.session_state["comparacion_referencias"]=pd.DataFrame(comparisons)
        st.session_state["configuraciones_elegidas"]=pd.DataFrame(chosen_configs)
        st.session_state["decisiones_pronostico"]=pd.DataFrame(decision_rows)
    except Exception as exc:
        st.error(f"No se pudo generar el pronóstico: {exc}")

out=st.session_state.get("resultado_pronostico")
if out is not None:
    st.subheader("Demanda propuesta para planificación")
    st.caption("La propuesta puede ser una LSTM, una referencia sencilla o su combinación. "
               "Los pesos se eligen exclusivamente en VALIDACIÓN; no se eligen por TEST.")
    view=out.copy()
    for col in ("Maquila","Corte propio","Total"):
        view[col]=view[col].round().astype(int)
    st.dataframe(view,hide_index=True,use_container_width=True)
    st.line_chart(out.set_index("Periodo")[["Maquila","Corte propio"]])
    st.caption("Los totales se calculan con decimales y se redondean al presentar. "
               "Las columnas LSTM conservan la predicción neuronal sin mezcla.")
    st.download_button("Descargar pronósticos (CSV)",
                       out.to_csv(index=False,sep=";").encode("utf-8-sig"),
                       file_name="PRONOSTICO_LSTM.csv",mime="text/csv")

    st.divider()
    st.subheader("Métricas por etapa — LSTM pura")
    full_scores=st.session_state.get("metricas_etapas")
    if full_scores is not None:
        for stage,label in (
            ("TRAIN","Entrenamiento (TRAIN) — ajuste dentro de muestra"),
            ("VALIDATION","Validación (VALIDATION) — selección del modelo"),
            ("TEST","Prueba (TEST) — evaluación histórica reservada"),
        ):
            st.markdown(f"**{label}**")
            part=full_scores.loc[full_scores["Etapa"].eq(stage)].drop(columns="Etapa")
            st.dataframe(part.round(3),hide_index=True,use_container_width=True)
        st.caption("TRAIN es ajuste dentro de muestra y tiende a ser optimista. VALIDATION "
                   "se utilizó para seleccionar la arquitectura. TEST no se usó para escoger "
                   "parámetros, pero fue consultado durante iteraciones anteriores. "
                   "El total se calcula únicamente sobre semanas compartidas; "
                   "las métricas corresponden siempre a LSTM pura.")
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

    st.subheader("Comparación frente a referencia sencilla")
    benchmark=st.session_state.get("comparacion_referencias")
    if benchmark is not None:
        st.dataframe(benchmark.round(3),hide_index=True,use_container_width=True)
        st.caption("La referencia se selecciona por VALIDACIÓN, nunca por TEST. "
                   "En TEST comprobamos si la LSTM la supera en RMSE.")
    st.subheader("Método seleccionado en validación")
    decision=st.session_state.get("decisiones_pronostico")
    if decision is not None:
        st.dataframe(decision.round(3),hide_index=True,use_container_width=True)
        st.caption("Las métricas TEST de la propuesta no son las métricas de la LSTM pura. "
                   "Como este TEST ya se ha consultado varias veces, los resultados "
                   "siguen siendo exploratorios hasta una evaluación futura.")
    st.subheader("Configuración LSTM elegida")
    chosen=st.session_state.get("configuraciones_elegidas")
    if chosen is not None:
        st.dataframe(chosen,hide_index=True,use_container_width=True)
