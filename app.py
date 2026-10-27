"""Aplicación final de pronóstico de demanda y producción para tesis de jeans.

Inicia: streamlit run app.py
Requiere: Python 3.11, requirements.txt, hoja DEMANDA.
La procedencia se conserva en los datos de entrada y en la ficha técnica.
"""
from __future__ import annotations

from dataclasses import asdict
import hmac
import os
import sys
import importlib.util
from datetime import date, datetime, timedelta
from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from motor_lstm import (APP_TIMEZONE, TARGETS, Settings, extract_series, final_test, combined_test_metrics,
                        fit_one, full_refit_forecast, make_forecast_table, recursive_test,
                        prepare_base, metrics)

st.set_page_config(page_title="LSTM | Planificación de jeans",page_icon="📈",layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1440px}
h1,h2,h3{font-family:Arial,sans-serif}
[data-testid="stMetric"]{border:1px solid #e6e6e0;border-radius:6px;padding:9px 13px;background:#fff}
[data-testid="stSidebar"]{background:#f8f8f5}
hr{border-color:#e8e8e6}
</style>
""",unsafe_allow_html=True)

required_password=os.environ.get("TESIS_ACCESS_PASSWORD", "").strip()
if required_password:
    access=st.text_input("Clave de acceso al modelo",type="password")
    if not hmac.compare_digest(str(access),required_password):
        st.info("Introduce la clave facilitada por el autor para acceder a la aplicación.")
        st.stop()

st.title("Pronóstico de demanda y producción de jeans")
st.caption("Redes LSTM independientes · Evaluación temporal · Horizonte de 12 semanas · Modalidades maquila y corte propio")

with st.sidebar:
    st.header("Configuración")
    st.caption("Parámetros fijados antes de examinar el conjunto de prueba")
    window=st.number_input("Ventana histórica (semanas)",min_value=4,max_value=24,value=12,step=1)
    epochs=st.number_input("Épocas máximas",min_value=20,max_value=250,value=100,step=10)
    units=st.select_slider("Unidades LSTM",options=[8,16,24,32,48,64],value=24)
    train_pct=st.slider("Entrenamiento (%)",min_value=60,max_value=80,value=70,step=5)
    val_pct=st.slider("Validación (%)",min_value=10,max_value=20,value=15,step=5)
    seed=st.number_input("Semilla aleatoria",min_value=0,max_value=999999,value=42,step=1)
    default_date=date(2026,11,2)
    start_plan=st.date_input("Inicio del horizonte futuro (lunes)",value=default_date)
    horizon=st.number_input("Semanas a pronosticar",min_value=1,max_value=26,value=12,step=1)
    st.caption("La prueba se reserva hasta finalizar la validación. No se utiliza para ajustar hiperparámetros.")

config=Settings(lookback=int(window),epochs=int(epochs),units=int(units),
                train_ratio=float(train_pct)/100,val_ratio=float(val_pct)/100,seed=int(seed))

upload=st.file_uploader("1. Cargar base de datos Excel",type=["xlsx","xlsm"],help="Utiliza la hoja DEMANDA y el encabezado de la primera fila.")
if upload is None:
    st.info("Adjunta 04_BASE_DE_DATOS.xlsx para iniciar. El archivo se procesa durante esta sesión y no es necesario modificar sus datos originales.")
    st.stop()

content=upload.getvalue()
try:
    book=pd.ExcelFile(BytesIO(content))
    sheet=st.selectbox("Hoja del archivo",book.sheet_names,index=(book.sheet_names.index("DEMANDA") if "DEMANDA" in book.sheet_names else 0))
    raw=pd.read_excel(BytesIO(content),sheet_name=sheet,header=0)
    data,info=prepare_base(raw)
except Exception as exc:
    st.error(f"No fue posible leer la base: {exc}")
    st.stop()

signature=sha256(content+sheet.encode()+repr(asdict(config)).encode()).hexdigest()
forecast_signature=f"{start_plan}-{horizon}"
if st.session_state.get("forecast_signature") != forecast_signature:
    st.session_state["forecast_signature"]=forecast_signature
    st.session_state.pop("pronosticos",None)
if st.session_state.get("signature")!=signature:
    st.session_state["signature"]=signature
    for key in ("ajustes","pruebas","pronosticos"):
        st.session_state.pop(key,None)

c1,c2,c3,c4=st.columns(4)
c1.metric("Semanas cerradas",info["semanas_completas"])
c2.metric("Última semana cerrada",info["fecha_ultima_semana"].strftime("%d/%m/%Y"))
c3.metric("Semanas en curso o futuras",info["semanas_no_cerradas"])
c4.metric("Series a entrenar",len(TARGETS))
if info["semanas_no_cerradas"]:
    st.caption("Las semanas cuyo viernes aún no ha transcurrido no se utilizan para entrenar ni evaluar.")

with st.expander("Ver datos históricos y ficha técnica",expanded=False):
    shown=data[["fecha_inicio","demanda_mq","demanda_cp","produccion_cp"]].rename(columns={
        "fecha_inicio":"Inicio", "demanda_mq":"Demanda maquila", "demanda_cp":"Demanda corte propio",
        "produccion_cp":"Producción corte propio"})
    st.dataframe(shown,hide_index=True,use_container_width=True,height=300)
    metadata={
        "Archivo cargado":upload.name,"Hoja":sheet,
        "Período disponible":f"{info['fecha_primera_semana']} a {info['fecha_ultima_semana']}",
        "SHA-256":sha256(content).hexdigest(),
        "Fuente declarada en la base": ", ".join(info["fuentes_declaradas"]) if info["fuentes_declaradas"] else "No especificada",
    }
    st.table(pd.DataFrame(list(metadata.items()),columns=["Dato","Valor"]))
    st.caption("Esta ficha reproduce la información declarada en el archivo; no certifica documentos externos. Los valores faltantes anteriores al inicio de corte propio no se convierten en demanda cero.")
    if info["diferencias_en_total_archivo"]:
        st.warning(f"El total de producción del archivo difiere de maquila + corte propio en {info['diferencias_en_total_archivo']} filas. Los pronósticos de totales se calcularán por suma explícita.")

series={}
try:
    for name,target in TARGETS.items():
        series[name]=extract_series(data,target)
except Exception as exc:
    st.error(f"No fue posible preparar las tres series: {exc}")
    st.stop()

with st.expander("Gráfica del historial",expanded=False):
    selected=st.selectbox("Variable histórica",list(TARGETS.keys()),key="history_choice")
    t=series[selected]
    plot=go.Figure()
    plot.add_trace(go.Scatter(x=t.fecha_inicio,y=t.valor,mode="lines+markers",name=selected,line_color="#916900",marker_size=3))
    plot.update_layout(height=310,margin=dict(l=10,r=10,t=15,b=10),xaxis_title="Semana",yaxis_title="Pantalones",template="plotly_white")
    st.plotly_chart(plot,use_container_width=True)

train_tab,valid_tab,test_tab,pred_tab=st.tabs(["2. Entrenamiento","3. Validación","4. Prueba","5. Pronóstico final"])

with train_tab:
    st.markdown("#### Entrenar las tres series")
    st.write("Se entrenan **tres redes LSTM**, con conjuntos cronológicos de entrenamiento, validación y prueba independientes para cada serie. La normalización se ajusta solo con entrenamiento.")
    if st.button("Entrenar modelos LSTM",type="primary",use_container_width=True):
        st.session_state.pop("ajustes",None)
        st.session_state.pop("pruebas",None)
        st.session_state.pop("pronosticos",None)
        fitted={}
        bar=st.progress(0)
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            for n,(name,frame) in enumerate(series.items(),start=1):
                with st.spinner(f"Entrenando {name}…"):
                    fitted[name]=fit_one(name,frame,config)
                bar.progress(n/len(series))
            st.session_state["ajustes"]=fitted
            st.success("Entrenamiento y validación finalizados. Continúa en la pestaña Validación.")
        except ImportError as exc:
            st.error("No se pudo importar TensorFlow o una de sus dependencias.")
            st.code(f"Python: {sys.version.split()[0]} | TensorFlow disponible: {importlib.util.find_spec('tensorflow') is not None}\n{type(exc).__name__}: {exc}")
        except Exception as exc:
            st.error(f"No se completó el entrenamiento: {type(exc).__name__}: {exc}")

fits=st.session_state.get("ajustes")
with valid_tab:
    if not fits:
        st.info("Primero ejecuta el entrenamiento en la pestaña anterior.")
    else:
        st.markdown("#### Resultados de validación")
        summary=[]
        for name,fit in fits.items():
            for stage,values in [("Entrenamiento",fit.train_metrics),("Validación",fit.validation_metrics),("Referencia validación",fit.baseline_validation_metrics)]:
                summary.append({"Variable":name,"Etapa":stage,"Semanas":len(fit.training) if stage=="Entrenamiento" else len(fit.validation),**values})
        report=pd.DataFrame(summary)
        st.dataframe(report,hide_index=True,use_container_width=True,column_config={k:st.column_config.NumberColumn(format="%.3f") for k in ["MAE","RMSE","R²","Sesgo","WAPE (%)","MASE"]})
        choice=st.selectbox("Variable para inspección",list(fits.keys()),key="valid_choice")
        res=fits[choice]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=res.validation.fecha_inicio,y=res.validation.real,name="Real",mode="lines+markers",line_color="#1B365D"))
        fig.add_trace(go.Scatter(x=res.validation.fecha_inicio,y=res.validation.LSTM,name="LSTM",mode="lines+markers",line_color="#C39B00"))
        fig.add_trace(go.Scatter(x=res.validation.fecha_inicio,y=res.validation.Referencia_ultimo_valor,name="Referencia: última semana",line=dict(color="#999999",dash="dash")))
        fig.update_layout(title=f"Validación: {choice}",template="plotly_white",height=350,margin=dict(l=10,r=10,t=50,b=10),yaxis_title="Pantalones")
        st.plotly_chart(fig,use_container_width=True)
        m=res.validation_metrics["MAE"]
        ref=res.baseline_validation_metrics["MAE"]
        st.caption(f"MAE validación LSTM: {m:.2f} · MAE persistencia: {ref:.2f} · Mejor época de validación: {res.best_epoch}")
        if m >= ref:
            st.info("En esta variable, la LSTM no supera en MAE al pronóstico de persistencia durante validación. Es un resultado importante que debe reportarse.")
        with st.expander("Curva de aprendizaje"):
            losses=res.loss.rename(columns={"epoch":"Época","train_loss":"Entrenamiento","val_loss":"Validación"}).set_index("Época")
            st.line_chart(losses)

with test_tab:
    if not fits:
        st.info("Entrena los modelos antes de evaluar la prueba final.")
    else:
        st.write("La prueba se evalúa con datos reservados, **sin reentrenar ni modificar los hiperparámetros según sus resultados**.")
        if not st.session_state.get("pruebas"):
            if st.button("Evaluar PRUEBA final",type="primary",use_container_width=True):
                results={}
                try:
                    for name,fit in fits.items():
                        with st.spinner(f"Evaluando {name}…"):
                            one=final_test(fit)
                            twelve=recursive_test(fit,12)
                            results[name]=(*one,*twelve)
                    st.session_state["pruebas"]=results
                    st.success("Prueba final evaluada. Los resultados quedan fijados para esta carga y configuración.")
                except Exception as exc:
                    st.error(f"Error en prueba: {exc}")
        tests=st.session_state.get("pruebas")
        if tests:
            test_rows=[]
            for name,(table,m,b) in tests.items():
                test_rows.append({"Variable":name,"Modelo":"LSTM","Semanas":len(table),**m})
                test_rows.append({"Variable":name,"Modelo":"Persistencia","Semanas":len(table),**b})
            totals=combined_test_metrics(tests)
            for total_label,(table,m,b) in totals.items():
                test_rows.append({"Variable":total_label,"Modelo":"Suma LSTM","Semanas":len(table),**m})
                test_rows.append({"Variable":total_label,"Modelo":"Suma persistencia","Semanas":len(table),**b})
            for name,(full,m,b,backtest,rm,rb) in tests.items():
                test_rows.append({"Variable":name,"Modelo":"LSTM recursivo 12 semanas","Semanas":len(backtest),**rm})
                test_rows.append({"Variable":name,"Modelo":"Persistencia fija 12 semanas","Semanas":len(backtest),**rb})
            st.dataframe(pd.DataFrame(test_rows),hide_index=True,use_container_width=True)
            st.caption("PRUEBA un paso utiliza el histórico anterior a cada semana. PRUEBA recursiva proyecta 12 semanas sin consultar valores reales intermedios; es la comparación más semejante al pronóstico futuro.")
            chosen=st.selectbox("Variable en PRUEBA",list(tests.keys()),key="test_choice")
            frame=tests[chosen][0]
            p=go.Figure()
            p.add_trace(go.Scatter(x=frame.fecha_inicio,y=frame.real,name="Real",line_color="#1B365D"))
            p.add_trace(go.Scatter(x=frame.fecha_inicio,y=frame.LSTM,name="LSTM",line_color="#C39B00"))
            p.add_trace(go.Scatter(x=frame.fecha_inicio,y=frame.Referencia_ultimo_valor,name="Persistencia",line=dict(color="#999999",dash="dash")))
            p.update_layout(height=350,template="plotly_white",margin=dict(l=10,r=10,t=20,b=10),yaxis_title="Pantalones")
            st.plotly_chart(p,use_container_width=True)
            with st.expander("Prueba recursiva de 12 semanas (sin datos reales intermedios)"):
                rolling=tests[chosen][3]
                gp=go.Figure()
                gp.add_trace(go.Scatter(x=rolling.fecha_inicio,y=rolling.real,name="Real",line_color="#1B365D"))
                gp.add_trace(go.Scatter(x=rolling.fecha_inicio,y=rolling.LSTM,name="LSTM recursiva",line_color="#C39B00"))
                gp.update_layout(height=310,template="plotly_white",margin=dict(l=10,r=10,t=20,b=10),yaxis_title="Pantalones")
                st.plotly_chart(gp,use_container_width=True)
                st.caption(f"MAE LSTM recursivo: {tests[chosen][4]['MAE']:.2f} · MAE referencia: {tests[chosen][5]['MAE']:.2f}")
            st.caption("Las métricas de prueba describen únicamente la capacidad predictiva sobre los datos reservados; no garantizan resultados futuros.")

with pred_tab:
    fits=st.session_state.get("ajustes")
    tests=st.session_state.get("pruebas")
    if not fits or not tests:
        st.info("Para generar el pronóstico final, primero entrena los modelos y evalúa PRUEBA.")
    else:
        st.markdown("#### Pronóstico para la planificación de producción")
        st.write("El modelo final se reajusta con el histórico cerrado, manteniendo la arquitectura fija y las épocas seleccionadas con validación. No vuelve a seleccionar hiperparámetros en prueba.")
        if st.button("Generar pronóstico futuro",type="primary",use_container_width=True):
            try:
                fcs={}
                for name,fit in fits.items():
                    with st.spinner(f"Generando futuro: {name}…"):
                        fcs[name]=full_refit_forecast(fit,start_plan,int(horizon))
                st.session_state["pronosticos"]=make_forecast_table(fcs)
                st.success("Pronósticos generados. Ya puedes descargarlos para tu planificación agregada y PMP.")
            except Exception as exc:
                st.error(f"No fue posible generar el pronóstico: {exc}")
        forecast=st.session_state.get("pronosticos")
        if forecast is not None:
            display=forecast.rename(columns={
                "fecha_inicio":"Semana desde","fecha_fin":"Semana hasta",
                "demanda_mq":"Demanda MQ","demanda_cp":"Demanda CP","demanda_total":"Demanda total",
                "produccion_cp":"Producción CP","produccion_total_referencia":"Producción total de referencia",
            }).copy()
            for key in list(display.columns)[2:]:
                display[key]=np.rint(display[key].to_numpy(float)).astype(int)
            st.dataframe(display,hide_index=True,use_container_width=True)
            plot=go.Figure()
            for c,color in [("demanda_mq","#1B365D"),("demanda_cp","#C39B00"),("produccion_cp","#26816F")]:
                plot.add_trace(go.Scatter(x=forecast.fecha_inicio,y=forecast[c],name=c.replace("_"," ").capitalize(),mode="lines+markers",line=dict(color=color)))
            plot.update_layout(height=350,template="plotly_white",margin=dict(l=10,r=10,t=20,b=10),yaxis_title="Unidades semanales")
            st.plotly_chart(plot,use_container_width=True)
            st.caption("Producción total de referencia = demanda de maquila (supuesto de producción bajo pedido) + producción pronosticada de corte propio. No es una medición independiente de producción de maquila.")

            # Reporte reproducible con métricas y series temporales; el archivo de origen se conserva sin alterar.
            try:
                buffer=BytesIO()
                rows=[]
                valid_frames=[]
                test_frames=[]
                for name,fit in fits.items():
                    for stage,m in [("Entrenamiento",fit.train_metrics),("Validación",fit.validation_metrics),
                                    ("Prueba",tests[name][1]),("Referencia validación",fit.baseline_validation_metrics),
                                    ("Referencia prueba",tests[name][2])]:
                        rows.append({"Serie":name,"Etapa":stage,**m,"Mejor época":fit.best_epoch,
                                     "Semanas entrenamiento":len(fit.training),"Semanas validación":len(fit.validation),
                                     "Semanas prueba":len(fit.idx_test)})
                    for stage,values,periods in [("Prueba recursiva 12 sem",tests[name][4],len(tests[name][3])),
                                                   ("Referencia recursiva 12 sem",tests[name][5],len(tests[name][3]))]:
                        rows.append({"Serie":name,"Etapa":stage,**values,"Mejor época":fit.best_epoch,
                                     "Semanas entrenamiento":len(fit.training),"Semanas validación":len(fit.validation),
                                     "Semanas prueba":periods})
                    v=fit.validation.copy();v.insert(0,"Serie",name);valid_frames.append(v)
                    t=tests[name][0].copy();t.insert(0,"Serie",name);t.insert(1,"Tipo","Un paso");test_frames.append(t)
                    br=tests[name][3].copy();br.insert(0,"Serie",name);br.insert(1,"Tipo","Recursivo 12 semanas");test_frames.append(br)
                for total_label,(table,m,b) in combined_test_metrics(tests).items():
                    for stage,values in [("Prueba total",m),("Referencia prueba total",b)]:
                        rows.append({"Serie":total_label,"Etapa":stage,**values,"Mejor época":None,
                                     "Semanas entrenamiento":None,"Semanas validación":None,
                                     "Semanas prueba":len(table)})
                meta=pd.DataFrame([
                    ("Archivo",upload.name),("SHA256",sha256(content).hexdigest()),
                    ("Fecha última semana cerrada",str(info["fecha_ultima_semana"])),
                    ("Inicio pronóstico",str(start_plan)),("Horizonte semanal",int(horizon)),
                    ("Versión metodológica","LSTM-univariable-tres-series-v1"),
                    ("Separación",f"{train_pct}% TRAIN, {val_pct}% VALIDACIÓN, {100-train_pct-val_pct}% TEST"),
                    ("Ventana",config.lookback),("Unidades LSTM",config.units),
                    ("Épocas máximas",config.epochs),("Semilla",config.seed),
                    ("Total de producción","MQ demanda + CP producción pronosticada (supuesto)"),
                    ("Fuente declarada",", ".join(info["fuentes_declaradas"]) or "No especificada"),
                ],columns=["Parámetro","Valor"])
                with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
                    display.to_excel(writer,sheet_name="PRONOSTICO_12S",index=False)
                    pd.DataFrame(rows).to_excel(writer,sheet_name="METRICAS",index=False)
                    pd.concat(valid_frames,ignore_index=True).to_excel(writer,sheet_name="VALIDACION",index=False)
                    pd.concat(test_frames,ignore_index=True).to_excel(writer,sheet_name="PRUEBA",index=False)
                    meta.to_excel(writer,sheet_name="FICHA_TECNICA",index=False)
                    from openpyxl.styles import PatternFill,Font,Alignment
                    for ws in writer.book.worksheets:
                        ws.freeze_panes="A2"
                        ws.auto_filter.ref=ws.dimensions
                        for cell in ws[1]:
                            cell.fill=PatternFill("solid",fgColor="FFF200")
                            cell.font=Font(bold=True,color="1D1D1D")
                            cell.alignment=Alignment(wrap_text=True,vertical="center")
                        ws.row_dimensions[1].height=32
                        for cells in ws.columns:
                            letter=cells[0].column_letter
                            limit=min(31,max(13,max(len(str(v.value or "")) for v in cells[:min(35,len(cells))])+2))
                            ws.column_dimensions[letter].width=limit
                st.download_button("⬇ Descargar pronóstico, métricas y validación (Excel)",data=buffer.getvalue(),
                                   file_name="REPORTE_LSTM_DEMANDA_PRODUCCION.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",type="primary",use_container_width=True)
            except Exception as exc:
                st.error(f"No se pudo preparar el informe Excel: {exc}")
            st.download_button("⬇ Descargar pronóstico (CSV)",data=display.to_csv(index=False,sep=";").encode("utf-8-sig"),
                               file_name="PRONOSTICO_12_SEMANAS.csv",mime="text/csv",use_container_width=True)

st.divider()
st.caption("Investigación de Ingeniería Industrial · Datos y pronósticos por semana · Valores esperados; comprobar factibilidad con las capacidades de estaciones y operarios antes de programar pedidos.")