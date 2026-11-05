"""Aplicación de tesis: pronóstico semanal de demanda para planificación productiva.
La producción propia es análisis opcional. Datos originales inalterados.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import date, datetime, timedelta
from hashlib import sha256
from io import BytesIO
import hmac
import os
import sys
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from motor_lstm import (APP_TIMEZONE,Settings,prepare_base,audit_inventory,extract_series,fit_one,
                         final_test,recursive_test,make_forecast_table)
from pronostico_horizonte import (select_demand_by_validation,evaluate_demand_test,
                                  forecast_lstm_verified,forecast_demand_selected)
from planificacion_reposicion import calcular_plan_reposicion
from capacidad_recursos import read_capacity, loads as resource_loads
from lstm_validacion_horizonte import select_frozen, future_search
from lstm_directo import select_direct
from produccion_intermitente import (select_by_validation,evaluate_test as test_intermitente,
                                     forecast_future as forecast_intermitente)
from protocolo_final import (confirm_pre_sales, select as select_protocol,
                             evaluate as evaluate_protocol, future as future_protocol, demand_from_fit)

st.set_page_config(page_title="Planificación de jeans | Pronóstico LSTM",page_icon="📈",layout="wide")
st.title("Pronóstico semanal de demanda y producción de jeans")
st.caption("Demanda MQ y CP · Producción CP estimada · Evaluación temporal · Planificación agregada y PMP")
st.caption("Versión v7 · Evaluación fija y rolling de 12 semanas · Comparación LSTM y referencias · Capacidad compartida por recurso.")
password=os.environ.get("TESIS_ACCESS_PASSWORD","").strip()
if password and not hmac.compare_digest(st.text_input("Clave de acceso",type="password"),password):
    st.info("Introduce la clave de acceso."); st.stop()

with st.sidebar:
    st.header("Configuración")
    weeks=st.number_input("Semanas del pronóstico",min_value=1,max_value=26,value=12,step=1)
    today_local=datetime.now(APP_TIMEZONE).date()
    days_to_monday=(7-today_local.weekday())%7
    default_start=today_local+timedelta(days=days_to_monday or 7)
    start=st.date_input("Inicio del plan (lunes)",value=default_start,
                        min_value=default_start,help="Se propone automáticamente el próximo lunes para evitar fechas de planificación vencidas.")
    st.caption("Ajustes de entrenamiento")
    look=st.number_input("Ventana histórica",min_value=4,max_value=24,value=12)
    ep=st.number_input("Épocas máximas",min_value=20,max_value=250,value=100,step=10)
    units=st.select_slider("Unidades LSTM",options=[8,16,24,32,48,64],value=16)
    seed=st.number_input("Semilla",min_value=0,max_value=999999,value=42)
    optional=st.checkbox("Incluir pronóstico estadístico de producción CP (complementario)",value=False,
                         help="No es necesario para pronosticar las dos demandas ni para calcular la reposición. Actívalo solo si deseas analizar la producción histórica como serie independiente.")
    with st.expander("Detalles de evaluación",expanded=False):
        train=st.slider("Entrenamiento (%)",60,80,70,5)
        val=st.slider("Validación (%)",10,20,15,5)
    st.caption("No se usan las semanas de prueba para seleccionar métodos.")

settings=Settings(lookback=int(look),epochs=int(ep),units=int(units),seed=int(seed),
                  train_ratio=train/100,val_ratio=val/100)
file=st.file_uploader("Cargar 04_BASE_DE_DATOS.xlsx",type=["xlsx","xlsm"])
if file is None:
    st.info("Carga el archivo con la hoja DEMANDA para iniciar."); st.stop()
payload=file.getvalue()
source_digest=sha256(payload).hexdigest()
config_path=Path(__file__).with_name('configuracion_lstm_validada.json')
validated_config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
try:
    xl=pd.ExcelFile(BytesIO(payload))
    if "DEMANDA" not in xl.sheet_names:
        raise ValueError("No existe la hoja DEMANDA en el archivo.")
    base,info=prepare_base(pd.read_excel(BytesIO(payload),sheet_name="DEMANDA"))

    # Confirmación previa del usuario, registrada el 03-11-2026.
    # Solo se preselecciona para el hash auditado y el tramo exacto de cinco semanas.
    first_prod=np.flatnonzero(base["produccion_cp"].fillna(0).to_numpy(float)>0)
    first_demand=np.flatnonzero(base["demanda_cp"].notna().to_numpy())
    auto_confirm_zeros=False
    if (source_digest==validated_config.get("source_sha256")
            and len(first_prod)>0 and len(first_demand)>0):
        a,b=int(first_prod[0]),int(first_demand[0])
        gap=base["demanda_cp"].iloc[a:b]
        auto_confirm_zeros=(b-a==5 and gap.isna().all()
                            and base["demanda_cp"].iloc[:a].isna().all()
                            and base["demanda_cp"].iloc[b:].notna().all())
    confirm_zeros=st.checkbox(
        "Confirmación previa: no hubo ventas CP en las 5 semanas entre la primera producción y el primer registro",
        value=auto_confirm_zeros,
        help="Se preselecciona únicamente para el archivo exacto ya auditado y confirmado. Completa solo esas cinco semanas; no convierte en ceros las semanas anteriores al inicio de corte propio."
    )
    base,zero_trace=confirm_pre_sales(base,confirm_zeros)
    inventory_audit=audit_inventory(base)
    names={"Demanda de maquila":"demanda_mq","Demanda de corte propio":"demanda_cp"}
    if optional: names["Producción de corte propio"]="produccion_cp"
    series={name:extract_series(base,col) for name,col in names.items()}
except Exception as e:
    st.error("No fue posible validar la base: "+str(e)); st.stop()

profile_available=(validated_config.get('source_sha256')==source_digest
                   and confirm_zeros and settings==Settings())
use_validated=st.checkbox('Usar configuración LSTM elegida por validación de 12 semanas',
    value=profile_available,disabled=not profile_available,
    help='Disponible para la base auditada, con ceros iniciales confirmados y ajustes manuales por defecto. No garantiza alta precisión en las tres series.')
if use_validated:
    st.caption('Configuración de referencia: 16 unidades; épocas previamente elegidas por serie. La app vuelve a comparar LSTM recursiva y directa usando únicamente VALIDACIÓN.')
    if validated_config.get("test_previously_observed", False):
        st.warning("Nota metodológica: el bloque TEST de esta base ya fue consultado en iteraciones anteriores. Se mantiene fuera de la selección actual, pero debe interpretarse como un holdout reutilizado, no como una evaluación completamente independiente o ciega.")
signature=sha256(b'lstm-horizonte-v5-direct'+payload+repr(asdict(settings)).encode()+str((optional,confirm_zeros,use_validated)).encode()).hexdigest()
if st.session_state.get("signature")!=signature:
    st.session_state["signature"]=signature
    for k in ("fits","tests","protocol_test_results","forecast","cp_choice","demand_choice","protocols","plan_preliminar","plan_signature"):
        st.session_state.pop(k,None)
forecast_key=(str(start),int(weeks))
if st.session_state.get("forecast_key")!=forecast_key:
    st.session_state["forecast_key"]=forecast_key
    st.session_state.pop("forecast",None)

c1,c2,c3=st.columns(3)
c1.metric("Semanas completas",info["semanas_completas"])
c2.metric("Último viernes registrado",str(info["fecha_ultima_semana"]))
c3.metric("Series analizadas",len(series))
if info["semanas_no_cerradas"]:
    st.caption("Se excluyen las semanas todavía no cerradas.")
with st.expander("Datos y trazabilidad"):
    st.dataframe(base[["fecha_inicio","demanda_mq","demanda_cp","produccion_cp"]],hide_index=True,use_container_width=True,height=230)
    st.caption("Archivo: "+file.name+" | SHA-256: "+sha256(payload).hexdigest())
    st.caption("Origen declarado: "+(", ".join(info["fuentes_declaradas"]) if info["fuentes_declaradas"] else "No declarado. Verificar documentación de origen."))
    st.caption("Los registros se leen sin modificar. Los ceros reales de producción permanecen.")
    if not zero_trace.empty:
        st.dataframe(zero_trace,hide_index=True)
    if inventory_audit.get("available"):
        st.caption(f"Auditoría de inventario CP: {inventory_audit['weeks_checked']}/{inventory_audit['weeks_expected']} semanas verificadas; "
                   f"errores de balance: {inventory_audit['balance_errors']}; errores de continuidad: {inventory_audit['carry_errors']}.")
        if inventory_audit['weeks_missing'] or inventory_audit['balance_errors'] or inventory_audit['carry_errors']:
            st.warning("La auditoría detectó faltantes o diferencias en el kardex CP. Revisa los registros antes de planificar.")
        else:
            st.success("Kardex CP verificado: inventario final = inicial + producción − demanda, con continuidad semanal.")
    if info["diferencias_en_total_archivo"]:
        st.warning("El campo de producción total del archivo difiere de la suma MQ+CP en "+str(info["diferencias_en_total_archivo"])+" semanas.")

t1,t2,t3,t4=st.tabs(["1. Entrenamiento","2. Validación","3. Prueba","4. Pronóstico y Excel"])
with t1:
    st.subheader("Entrenar LSTM de demanda y producción")
    st.write("Se entrenan por separado las dos demandas. La búsqueda compara ventanas LSTM locales y largas (8, 12, 16, 26 y 52 semanas) y referencias estacionales de 48 y 52 semanas. La producción CP es complementaria y puede seleccionar LSTM Hurdle o referencias estadísticas usando solo VALIDACIÓN.")
    if st.button("Entrenar modelos",type="primary",use_container_width=True):
        for k in ("fits","tests","protocol_test_results","forecast","cp_choice","demand_choice","protocols","plan_preliminar","plan_signature"):
            st.session_state.pop(k,None)
        fitted={}
        progress=st.progress(0)
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            for i,(name,frame) in enumerate(series.items()):
                with st.spinner("Entrenando "+name):
                    fitted[name]=fit_one(name,frame,settings)
                progress.progress((i+1)/len(series))
            st.session_state["fits"]=fitted
            st.session_state["demand_choice"]=select_demand_by_validation(fitted["Demanda de corte propio"],min(int(weeks),12))
            if optional:
                st.session_state["cp_choice"]=select_by_validation(fitted["Producción de corte propio"])
            with st.spinner("Comparando candidatos a 12 semanas con validación temporal separada"):
                st.session_state["protocols"]={name:(demand_from_fit(fitted[name]) if name.startswith("Demanda")
                                               else select_protocol(frame,True,settings))
                                               for name,frame in series.items()}
                if use_validated:
                    st.session_state['protocols']={name:select_frozen(series[name],validated_config['models'][name],p,calibrate_bias=(name=='Demanda de corte propio'),allow_statistical=(name=='Producción de corte propio'))
                                                  for name,p in st.session_state['protocols'].items()}
                # Experimento predefinido: comparar LSTM recursiva y directa
                # con los mismos orígenes temporales de VALIDACIÓN.
                with st.spinner("Comparando LSTM recursiva y directa a 12 semanas; puede tardar unos minutos..."):
                    for name in ("Demanda de maquila", "Demanda de corte propio"):
                        st.session_state['protocols'][name]=select_direct(
                            series[name], settings,
                            control=st.session_state['protocols'][name],
                            calibrate_bias=(name=="Demanda de corte propio"))
            st.success("Entrenamiento y comparación temporal completados.")
        except ImportError as e:
            st.session_state.pop("fits",None)
            st.session_state.pop("protocols",None)
            st.error("No se pudo importar TensorFlow: "+str(e))
            st.caption("Python: "+sys.version.split()[0])
        except Exception as e:
            st.session_state.pop("fits",None)
            st.session_state.pop("protocols",None)
            st.error("No se completó el entrenamiento: "+type(e).__name__+": "+str(e))

fits=st.session_state.get("fits")
protocols=st.session_state.get("protocols")
with t2:
    if not fits:
        st.info("Primero entrena en la pestaña 1.")
    else:
        if protocols:
            st.subheader("Validación de la LSTM seleccionada")
            st.caption("Las demandas se seleccionan entre LSTM recursiva y directa mediante VALIDACIÓN. Se incluyen referencias estacionales de 48 y 52 semanas. La producción CP es complementaria y puede elegir entre LSTM Hurdle y métodos estadísticos, siempre por VALIDACIÓN.")
            for name,protocol in protocols.items():
                st.write(name+": "+protocol['selected'])
                if name=="Demanda de corte propio" and protocol.get('bias_correction',0.0)!=0:
                    st.caption("Calibración aditiva CP estimada exclusivamente con residuos de VALIDACIÓN: "+
                               format(protocol['bias_correction'],"+.2f")+" pantalones/semana. La tabla de validación conserva métricas sin calibrar; PRUEBA evalúa la corrección sin reajustarla.")
                if protocol.get('best_comparison')!=protocol['selected']:
                    st.info("La comparación de validación favorece a "+protocol['best_comparison']+". Se mantiene LSTM por el alcance de la tesis; no se afirma superioridad.")
                selected_validation=protocol['validation'].loc[
                    protocol['validation']['Modelo']==protocol['selected']]
                st.dataframe(selected_validation.round(3),hide_index=True,use_container_width=True)
            st.caption("Las dos demandas mantienen una LSTM como modelo principal por el alcance de la tesis. Las referencias estacionales permiten evaluar si la red añade valor; no se afirma superioridad cuando no la hay. Producción CP se selecciona por VALIDACIÓN como análisis complementario.")
        with st.expander("Anexo técnico: comparaciones y diagnóstico del motor anterior",expanded=False):
            if protocols:
                st.subheader("Comparación de modelos a 12 semanas — VALIDACIÓN")
                st.caption("Todos los candidatos comparten orígenes y horizonte. TEST no participa en la selección.")
                for protocol_name, protocol in protocols.items():
                    st.markdown("**"+protocol_name+"**")
                    st.dataframe(protocol['validation'].round(3),hide_index=True,use_container_width=True)
            st.subheader("Diagnóstico del motor anterior: validación a un paso")
            rows=[]
            reference_rows=[]
            for name,fit in fits.items():
                row={"Serie":name,"Modelo":"LSTM","Etapa":"Validación",**fit.validation_metrics}
                base_row={"Serie":name,"Modelo":"Persistencia","Etapa":"Validación",**fit.baseline_validation_metrics}
                if name.startswith("Demanda"):
                    rows.append(row)
                reference_rows.extend([row,base_row])
            st.dataframe(pd.DataFrame(rows).round(3),hide_index=True,use_container_width=True)
            st.caption("Estas métricas usan el valor real de la semana anterior para predecir una semana. No representan el desempeño de un pronóstico fijo de 12 semanas.")
            selected=st.session_state.get("demand_choice")
            if selected is not None:
                row=selected.comparison.iloc[0]
                st.caption("Demanda CP a "+str(selected.horizon)+" semanas: mejor método por MAE de validación = "+selected.method+
                           "; MAE = "+format(row["MAE"],".2f")+".")
            series_choice=st.selectbox("Gráfico de validación",["Demanda de maquila","Demanda de corte propio"],key="valid_series")
            f=fits[series_choice]
            fig=go.Figure()
            for label,field,color in [("Real","real","#1B365D"),("LSTM","LSTM","#C39B00"),("Persistencia","Referencia_ultimo_valor","#8A8A8A")]:
                fig.add_trace(go.Scatter(x=f.validation.fecha_inicio,y=f.validation[field],name=label,mode="lines+markers",line_color=color))
            fig.update_layout(template="plotly_white",height=330,yaxis_title="Pantalones/semana")
            st.plotly_chart(fig,use_container_width=True)
            with st.expander("Comparaciones metodológicas y curva de aprendizaje"):
                st.markdown("**Comparación con persistencia (validación)**")
                st.dataframe(pd.DataFrame(reference_rows).round(3),hide_index=True,use_container_width=True)
                if selected is not None:
                    st.markdown("**Métodos de demanda CP: pronóstico recursivo**")
                    st.dataframe(selected.comparison.round(3),hide_index=True,use_container_width=True)
                    st.caption("Las ventanas de validación se solapan y no equivalen a observaciones independientes.")
                if optional and st.session_state.get("cp_choice") is not None:
                    st.markdown("**Producción CP: métodos de producción intermitente**")
                    st.dataframe(st.session_state["cp_choice"].comparison.round(3),hide_index=True,use_container_width=True)
                loss=f.loss.rename(columns={"epoch":"Época","train_loss":"Entrenamiento","val_loss":"Validación"}).set_index("Época")
                st.line_chart(loss)

with t3:
    if not fits:
        st.info("Primero entrena los modelos.")
    else:
        st.subheader("Prueba final de la LSTM seleccionada")
        if st.session_state.get("tests") is None:
            if st.button("Evaluar PRUEBA",type="primary",use_container_width=True):
                total_steps=len(fits)+len(protocols or {})
                completed=0
                evaluation_progress=st.progress(0,text="Preparando evaluación de prueba…")
                st.info("La evaluación puede tardar varios minutos en Streamlit Cloud. No vuelvas a pulsar ni recargues mientras se calculan las métricas.")
                try:
                    result={}
                    protocol_test_results={}
                    for name,fit in fits.items():
                        label="Evaluando diagnóstico LSTM: "+name
                        evaluation_progress.progress(completed/total_steps,text=label)
                        with st.spinner(label):
                            result[name]=(*final_test(fit),*recursive_test(fit,12))
                        completed+=1
                    if protocols:
                        for name,protocol in protocols.items():
                            label="Comparando modelos a 12 semanas: "+name
                            evaluation_progress.progress(completed/total_steps,text=label)
                            with st.spinner(label):
                                protocol_test_results[name]=evaluate_protocol(protocol)
                            completed+=1
                    st.session_state["tests"]=result
                    st.session_state["protocol_test_results"]=protocol_test_results
                    evaluation_progress.progress(1.0,text="Evaluación completada")
                    st.success("Prueba evaluada. No se usó para elegir modelos.")
                except Exception as e:
                    st.error("Error en prueba: "+str(e))
        tests=st.session_state.get("tests")
        if tests:
            if protocols:
                st.subheader("Prueba del protocolo seleccionado sin reajustes")
                protocol_test_results=st.session_state.get("protocol_test_results",{})
                for name,protocol in protocols.items():
                    st.write(name+": elegido en validación = "+protocol['selected'])
                    if name not in protocol_test_results:
                        protocol_test_results[name]=evaluate_protocol(protocol)
                st.session_state["protocol_test_results"]=protocol_test_results
                for name,protocol in protocols.items():
                    protocol_test=protocol_test_results[name]
                    selected_test=protocol_test.loc[protocol_test['Modelo']==protocol['selected']]
                    st.markdown("**Evaluación rolling-origin: varios orígenes, horizonte de 12 semanas — "+name+"**")
                    st.dataframe(selected_test.round(3),hide_index=True,use_container_width=True)
                    fixed=protocol.get("fixed_test")
                    if fixed is not None and not fixed.empty:
                        st.markdown("**Evaluación fija: un único pronóstico de 12 semanas desde el inicio de TEST**")
                        st.caption("Origen: "+str(protocol.get("fixed_test_origin_date"))+
                                   ". No se incorporan ventas reales intermedias. Esta prueba es comparable con el diagnóstico anterior de primer origen.")
                        keep=[protocol["selected"],protocol.get("best_comparison"),"LSTM demanda original"]
                        keep=list(dict.fromkeys(x for x in keep if x))
                        fixed_show=fixed.loc[fixed["Modelo"].isin(keep)]
                        if fixed_show.empty:
                            fixed_show=fixed.head(5)
                        st.dataframe(fixed_show.round(3),hide_index=True,use_container_width=True)
            with st.expander("Anexo técnico: diagnóstico del motor anterior y referencias de prueba",expanded=False):
                    main_rows=[]
                    detail_rows=[]
                    for name,values in tests.items():
                        one,m,b,rolling,rm,rb=values
                        if name.startswith("Demanda"):
                            main_rows.append({"Serie":name,"Método":"LSTM recursiva (12 semanas)","Semanas":len(rolling),**rm})
                        detail_rows.extend([
                            {"Serie":name,"Método":"LSTM, un paso","Semanas":len(one),**m},
                            {"Serie":name,"Método":"Persistencia, un paso","Semanas":len(one),**b},
                            {"Serie":name,"Método":"LSTM, recursivo 12","Semanas":len(rolling),**rm},
                            {"Serie":name,"Método":"Persistencia fija, recursivo 12","Semanas":len(rolling),**rb}])
                    selected=st.session_state.get("demand_choice")
                    if selected:
                        chosen_table,chosen_metrics=evaluate_demand_test(fits["Demanda de corte propio"],selected,12)
                        if selected.method!="LSTM validada":
                            main_rows.append({"Serie":"Demanda de corte propio","Método":selected.method+" (elegido en validación)","Semanas":len(chosen_table),**chosen_metrics})
                        detail_rows.append({"Serie":"Demanda de corte propio","Método":selected.method+" elegido, recursivo 12","Semanas":len(chosen_table),**chosen_metrics})
                    st.subheader("Diagnóstico del motor anterior: prueba fija de 12 semanas")
                    st.caption("Este diagnóstico usa el primer bloque de TEST y corresponde al motor anterior. Compáralo con la tabla 'Evaluación fija' del protocolo actual, no con la evaluación rolling-origin.")
                    st.dataframe(pd.DataFrame(main_rows).round(3),hide_index=True,use_container_width=True)
                    if any(pd.notna(row.get("R²")) and row["R²"]<0 for row in main_rows):
                        st.info("Un R² negativo indica que la predicción no reproduce suficientemente la variación de esa serie. Se informa tal cual, sin modificar los datos de prueba.")
                    with st.expander("Ver todas las pruebas y métodos de referencia"):
                        st.dataframe(pd.DataFrame(detail_rows).round(3),hide_index=True,use_container_width=True)
                        st.caption("Un paso utiliza el valor real de la semana previa. La evaluación fija de 12 semanas no utiliza valores reales intermedios; rolling-origin sí actualiza el contexto con semanas reales al avanzar cada origen.")
                    choice=st.selectbox("Gráfico diagnóstico de prueba del motor anterior",["Demanda de maquila","Demanda de corte propio"],key="test_series")
                    test_df=tests[choice][3]
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=test_df.fecha_inicio,y=test_df.real,name="Real",mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=test_df.fecha_inicio,y=test_df.LSTM,name="LSTM recursiva",mode="lines+markers"))
                    fig.update_layout(template="plotly_white",height=310,yaxis_title="Pantalones/semana")
                    st.plotly_chart(fig,use_container_width=True)
                    st.caption("El R² puede ser negativo cuando no se anticipan bien las variaciones. Las métricas de TEST se conservan sin ajustes posteriores.")

with t4:
    tests=st.session_state.get("tests")
    if not fits or not tests:
        st.info("Para pronosticar, completa entrenamiento y prueba.")
    else:
        st.subheader("Pronóstico para planificación agregada y PMP")
        selection=st.session_state.get("demand_choice")
        refit_demands=st.checkbox("Reentrenar las demandas con todo el historial después de evaluar la prueba",value=True,
            help="Recomendado para el pronóstico operativo: conserva arquitectura y épocas elegidas en validación, reajusta con todas las semanas disponibles y no modifica las métricas de prueba ya calculadas.")
        if st.session_state.get('refit_demands')!=refit_demands:
            st.session_state['refit_demands']=refit_demands
            st.session_state.pop('forecast',None)
            st.session_state.pop('plan_preliminar',None)
        if protocols:
            for name,protocol in protocols.items():
                mode=". Reajuste final con todo el historial y épocas congeladas." if (not name.startswith('Demanda') or refit_demands) else ". Pesos y escala originales; recursión sin valores reales futuros."
                st.caption(name+": "+protocol['selected']+mode)
        elif selection:
            st.caption("Demanda CP: "+selection.method+" (elegido en validación del horizonte). Demanda MQ: LSTM.")
        cp_choice=st.session_state.get("cp_choice")
        if cp_choice is not None and not protocols:
            st.caption("Producción CP: "+cp_choice.method+" (modelo elegido en validación; uso descriptivo).")
        if int(weeks)>12:
            st.warning("La comparación de modelos se validó hasta 12 semanas. Para horizontes mayores, realiza una evaluación específica.")
        if st.button("Generar pronóstico",type="primary",use_container_width=True):
            try:
                forecasts={}
                for name,fit in fits.items():
                    if protocols:
                        forecasts[name]=future_search(protocols[name],start,int(weeks),
                            refit=(refit_demands if name.startswith('Demanda') else True)).rename(columns={'pronostico':fit.target})
                    elif name=="Demanda de corte propio" and selection:
                        forecasts[name]=forecast_demand_selected(fit,selection,start,int(weeks))
                    elif name=="Producción de corte propio" and st.session_state.get("cp_choice") is not None and st.session_state["cp_choice"].method!="LSTM original":
                        # La producción histórica no determina cantidades de planificación.
                        dates=forecasts["Demanda de maquila"]["fecha_inicio"]
                        forecasts[name]=forecast_intermitente(fit,st.session_state["cp_choice"],dates)
                    else:
                        forecasts[name]=forecast_lstm_verified(fit,start,int(weeks))
                st.session_state["forecast"]=make_forecast_table(forecasts)
                st.session_state.pop("plan_preliminar",None)
                st.session_state.pop("plan_signature",None)
                st.success("Pronóstico generado.")
            except Exception as e:
                st.error("No fue posible pronosticar: "+type(e).__name__+": "+str(e))
        forecast=st.session_state.get("forecast")
        if forecast is not None:
            view=forecast.copy()
            view["fecha_inicio"]=pd.to_datetime(view.fecha_inicio).dt.date
            view["fecha_fin"]=pd.to_datetime(view.fecha_fin).dt.date
            show=view.rename(columns={"fecha_inicio":"Semana desde","fecha_fin":"Semana hasta",
                "demanda_mq":"Demanda MQ","demanda_cp":"Demanda CP","demanda_total":"Demanda total",
                "produccion_cp":"Producción CP histórica estimada","produccion_total_referencia":"Producción total de referencia"})
            numeric=list(show.columns)[2:]
            show[numeric]=show[numeric].round(2)
            st.caption("Pronósticos puntuales, con dos decimales. El plan de fabricación realiza su propio redondeo a unidades enteras.")
            if protocols:
                weak=[n for n,p in protocols.items() if (p['test'].loc[p['test'].Modelo==p['selected'],'R²']<0).any()]
                if weak:
                    st.warning('R² negativo en la evaluación rolling-origin de: '+', '.join(weak)+'. Revisa también la evaluación fija de 12 semanas; ambas responden a escenarios de actualización distintos. Generar el pronóstico no certifica su precisión.')
            st.dataframe(show,hide_index=True,use_container_width=True)
            fig=go.Figure()
            for field,label in [("demanda_mq","Maquila"),("demanda_cp","Corte propio"),("demanda_total","Demanda total")]:
                fig.add_trace(go.Scatter(x=forecast.fecha_inicio,y=forecast[field],name=label,mode="lines+markers"))
            fig.update_layout(template="plotly_white",height=330,yaxis_title="Pantalones/semana")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Producción CP estadística = predicción del comportamiento empírico histórico, NO una orden de fabricación. El cálculo de reposición siguiente utiliza demanda, inventario y capacidad.")
            st.caption("Producción total de referencia supone MQ producida igual a su demanda. Es una suma descriptiva no verificada, no demanda total ni capacidad requerida del plan.")
            st.divider()
            st.subheader("Plan preliminar de reposición de corte propio")
            st.write("Diferente del pronóstico estadístico: calcula cantidades propuestas a partir de demanda, inventario y capacidad compartida.")
            capacity_file=st.file_uploader("Cargar 03_CAPACIDADES.xlsx para restricciones por recurso",type=['xlsx'],key='capacity_file')
            resources=None
            capacity_digest=None
            if capacity_file is not None:
                try:
                    capacity_payload=capacity_file.getvalue()
                    resources,capacity_checks=read_capacity(capacity_payload)
                    capacity_digest=sha256(capacity_payload).hexdigest()
                    if not capacity_checks.coincide.all():
                        st.warning("Hay diferencias entre capacidades guardadas y horas/tiempos. Se usan horas y tiempos; revisa la auditoría.")
                    st.caption("Capacidad compartida por máquina y trabajador del Excel; las capacidades de ambas modalidades no se suman.")
                    with st.expander('Auditoría de capacidad del Excel'):
                        st.dataframe(capacity_checks,hide_index=True)
                except Exception as e:
                    st.error('No se pudo validar capacidad: '+str(e)); st.stop()
            with st.expander("Supuestos de reposición y capacidad",expanded=True):
                left,right=st.columns(2)
                with left:
                    recorded_stock=base['inventario_final_cp'].iloc[-1] if 'inventario_final_cp' in base else None
                    stock_default=int(recorded_stock) if recorded_stock is not None and pd.notna(recorded_stock) and recorded_stock>=0 else None
                    stock=st.number_input("Inventario CP disponible al inicio (unidades)",min_value=0,value=stock_default,step=1,
                        help="Inicializado con el inventario final registrado en el Excel. Ajusta si el inventario físico disponible difiere.")
                    wash=st.number_input("CP ya confeccionados, aún en lavado (recibir la primera semana)",min_value=0,value=0,step=10)
                    cover=st.slider("Cobertura objetivo CP (semanas)",1,8,4)
                    minlot=st.number_input("Lote CP mínimo orientativo",min_value=0,value=80,step=10)
                with right:
                    capacity=st.number_input("Capacidad base semanal equivalente a maquila",min_value=1,value=420,step=10,disabled=resources is not None)
                    te_mq=st.number_input("TE maquila, min/unidad",min_value=0.01,value=22.04,step=0.01,disabled=resources is not None)
                    te_cp=st.number_input("TE corte propio, min/unidad",min_value=0.01,value=35.16,step=0.01,disabled=resources is not None)
                    cp_share=st.slider("Prioridad de capacidad para CP (%)",0,100,35,5,
                        help="Porcentaje de capacidad CP que se prioriza antes de asignar MQ; CP también puede utilizar la holgura posterior.")
            st.caption("Los valores por defecto son supuestos de escenario. Introduce el inventario físico y comprueba los tiempos estándar definitivos antes de usar el PMP.")
            st.caption("La reposición supone ingreso a bodega una semana después de confección y lavado. Cobertura, cuota CP y lote mínimo son políticas editables, no capacidades medidas.")
            plan=None
            if st.button("Calcular plan preliminar",type="secondary",use_container_width=True):
                try:
                    plan=calcular_plan_reposicion(
                        forecast,stock,wash,capacity,te_mq,te_cp,cover,cp_share,minlot,recursos=resources)
                    st.session_state["plan_preliminar"]=plan
                    st.session_state["plan_signature"]=(st.session_state.get("signature"),forecast_key,
                        stock,wash,capacity,te_mq,te_cp,cover,cp_share,minlot,capacity_digest)
                except Exception as exc:
                    st.error("No se pudo calcular el plan: "+str(exc))
            keys=(st.session_state.get("signature"),forecast_key,stock,wash,capacity,te_mq,te_cp,cover,cp_share,minlot,capacity_digest)
            if st.session_state.get("plan_signature")==keys:
                plan=st.session_state.get("plan_preliminar")
            if plan is not None:
                st.dataframe(plan,hide_index=True,use_container_width=True)
                left,right,third=st.columns(3)
                left.metric("CP propuesta (horizonte)",int(plan["Producción CP propuesta"].sum()))
                right.metric("Maquila pendiente al cierre",int(plan["Maquila pendiente al cierre"].iloc[-1]))
                third.metric("Ventas CP no cubiertas",int(plan["Ventas CP no cubiertas"].sum()))
                st.caption("Se comprueban recursos del Excel cuando está cargado. No se programa la secuencia de operaciones dentro de cada semana ni la capacidad externa de lavado. Una prioridad CP mayor puede desplazar maquila cuando los recursos están saturados; CP puede aprovechar holguras posteriores. Los lotes mínimos solo se señalan.")
                st.download_button("Descargar plan preliminar (CSV)",
                    data=plan.to_csv(index=False,sep=";").encode("utf-8-sig"),
                    file_name="PLAN_PRELIMINAR_REPOSICION_CP.csv")
            buffer=BytesIO()
            try:
                with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
                    show.to_excel(writer,sheet_name="PRONOSTICO",index=False)
                    zero_trace.to_excel(writer,sheet_name="CEROS_CONFIRMADOS",index=False)
                    if protocols:
                        pd.concat([p['validation'].assign(Serie=n) for n,p in protocols.items()]).to_excel(writer,sheet_name="VALIDACION_12",index=False)
                        pd.concat([evaluate_protocol(p).assign(Serie=n) for n,p in protocols.items()]).to_excel(writer,sheet_name="PRUEBA_12",index=False)
                        pd.DataFrame([{'Serie':n,'Modelo':p['selected'],'Epocas':p['candidates'][p['selected']].epochs,
                                       'Smearing':p['candidates'][p['selected']].smear,'Criterio':'RMSE a 12 semanas',
                                       'Refit':('Todo el historial, después de prueba' if not n.startswith('Demanda') or refit_demands else 'Sin reajuste; pesos y escala evaluados')} for n,p in protocols.items()]).to_excel(writer,sheet_name="SELECCION_FINAL",index=False)
                    if plan is not None:
                        plan.to_excel(writer,sheet_name="PLAN_REPOSICION",index=False)
                        if resources is not None:
                            resources.to_excel(writer,sheet_name='RECURSOS_CAPACIDAD',index=False)
                            resource_loads(plan,resources).to_excel(writer,sheet_name='CARGA_RECURSOS',index=False)
                    if plan is not None:
                        plan.to_excel(writer,sheet_name="PLAN_PRELIMINAR",index=False)
                    metric_rows=[]
                    val_tables=[];test_tables=[]
                    for name,f in fits.items():
                        for phase,m in [("Entrenamiento",f.train_metrics),("Validación",f.validation_metrics),
                                        ("Referencia validación",f.baseline_validation_metrics),
                                        ("Prueba",tests[name][1]),("Referencia prueba",tests[name][2]),
                                        ("Prueba recursiva LSTM",tests[name][4]),
                                        ("Referencia recursiva",tests[name][5])]:
                            metric_rows.append({"Serie":name,"Etapa":phase,**m})
                        v=f.validation.copy();v.insert(0,"Serie",name);val_tables.append(v)
                        t=tests[name][0].copy();t.insert(0,"Serie",name);test_tables.append(t)
                    if selection:
                        result,m=evaluate_demand_test(fits["Demanda de corte propio"],selection,12)
                        metric_rows.append({"Serie":"Demanda de corte propio","Etapa":"Prueba recursiva método elegido: "+selection.method,**m})
                        selection.comparison.to_excel(writer,sheet_name="COMPARACION_CP",index=False)
                    pd.DataFrame(metric_rows).assign(Alcance="Diagnóstico motor anterior; selección final en PRUEBA_12").to_excel(writer,sheet_name="METRICAS",index=False)
                    pd.concat(val_tables,ignore_index=True).to_excel(writer,sheet_name="VALIDACION",index=False)
                    pd.concat(test_tables,ignore_index=True).to_excel(writer,sheet_name="PRUEBA",index=False)
                    meta=pd.DataFrame([("Archivo",file.name),("SHA256",sha256(payload).hexdigest()),
                        ("Origen declarado",", ".join(info["fuentes_declaradas"]) if info["fuentes_declaradas"] else "No declarado"),
                        ("Fin de historial",str(info["fecha_ultima_semana"])),
                        ("Modelo demanda MQ",protocols["Demanda de maquila"]["selected"] if protocols else "LSTM validada"),
                        ("Modelo demanda CP",protocols["Demanda de corte propio"]["selected"] if protocols else (selection.method if selection else "LSTM")),
                        ("Ventana",settings.lookback),("Épocas máximas",settings.epochs),("Unidades LSTM",settings.units),
                        ("Semilla",settings.seed),("Entrenamiento",train),("Validación",val),
                        ("Horizonte",int(weeks)),("Inicio plan",str(start)),
                        ("Nota","Producción CP estimada es informativa; planificación requiere inventarios y capacidad.")],columns=["Dato","Valor"])
                    meta.to_excel(writer,sheet_name="FICHA_TECNICA",index=False)
                    for ws in writer.book.worksheets:
                        ws.freeze_panes="A2";ws.auto_filter.ref=ws.dimensions
                        ws.row_dimensions[1].height=28
                        from openpyxl.styles import PatternFill,Font
                        for c in ws[1]:
                            c.fill=PatternFill("solid",fgColor="FFF200")
                            c.font=Font(color="111111",bold=True)
                        for col in ws.columns:
                            from openpyxl.utils import get_column_letter
                            width=min(40,max(12,max(len(str(c.value or "")) for c in list(col)[:40])+2))
                            ws.column_dimensions[get_column_letter(col[0].column)].width=width
                st.download_button("Descargar pronóstico y reporte de tesis (Excel)",data=buffer.getvalue(),
                    file_name="PRONOSTICO_LSTM_PARA_PLANIFICACION.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",use_container_width=True)
            except Exception as e:
                st.error("No se pudo generar el reporte Excel: "+str(e))
            st.download_button("Descargar pronóstico CSV",data=show.to_csv(index=False,sep=";").encode("utf-8-sig"),
                               file_name="PRONOSTICO_PARA_PLANIFICACION.csv",mime="text/csv")
st.divider()
st.caption("Investigación de Ingeniería Industrial · Evaluar capacidad, inventarios y costos antes de convertir demanda pronosticada en órdenes de producción.")
