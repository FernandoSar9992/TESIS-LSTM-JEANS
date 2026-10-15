from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from demand_data import FREQUENCIES, prepare_series, read_upload, detect_column
from thesis_engine import ThesisConfig as LSTMConfig, run_thesis as run_lstm, split_points, finalize_test, generate_future, ENGINE_VERSION
from thesis_report import export_thesis
from presentation import readable_table, data_preview
from workshop_data import workshop_columns

st.set_page_config(page_title="Pronóstico de la demanda", page_icon="📈", layout="wide")
st.title("Pronóstico de la demanda")
st.caption("LSTM · Validación temporal, trazabilidad y comparación de modelos")


@st.cache_data(show_spinner=False, max_entries=5)
def sheets(content):
    with pd.ExcelFile(BytesIO(content)) as book:
        return book.sheet_names


@st.cache_data(show_spinner=False, max_entries=5)
def load_file(content, name, sheet, decimal):
    return read_upload(content, name, sheet, decimal)


with st.sidebar:
    st.header("Tu pronóstico LSTM")
    uploaded = st.file_uploader("Cargar Excel o CSV", type=["xlsx", "csv"], key="source_upload")
    horizon = st.number_input("Períodos a pronosticar", 1, 120, 8, key="horizon",
                              help="Se expresa en la frecuencia de tus datos: días, semanas, meses o períodos ordenados.")
    if uploaded is None:
        st.info("Carga un archivo y selecciona su período y demanda.")
        st.stop()
    try:
        content = uploaded.getvalue()
        declaration_source = sha256(content).hexdigest()
        if st.session_state.get('declaration_source') != declaration_source:
            st.session_state['declared_real'] = False
            st.session_state['declaration_source'] = declaration_source
        sheet = st.selectbox("Hoja del Excel", sheets(content)) if uploaded.name.lower().endswith(".xlsx") else 0
        decimal = st.selectbox("Separador decimal", [".", ","], help="Para CSV o celdas numéricas guardadas como texto.")
        raw = load_file(content, uploaded.name, sheet, decimal)
        if raw.empty or len(raw.columns) < 2:
            raise ValueError("El archivo debe tener una columna de período, una de demanda y registros.")
    except Exception as exc:
        st.error(f"No se pudo leer el archivo: {exc}")
        st.stop()
    columns = list(raw.columns)
    suggested_period = detect_column(columns, "Etiqueta de Semana", "Semana", "Periodo", "Fecha")
    period_col = st.selectbox("Columna de período o fecha", columns, index=columns.index(suggested_period) if suggested_period is not None else 0, key="period_col")
    targets = [col for col in columns if col != period_col]
    numeric = [col for col in targets if pd.api.types.is_numeric_dtype(raw[col])]
    suggested_target = detect_column(targets, "Demanda_Final", "Demanda", "Demanda (unidades)")
    demand_col = st.selectbox("Columna de demanda", targets, index=targets.index(suggested_target) if suggested_target is not None else targets.index(numeric[0]) if numeric else 0, key="demand_col")
    workshop_mapping = workshop_columns(raw)
    is_workshop = all(workshop_mapping.values())
    if not is_workshop:
        with st.expander("Preparar la serie"):
            group_options = [None] + [col for col in columns if col not in (period_col, demand_col)]
            group_col = st.selectbox("Filtrar producto, tienda u otra serie", group_options,
                                     format_func=lambda value: "Sin filtro" if value is None else str(value))
            group_value = None
            selected_frame = raw
            if group_col is not None:
                values = raw[group_col].dropna().unique().tolist()
                if not values:
                    st.error("La columna del filtro no tiene valores.")
                    st.stop()
                group_value = st.selectbox("Serie a pronosticar", values, format_func=str)
                selected_frame = raw.loc[raw[group_col] == group_value]
            is_date = pd.api.types.is_datetime64_any_dtype(raw[period_col]) or raw[period_col].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").all()
            mode = st.selectbox("Formato del período", ["Etiquetas en el orden del archivo", "Fechas: día/mes/año", "Fechas: mes/día/año"], index=1 if is_date else 0, key="period_mode")
            frequency = None
            fill_zero = False
            if mode.startswith("Fechas"):
                frequency_name = st.selectbox("Frecuencia de la demanda", list(FREQUENCIES), key="frequency")
                frequency = FREQUENCIES[frequency_name]
                if frequency is not None:
                    st.caption("Se suman los registros de cada período. Filtra antes si el archivo contiene varias series.")
                    fill_zero = st.checkbox("Los períodos sin registros significan demanda cero", value=False)
            else:
                st.caption("Las filas deben estar ordenadas, sin saltos y con la misma frecuencia. No se impone un calendario al modelo.")
            declared_real = False
            if detect_column(columns, "Origen") is None:
                declared_real = st.checkbox("Confirmo que todos los registros cargados son observaciones reales", value=False, key="declared_real",
                                            help="Marca solo si no hay demandas estimadas o reconstruidas. Se conserva como declaración del usuario.")

if is_workshop:
    from workshop_app import render_workshop
    workshop_mapping.update(period=period_col, target=demand_col)
    # El usuario confirmó la procedencia real de los datos de su taller.
    # Siempre usar la serie completa; ninguna selección anterior filtra clientes.
    render_workshop(raw, workshop_mapping, horizon, declared_real=True)
    st.stop()

try:
    prepared = prepare_series(selected_frame, period_col, demand_col, date_mode=mode.startswith("Fechas"),
                              dayfirst=mode != "Fechas: mes/día/año", frequency=frequency, fill_missing_zero=fill_zero, declared_real=declared_real)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
data = prepared.data

with st.sidebar:
    automatic = st.checkbox("Ajustar LSTM automáticamente", value=True, key="automatic",
                            help="Compara ventanas, redes pequeñas y formas de representar la demanda. Selección por RMSE de validación temporal.")
    lookback, units = 8, 16
    with st.expander("Opciones LSTM"):
        auto_representation = st.checkbox("Comparar formas de aprendizaje", value=True, disabled=not automatic,
                                          help="Compara nivel, cambio y variabilidad reciente en las variantes básicas. Las variantes estacionales adicionales usan representación local cuando hay historia suficiente.")
        representation = st.selectbox("Aprendizaje", ["local", "nivel", "cambio"],
                                       disabled=automatic and auto_representation,
                                       format_func=lambda v: {"local": "Nivel y variabilidad de la ventana reciente", "nivel": "Nivel con escala de TRAIN", "cambio": "Cambio respecto al último valor"}[v])
        if not automatic:
            lookback = st.number_input("Ventana histórica", 2, 24, 8)
            units = st.number_input("Unidades LSTM", 8, 32, 16, step=8)
        extended = st.checkbox("Incluir ventanas 16 y 24 si hay suficientes secuencias", value=False)
        epochs = st.number_input("Máximo de épocas", 10, 500, 300, key="epochs")
        patience = st.number_input("Paciencia de parada temprana", 3, 50, 20)
        dropout = st.slider("Dropout manual", 0.0, 0.6, 0.1, .05, disabled=automatic,
                            help="En automático se comparan 0, 0.1 y 0.2 según la arquitectura.")
        learning_rate = st.number_input("Tasa de aprendizaje", .00001, .1, .001, format="%.5f")
        batch_size = st.selectbox("Tamaño de lote", [8, 16, 32, 64], index=1)
        reduce_lr = st.checkbox("Reducir tasa si se estanca la validación", value=True)
        average = st.checkbox("Promediar tres entrenamientos LSTM", value=False,
                              help="Reduce dependencia de una sola inicialización. Tarda más; no garantiza mejorar el error.")
        seed = st.number_input("Semilla", 0, 99999, 42)
        seasonal_features = st.checkbox("Comparar entradas estacionales", value=True, key="seasonal_features",
                                        help="Incluye calendario seno/coseno y demanda de hace un ciclo. En automático compite contra variantes sin estacionalidad; no impone oscilaciones.")
        refit_future = st.checkbox("Reajustar para el futuro con todo el histórico", value=True,
                                   help="Después de medir TEST, entrena un modelo separado con configuración y épocas ya fijadas. Conserva las métricas del modelo evaluado.")
    with st.expander("Partición temporal y periodicidad"):
        train_pct = st.slider("TRAIN (%)", 50, 85, 70)
        val_pct = st.slider("VALIDATION (%)", 5, min(30, 95-train_pct), min(15, 95-train_pct))
        periodicity = int(st.number_input("Períodos por ciclo estacional", 2, 366, 48, key="periodicity",
                                         help="48 para cuatro períodos por mes, 52 para semanas, 12 para meses. Configura según tu frecuencia real."))
        st.caption("Esta periodicidad se usa en las entradas estacionales y en la referencia estacional. Cambiarla requiere entrenar de nuevo.")
    run = st.button("Entrenar y validar", type="primary", width="stretch", key="run")

config = LSTMConfig(horizon=int(horizon), automatic=automatic, lookback=int(lookback), units=int(units),
                    representation=representation, epochs=int(epochs), patience=int(patience), dropout=float(dropout),
                    learning_rate=float(learning_rate), batch_size=int(batch_size), members=3 if average else 1, seed=int(seed),
                    train_ratio=train_pct/100, validation_ratio=val_pct/100, periodicity=periodicity,
                    extended_windows=extended, reduce_lr=reduce_lr, auto_representation=auto_representation,
                    seasonal_features=seasonal_features, refit_future=refit_future)
preparation = {"archivo": uploaded.name, "hoja": sheet, "periodo": str(period_col), "demanda": str(demand_col),
               "filtro": str(group_col), "valor_filtro": str(group_value), "formato": mode,
               "frecuencia": prepared.frequency, "ceros_ausentes": fill_zero, "decimal": decimal, "declaracion_todos_reales": declared_real}
training_config = {key: value for key, value in asdict(config).items() if key != "horizon"}
signature = sha256((ENGINE_VERSION + data.to_csv(index=False) + json.dumps(training_config, sort_keys=True) + json.dumps(preparation, sort_keys=True)).encode()).hexdigest()
if st.session_state.get("result") and st.session_state.result.get("signature") != signature:
    st.session_state.result = None
    st.info("La fuente o configuración cambió. Entrena de nuevo para actualizar el pronóstico.")
if run:
    st.session_state.result = None
    try:
        progress = st.empty()
        with st.spinner("Entrenando y comparando validación; TEST permanece reservado…"):
            result = run_lstm(data, config, prepared.future_labels(config.horizon, periodicity), progress=progress.info, evaluate_test=False)
        progress.empty()
        result.update(signature=signature, preparation=preparation)
        st.session_state.result = result
    except Exception as exc:
        st.error(f"No se pudo completar el entrenamiento: {exc}")
result = st.session_state.get("result")
if result and (result["config"]["horizon"] != config.horizon or result["config"]["periodicity"] != periodicity):
    result["config"].update(horizon=config.horizon, periodicity=periodicity)
    result["_artifacts"]["labels"] = prepared.future_labels(config.horizon, periodicity)
    if result["evaluated"]:
        try:
            generate_future(result)
            result["report_bytes"] = export_thesis(result)
        except ValueError as exc:
            result.pop("forecast", None)
            result.pop("report_bytes", None)
            st.error(str(exc))
st.session_state.setdefault("evaluated_tests", set())
if result and not result["evaluated"]:
    test_key = f"{result['data_hash']}:{result['val_end']}"
    already = test_key in st.session_state.evaluated_tests
    st.info("Revisa la validación y fija tu configuración antes de evaluar TEST. El pronóstico futuro se genera después.")
    if already:
        st.warning("Este TEST ya fue consultado en esta sesión. Puedes revisar validación, pero necesitas observaciones nuevas para otra evaluación final independiente.")
    if st.button("Evaluar TEST final y generar pronóstico", key="final_test", disabled=already, type="primary"):
        try:
            # Registrar antes de la operación impide reutilizar una prueba parcialmente expuesta.
            st.session_state.evaluated_tests.add(test_key)
            with st.spinner("Evaluando TEST una vez y calculando el futuro…"):
                result = finalize_test(result)
            result["report_bytes"] = export_thesis(result)
            st.session_state.result = result
        except Exception as exc:
            st.error(f"No se pudo terminar la evaluación final: {exc}")

forecast_tab, evaluation_tab, comparison_tab, diagnostic_tab, data_tab = st.tabs([
    "Pronóstico LSTM", "Evaluación LSTM", "Comparación", "Diagnóstico del modelo", "Datos"])
final = bool(result and result.get("evaluated") and "forecast" in result)
with data_tab:
    st.subheader("Tu histórico de demanda")
    st.caption(f"{len(data)} períodos · {data.periodo.iloc[0]} — {data.periodo.iloc[-1]}")
    a, b, c = st.columns(3)
    with a.container(border=True):
        st.metric("Períodos cargados", len(data))
    with b.container(border=True):
        st.metric("Demanda promedio", f"{data.demanda.mean():,.2f}")
    with c.container(border=True):
        st.metric("Última demanda", f"{data.demanda.iloc[-1]:,.2f}")
    st.caption("Demanda expresada en las unidades de tu archivo.")
    imputed = int(data.origen.isin(["IMPUTADO", "MIXTO"]).sum())
    unknown = int(data.origen.eq("SIN_DECLARAR").sum())
    if imputed/len(data) > .5:
        st.warning("Más del 50 % de la serie es imputado o mixto. Es una limitación importante: el modelo puede aprender patrones de la reconstrucción.")
    elif imputed/len(data) > .3:
        st.warning("Más del 30 % de la serie es imputado o mixto. Interpreta las métricas sobre reconstrucciones como descriptivas.")
    if unknown == len(data):
        st.caption("El archivo no especifica si los valores son observados o reconstruidos. Ese detalle no impide entrenar, pero limita la interpretación de las métricas.")
    st.dataframe(data_preview(data), hide_index=True, width="stretch", height=450,
                 column_config={"Período": st.column_config.TextColumn(width="large"),
                                "Demanda": st.column_config.NumberColumn(format="%.2f")})
    with st.expander("Procedencia y preparación de los datos", icon=":material/fact_check:"):
        if unknown == len(data):
            st.write("La columna Origen no aporta una clasificación de los registros. No se asumen como reales ni como imputados.")
        else:
            st.write(f"Observados: {int(data.origen.eq('REAL').sum())} · Reconstruidos o mixtos: {imputed} · Sin clasificación en el archivo: {unknown}.")
            st.caption(f"Reconstrucción declarada: {100*imputed/len(data):.2f} % del histórico.")
        for note in prepared.notes:
            if not note.startswith("No existe Origen"):
                st.caption(note)
        trace = data.drop(columns=['origen']) if unknown == len(data) else data
        trace = trace.drop(columns=['tipo_estimacion']) if trace.tipo_estimacion.eq('').all() else trace
        st.dataframe(readable_table(trace), hide_index=True, width="stretch")
        st.caption("La descarga conserva los valores originales, la demanda utilizada y la trazabilidad completa.")
    if result:
        st.subheader("Distribución temporal")
        st.dataframe(readable_table(result["split_table"]), hide_index=True, width="stretch")
        processed = result["processed"]
        st.plotly_chart(px.scatter(processed, x="Periodo", y="Demanda_utilizada", color="Grupo",
                                  title="Partición cronológica TRAIN / VALIDATION / TEST",
                                  labels={"Demanda_utilizada": "Demanda (unidades)"}), width="stretch")
with forecast_tab:
    a, b, c = st.columns(3)
    a.metric("Períodos históricos", len(data))
    b.metric("Demanda promedio", f"{data.demanda.mean():,.2f}")
    c.metric("Horizonte futuro", config.horizon)
    historical = data_preview(data)
    if "Procedencia" in historical:
        chart = px.scatter(historical, x="Período", y="Demanda", color="Procedencia", title="Histórico de demanda")
    else:
        chart = px.line(historical, x="Período", y="Demanda", title="Histórico de demanda")
    chart.update_yaxes(title="Demanda (unidades)")
    st.plotly_chart(chart, width="stretch")
    if final:
        st.subheader("Demanda futura estimada por LSTM")
        score = result['metrics'].set_index('Modelo').loc['LSTM']
        if np.isfinite(score.R2) and score.R2 <= 0:
            st.warning(f"Precisión no demostrada: R² de TEST = {score.R2:.3f}, MAE = {score.MAE:.2f} unidades. Este resultado no permite asegurar un pronóstico preciso. Revisa también los errores por horizonte.")
        st.plotly_chart(px.line(result["forecast"], x="Periodo", y="LSTM", markers=True,
                               title="Pronóstico futuro recursivo", labels={"LSTM": "Demanda estimada (unidades)"}), width="stretch")
        st.dataframe(result["forecast"], hide_index=True, width="stretch")
        st.caption(f"Referencia elegida en validación: {result['reference_model']}. Su columna se presenta separada de LSTM.")
        if result['config']['refit_future']:
            st.caption("El futuro usa un reajuste separado con todo el histórico. Las métricas de TRAIN, VALIDATION y TEST corresponden al modelo de evaluación, congelado antes de TEST.")
        if config.horizon > result.get('backtest_horizon', 0):
            st.warning(f"Solo se evaluaron trayectorias de hasta {result.get('backtest_horizon', 0)} períodos. El horizonte solicitado es mayor; su error no ha sido medido.")
        st.caption("Cada predicción futura alimenta el siguiente paso. El TEST evalúa un paso con pasado observado; no certifica la precisión de todo este horizonte recursivo.")
        if "report_bytes" in result:
            st.download_button("Descargar informe Excel", result["report_bytes"], "pronostico_LSTM_tesis.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    elif result:
        st.info("La configuración está validada. El futuro se calculará después de evaluar TEST.")
    else:
        st.info("Carga y revisa los datos, configura LSTM y pulsa «Entrenar y validar».")
        try:
            tr, va = split_points(len(data), 1, config.train_ratio, config.validation_ratio)
            st.caption(f"Reserva temporal: {tr} TRAIN, {va-tr} VALIDATION y {len(data)-va} TEST. El horizonte futuro no altera esta partición.")
        except ValueError as exc:
            st.warning(str(exc))
with evaluation_tab:
    if result:
        chosen = result["selected"]
        st.info(f"LSTM elegida por validación: ventana {chosen['lookback']}, {chosen['units']} unidades, dropout {chosen['dropout']}, representación {chosen['representation']}.")
        st.caption(f"Entradas estacionales: {'sí, ciclo de ' + str(chosen['periodicity']) if chosen.get('periodicity') else 'no; no seleccionadas o desactivadas'}.")
        if chosen['epochs'] == 1:
            st.warning("La mejor época fue la primera: más entrenamiento no mejoró la validación en este corte. Una red local puede quedar próxima al promedio reciente; esto no demuestra aprendizaje de los picos.")
        st.caption(f"Secuencias: {result['train_windows']} TRAIN, {result['validation_origins']} VALIDATION, {result['test_origins']} TEST. Salida de un período por secuencia.")
        lstm_groups = result["split_metrics"].query("Modelo == 'LSTM'")
        st.dataframe(readable_table(lstm_groups[["Grupo", "Alcance", "N", "MAE", "RMSE", "sMAPE_%", "R2"]]), hide_index=True, width="stretch")
        st.caption("Las métricas distinguen observaciones reales, procedencia no indicada y evaluación descriptiva de valores que pueden ser reconstruidos.")
        if final:
            score = result["metrics"].set_index("Modelo").loc["LSTM"]
            a, b, c = st.columns(3)
            a.metric("RMSE LSTM — TEST", f"{score['RMSE']:.2f}")
            b.metric("MAE LSTM — TEST", f"{score['MAE']:.2f}")
            c.metric("R² LSTM — TEST", f"{score['R2']:.3f}" if np.isfinite(score['R2']) else "No definido")
            if score["R2"] < 0:
                st.warning("El modelo presenta un desempeño inferior a utilizar la media del conjunto de prueba como predictor de referencia. Esa media es retrospectiva, no conocida al pronosticar.")
            detail = result["detail"]
            st.plotly_chart(px.line(detail, x="Periodo", y=["Demanda_real", "Prediccion_LSTM"], markers=True,
                                   title="TEST: objetivo registrado frente a LSTM (un paso)", labels={"value": "Demanda (unidades)", "variable": "Serie"}), width="stretch")
            st.caption("La línea del objetivo incluye todo el registro; consulte Origen para distinguir observado de reconstruido.")
            st.plotly_chart(px.bar(readable_table(detail), x="Período", y="Error", color="Procedencia", title="Residuos LSTM: objetivo menos predicción",
                                  labels={"Error": "Error (unidades)"}), width="stretch")
            with st.expander("Predicciones y error por período"):
                st.dataframe(readable_table(detail), hide_index=True, width="stretch")
            st.subheader("Error de trayectorias futuras de varios pasos")
            st.dataframe(readable_table(result['horizon_metrics'].query("Grupo == 'TEST'")), hide_index=True, width="stretch")
            st.caption("Cada trayectoria parte del pasado disponible y se alimenta de sus propias predicciones. Los orígenes se solapan: N no representa experimentos independientes. VALIDATION se usó para seleccionar el modelo.")
        with st.expander("Entrenamiento, búsqueda y cortes temporales", expanded=not final):
            st.dataframe(pd.DataFrame([result["training_summary"]]), hide_index=True, width="stretch")
            history = pd.DataFrame({"TRAIN": result["history"]["loss"], "VALIDATION": result["history"]["val_loss"]})
            history["Época"] = np.arange(1, len(history)+1)
            st.plotly_chart(px.line(history, x="Época", y=["TRAIN", "VALIDATION"], title="Pérdida de entrenamiento y validación",
                                   labels={"value": "MSE en escala de TRAIN", "variable": "Grupo"}), width="stretch")
            st.dataframe(readable_table(result["search"]), hide_index=True, width="stretch")
            st.dataframe(readable_table(result["validation_folds"]), hide_index=True, width="stretch")
            st.caption("RMSE promedio selecciona; MAE desempata. Curva y época corresponden a la red de la semilla principal. Scaler y pesos de evaluación permanecen ajustados solo en TRAIN; el reajuste futuro es independiente.")
    else:
        st.info("Aquí aparecerán las métricas de aprendizaje y validación; TEST se consulta después de fijar la configuración.")
with comparison_tab:
    if final:
        st.subheader("Todos los modelos sobre los mismos objetivos TEST")
        st.dataframe(readable_table(result["metrics"]), hide_index=True, width="stretch")
        comparison = result["metrics"].melt(id_vars="Modelo", value_vars=["MAE", "RMSE"], var_name="Métrica", value_name="Error")
        st.plotly_chart(px.bar(comparison, x="Modelo", y="Error", color="Métrica", barmode="group",
                              title="Comparación de errores fuera de muestra", labels={"Error": "Error (unidades)"}), width="stretch")
        st.caption("El orden por RMSE describe TEST; no se utiliza para volver a seleccionar hiperparámetros.")
        with st.expander("Separar observaciones reales de la evaluación descriptiva"):
            st.markdown("**Solo objetivos REAL**")
            st.dataframe(readable_table(result["metrics_real"]), hide_index=True, width="stretch")
            st.markdown("**Todos los objetivos — descriptivo, puede incluir imputados**")
            st.dataframe(readable_table(result["metrics_all"]), hide_index=True, width="stretch")
    elif result:
        st.subheader("Comparación en VALIDATION")
        st.dataframe(readable_table(result["split_metrics"].query("Grupo == 'VALIDATION'")), hide_index=True, width="stretch")
    else:
        st.info("Se compararán Naive, medias móviles de 4/8/12, Ridge y LSTM.")
with diagnostic_tab:
    if result:
        st.dataframe(result["diagnostic"], hide_index=True, width="stretch")
        st.dataframe(result["data_profile"], hide_index=True, width="stretch")
        st.caption("Las reglas indican posibles causas; no demuestran causalidad ni significancia estadística. TEST ya consultado requiere datos nuevos para una evaluación independiente posterior.")
    else:
        st.info("El diagnóstico incluirá trazabilidad, muestras efectivas, aprendizaje y cambios de nivel.")
