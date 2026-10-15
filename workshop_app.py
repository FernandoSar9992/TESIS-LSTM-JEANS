"""Interfaz del modo taller. Se importa después de cargar un archivo compatible."""
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from workshop_data import prepare_workshop, feature_audit
from workshop_engine import WorkshopConfig, train_workshop, finalize_workshop, forecast_workshop, ENGINE_VERSION
from presentation import readable_table
from workshop_planning import (planning_forecast, PLANNING_NOTE, CLIENT_VERSION,
                               select_client_model, evaluate_client_model)


def export_workshop(result):
    """Descarga de la aplicación, con separación explícita de los modelos."""
    if not result['evaluated']:
        raise ValueError('Evalúe TEST antes de descargar el informe final.')
    output = BytesIO()
    frames = {'Datos': result['processed'], 'Metricas_un_paso': result['split_metrics'],
              'Error_por_horizonte': result['horizon_metrics'], 'Predicciones': result['detail'],
              'Seleccion': result['search'], 'Cortes_validacion': result['validation_folds'],
              'Variables': result['feature_audit'],
              'Configuracion': pd.DataFrame([{**result['config'], 'Modelo_seleccionado': result['selected'],
                                             'SHA256_datos': result['data_hash'], 'Motor': result['engine_version'],
                                             'Modelo_principal_tesis': result.get('thesis_selected', result.get('lstm_name', result['selected']))}]),
              'Metodologia': pd.DataFrame({'Nota': [
                  'Selección por RMSE medio de trayectorias completas en hasta tres cortes anteriores a TEST.',
                  'La salida principal es el promedio de tres LSTM estacionales. Se comparan ventanas 12/24, 4/8 unidades, pérdidas MSE/MAE/Huber y exclusión de variables comerciales, cambiando un factor por variante.',
                  'Las épocas se eligen en un bloque interno de TRAIN; cada corte externo se utiliza solo para puntuar y seleccionar configuraciones.',
                  'MASE usa el error absoluto del rezago estacional exclusivamente en TRAIN. Si el denominador no existe o es cero, se reporta vacío.',
                  'RMSSE usa el error cuadrático del rezago estacional exclusivamente en TRAIN. No equivale al cociente con el error del benchmark en TEST.',
                  'La evaluación de un paso utiliza todos los objetivos del bloque. La evaluación multihorizonte usa trayectorias completas solapadas; sus N son distintos.',
                  'No se utilizan ingreso total, costo total ni utilidad como entradas; dependen de la cantidad objetivo.',
                  'Modo histórico: solo clientes, precios y costos anteriores al origen; no se supone conocido el cliente futuro.',
                  PLANNING_NOTE,
                  'Cada origen pronostica todos sus horizontes directamente, sin introducir demandas observadas intermedias.',
                  'El modelo para el futuro se reajusta después de congelar TEST, con las épocas y configuración fijadas en validación.',
                  'VALIDATION se usó para seleccionar; no es una prueba independiente. No hay garantía de R² positivo.',
                  'La salida de planificación contiene período, demanda pronosticada y cliente predicho.']})}
    if 'forecast' in result:
        frames['Pronostico'] = result.get('planning_forecast', result['forecast'])
    if 'client_model' in result:
        frames['Evaluacion_cliente'] = result['client_model']['metrics']
        frames['Configuracion']['Modelo_cliente'] = result['client_model']['selected']
        frames['Configuracion']['Motor_cliente'] = CLIENT_VERSION
    if 'seed_metrics' in result:
        frames['Estabilidad_semillas'] = result['seed_metrics']
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()


def render_workshop(frame, mapping, horizon, declared_real):
    st.subheader('Pronóstico multivariable del taller')
    st.caption('Todos los clientes · demanda total del taller · observaciones reales confirmadas por el usuario')
    try:
        prepared = prepare_workshop(frame, mapping, declared_real)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    data = prepared.data
    with st.sidebar:
        epochs = int(st.number_input('Máximo de épocas multivariable', 10, 500, 180, key='workshop_epochs'))
        periodicity = int(st.number_input('Períodos por ciclo del taller', 2, 366, 48, key='workshop_periodicity'))
        st.caption('El pronóstico principal usa LSTM. Se comparan sus variantes con referencias y Ridge; tres semillas permiten revisar su estabilidad.')
        run = st.button('Entrenar y validar taller', key='workshop_run', type='primary', width='stretch')
    config = WorkshopConfig(horizon=int(horizon), mode='historico', epochs=epochs, periodicity=periodicity)
    signature = sha256((ENGINE_VERSION+data.to_csv(index=False)+json.dumps(asdict(config), sort_keys=True)).encode()).hexdigest()
    result = st.session_state.get('workshop_result')
    if result and result.get('signature') != signature:
        st.session_state.workshop_result = result = None
        st.info('Los datos o la configuración cambiaron. Entrena de nuevo.')
    st.caption('Ingreso total, costo total y utilidad se excluyen como entradas. Se utiliza la serie completa de todos los clientes, con una fila por período del taller.')
    if run:
        st.session_state.workshop_result = None
        try:
            with st.status('Entrenando con TEST reservado…', expanded=True) as status:
                result = train_workshop(data, config, progress=st.write)
                status.update(label='Validación terminada', state='complete')
            result['signature'] = signature
            result['preparation_notes'] = prepared.notes
            st.session_state.workshop_result = result
        except Exception as exc:
            st.error(f'No se pudo entrenar: {exc}')
            result = None
    if result:
        if result.get('client_model', {}).get('version') != CLIENT_VERSION:
            result['client_model'] = select_client_model(data, result['train_end'], result['val_end'], config.horizon, config.periodicity)
        if result['evaluated']:
            evaluate_client_model(result['client_model'], data, result['val_end'])
    st.session_state.setdefault('workshop_test_ledger', set())
    if result and not result['evaluated']:
        # El modo y la procedencia no convierten el mismo objetivo en TEST nuevo.
        test_key = sha256(data[['periodo', 'demanda']].to_csv(index=False).encode()).hexdigest()+f":{result['val_end']}"
        used = test_key in st.session_state.workshop_test_ledger
        st.caption('Revisa la validación antes de abrir TEST. Los cambios posteriores requieren observaciones nuevas para una evaluación independiente.')
        if used:
            st.warning('Este bloque TEST ya se consultó en esta sesión.')
        if st.button('Evaluar TEST final del taller', key='workshop_final', disabled=used, type='primary'):
            try:
                with st.status('Evaluando TEST y reajustando para el futuro…'):
                    finalize_workshop(result, progress=st.write)
                    evaluate_client_model(result['client_model'], data, result['val_end'])
                st.session_state.workshop_test_ledger.add(test_key)
                st.session_state.workshop_result = result
            except Exception as exc:
                st.error(f'No se pudo completar la evaluación: {exc}')
    forecast_tab, summary_tab, metrics_tab, data_tab = st.tabs(
        ['Pronóstico del taller', 'Métricas', 'Validación y prueba', 'Datos y variables'])
    with summary_tab:
        st.subheader('Métricas de demanda — LSTM')
        if result:
            thesis_model = result.get('thesis_selected', result['lstm_name'])
            st.caption(f'Modelo evaluado: {thesis_model}')
            columns = ['Grupo', 'N', 'MAE', 'RMSE', 'MAPE_%', 'MSE', 'R2', 'MASE', 'RMSSE', 'Sesgo']
            labels = {'Grupo': 'Etapa', 'N': 'Observaciones', 'MAE': 'MAE (unidades)',
                      'RMSE': 'RMSE (unidades)', 'MAPE_%': 'MAPE (%)',
                      'MSE': 'MSE (unidades²)', 'R2': 'R²', 'Horizonte': 'Horizonte',
                      'MASE': 'MASE', 'RMSSE': 'RMSSE', 'Sesgo': 'Sesgo (unidades)'}
            stages = {'TRAIN': 'Entrenamiento', 'VALIDATION': 'Validación', 'TEST': 'Prueba final'}
            scores = result['split_metrics'].loc[
                result['split_metrics'].Modelo.eq(thesis_model), columns].copy()
            scores['Grupo'] = scores['Grupo'].replace(stages)
            formats = {labels[key]: st.column_config.NumberColumn(format='%.3f' if key == 'R2' else '%.2f')
                       for key in ('MAE', 'RMSE', 'MAPE_%', 'MSE', 'R2', 'MASE', 'RMSSE', 'Sesgo')}
            st.dataframe(scores.rename(columns=labels), hide_index=True, width='stretch', column_config=formats)
            st.caption('Evaluación de un paso: cada predicción utiliza el historial anterior a su origen. '
                       'Entrenamiento mide ajuste; validación se utiliza para seleccionar el modelo.')
            if not result['evaluated']:
                st.info('La prueba final aparecerá después de pulsar «Evaluar TEST final del taller».')
            st.subheader('Métricas por horizonte')
            by_horizon = result['horizon_metrics'].loc[
                result['horizon_metrics'].Modelo.eq(thesis_model), ['Grupo', 'Horizonte', *columns[1:]]].copy()
            by_horizon['Grupo'] = by_horizon['Grupo'].replace(stages)
            st.dataframe(by_horizon.rename(columns={**labels, 'N': 'Pares evaluados'}),
                         hide_index=True, width='stretch', column_config=formats)
            st.caption('Horizonte 1 es el siguiente período; horizonte 8 está ocho períodos después del origen. '
                       'Las trayectorias se solapan y sus pares no son observaciones independientes.')
        else:
            st.info('Entrena y valida el taller para ver las métricas de la LSTM.')
        with st.expander('Cómo interpretar las métricas'):
            st.markdown(
                '- **MAE:** promedio del error absoluto, en unidades de demanda. Menor es mejor.\n'
                '- **RMSE:** raíz del promedio de errores cuadrados; da más peso a errores grandes. Menor es mejor.\n'
                '- **MAPE:** promedio del error absoluto relativo, multiplicado por 100. '
                'El cálculo excluye demandas con valor absoluto ≤ 0,00000001; si todas lo son, queda vacío.\n'
                '- **MSE:** promedio de errores cuadrados, en unidades²; equivale a RMSE² antes del redondeo.\n'
                '- **R²:** 1 − suma de errores cuadrados / suma de desviaciones cuadradas respecto a la media real. '
                'Puede ser negativo; si la demanda real es constante, queda vacío. No es un porcentaje de precisión.\n'
                '- **MASE / RMSSE:** error escalado por las diferencias estacionales de TRAIN. '
                'Menos de 1 indica menor error que esa escala histórica; no sustituye comparar con Naive estacional en el mismo bloque. '
                'Quedan vacíos si no existe una escala válida.\n'
                '- **Sesgo:** promedio de predicción menos demanda real; negativo indica subestimación.'
            )
            st.caption('Estas métricas comparan predicciones con demanda observada. '
                       'No garantizan exactitud futura; para planificar revisa también los horizontes que utilizarás.')
    with data_tab:
        st.dataframe(readable_table(data), hide_index=True, width='stretch')
        st.dataframe(feature_audit(data, int(len(data)*.7)), hide_index=True, width='stretch')
        st.plotly_chart(px.scatter(data, x='periodo', y='demanda', color='cliente', title='Demanda registrada por cliente'), width='stretch')
        for note in prepared.notes:
            st.caption(note)
    with metrics_tab:
        if result:
            st.info(f"Modelo principal de la tesis: {result.get('thesis_selected', result['lstm_name'])}. "
                    f"Ganador de la comparación en validación: {result['selected']}.")
            st.dataframe(readable_table(result['split_metrics']), hide_index=True, width='stretch')
            st.caption('Esta tabla mide un paso. TRAIN mide ajuste; VALIDATION participa en selección. Las comparaciones usan los mismos objetivos y máscaras de procedencia.')
            st.caption(result.get('validation_protocol', 'Consulte el protocolo de la ejecución guardada.'))
            st.subheader('Pronóstico directo por horizonte')
            st.dataframe(readable_table(result['horizon_metrics']), hide_index=True, width='stretch')
            st.caption('Cada trayectoria usa solo historia anterior a su origen; los orígenes se solapan. N es número de pares, no número de experimentos independientes.')
            with st.expander('Evaluación del cliente'):
                st.caption(f"Modelo de cliente: {result['client_model']['selected']}.")
                st.dataframe(result['client_model']['metrics'], hide_index=True, width='stretch')
                st.caption(PLANNING_NOTE)
                st.caption('Exactitud y recall se expresan de 0 a 1. Los pares de horizontes se solapan; no son observaciones independientes.')
            with st.expander('Búsqueda y cortes'):
                st.dataframe(result['search'], hide_index=True, width='stretch')
                st.dataframe(result['validation_folds'], hide_index=True, width='stretch')
                if 'seed_metrics' in result:
                    st.dataframe(result['seed_metrics'], hide_index=True, width='stretch')
        else:
            st.info('Entrena para comparar las configuraciones antes de evaluar TEST.')
    with forecast_tab:
        if result and result['evaluated']:
            thesis_model = result.get('thesis_selected', result['lstm_name'])
            test = result['split_metrics'].query("Grupo == 'TEST'").set_index('Modelo').loc[thesis_model]
            mae_col, rmse_col = st.columns(2)
            mae_col.metric('MAE de TEST — LSTM, unidades', f'{test.MAE:.2f}')
            rmse_col.metric('RMSE de TEST — LSTM, unidades', f'{test.RMSE:.2f}')
            st.caption('Errores de prueba final a un paso: menor es mejor. '
                       'Consulta R², MAPE, MSE y los errores por horizonte en «Métricas».')
            if not np.isfinite(test.R2) or test.R2 <= 0:
                st.warning('La precisión predictiva no está demostrada por un R² positivo en esta prueba. Revisa también el error de varios pasos.')
            labels = prepared.future_labels(config.horizon, config.periodicity)
            try:
                if ('forecast' not in result or result['forecast'].Periodo.tolist() != labels or
                        not result['forecast'].Modelo_principal_tesis.eq(thesis_model).all()):
                    forecast_workshop(result, labels, model_name=thesis_model)
                result['planning_forecast'] = planning_forecast(result['forecast'], data, thesis_model, result['client_model'])
            except (ValueError, IndexError) as exc:
                st.error(str(exc))
            if 'forecast' in result:
                st.dataframe(result['planning_forecast'], hide_index=True, width='stretch',
                             column_config={'Periodo': 'Período', 'Pronostico': 'Pronóstico', 'Cliente': 'Cliente'})
                names = list(result['_artifacts']['production'])
                st.plotly_chart(px.line(result['forecast'], x='Periodo', y=names, markers=True, title='Pronóstico posterior al último dato'), width='stretch')
                st.caption('El futuro usa un modelo reajustado con todo el histórico. Las métricas corresponden al modelo de evaluación anterior. No hay garantía de precisión del reajuste.')
            st.download_button('Descargar informe del taller', export_workshop(result), 'pronostico_taller.xlsx',
                               'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='workshop_download')
        else:
            st.info('El pronóstico se genera después de fijar la configuración y evaluar TEST.')
