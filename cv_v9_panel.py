"""Pestaña de auditoría V9: no modifica el entrenamiento ni el pronóstico activos."""
from __future__ import annotations
from io import BytesIO
import hashlib
import pandas as pd
import streamlit as st
from cv_v9_diagnostico import AuditConfig, expanding_folds, evaluate_cross_validation, HORIZON
from ui_tesis import kpis


def _signature(digest, settings, confirmed, name, nfolds):
    fields = (digest, confirmed, name, nfolds, settings.train_ratio,
              settings.val_ratio, settings.lookback, settings.units,
              settings.seed, settings.epochs, settings.patience,
              settings.batch_size, settings.dropout)
    return hashlib.sha256(repr(fields).encode()).hexdigest()


def render_cv_tab(series, settings, source_digest, confirm_zeros, forecast_weeks):
    st.subheader("Validación cruzada temporal · LSTM V9")
    st.write("Evaluación complementaria de generalización y sobreajuste. "
             "Se entrenan copias independientes de la arquitectura V9, "
             "sin alterar el modelo de pronóstico ni su salida.")
    st.info(
        f"El pronóstico operativo está configurado para **{int(forecast_weeks)} "
        f"semana(s)**. La auditoría usa un horizonte de **{HORIZON} semanas** "
        "por fold porque evalúa el diseño multihorizonte fijo V9. "
        "Esto no impide pronosticar de 1 a 26 semanas."
    )
    available = [n for n in ("Demanda de maquila", "Demanda de corte propio")
                 if n in series]
    if not available:
        st.error("No están disponibles las series de demanda.")
        return
    left, right = st.columns([3, 1])
    with left:
        name = st.selectbox("Modalidad a evaluar", available,
                            key="cv_integrada_modalidad")
    with right:
        nfolds = st.number_input("Máximo de folds", min_value=1, max_value=5,
                                 value=3, key="cv_integrada_folds")
    config = AuditConfig(
        train_ratio=float(settings.train_ratio),
        val_ratio=float(settings.val_ratio),
        lookback=int(settings.lookback),
        units=int(settings.units),
        dropout=float(settings.dropout),
        epochs=int(settings.epochs),
        patience=int(settings.patience),
        batch_size=int(settings.batch_size),
        seed=int(settings.seed),
        max_folds=int(nfolds),
    )
    data = series[name]
    try:
        folds = expanding_folds(
            len(data), train_ratio=config.train_ratio,
            val_ratio=config.val_ratio, horizon=HORIZON,
            max_folds=config.max_folds
        )
    except ValueError as exc:
        st.warning(f"No se puede evaluar esta serie: {exc}")
        return

    st.caption(
        f"{name}: {len(data)} semanas registradas, {len(folds)} fold(s) "
        f"de {HORIZON} semanas; TRAIN {config.train_ratio:.0%}, "
        f"VALIDACIÓN {config.val_ratio:.0%}, TEST reservado."
    )
    if len(folds) < 2:
        st.warning("Con un solo fold no se puede comprobar estabilidad entre períodos.")
    with st.expander("Metodología de la validación cruzada"):
        st.write(
            "Ventana expansiva, selección anidada de arquitectura y épocas, "
            "tres semillas, Dropout, L2 y Early Stopping. Las ventanas de "
            "evaluación no se superponen; TEST no participa en la auditoría. "
            "La LSTM se compara con Naive estacional 48 y 52. "
            "El diagnóstico emplea el mismo archivo y la misma configuración "
            "temporal de la aplicación."
        )
    key = _signature(source_digest, settings, confirm_zeros, name, nfolds)
    cache = st.session_state.setdefault("cv_integrada_resultados", {})
    if st.button("Ejecutar validación cruzada V9", type="primary",
                 key="cv_integrada_ejecutar", use_container_width=True):
        status = st.empty()
        try:
            with st.spinner("Entrenando los folds de validación cruzada…"):
                result = evaluate_cross_validation(
                    data, config, progress=status.info
                )
            if len(cache) > 2 and key not in cache:
                cache.pop(next(iter(cache)))
            cache[key] = result
            st.success("Validación terminada. El pronóstico activo no cambió.")
        except Exception as exc:
            st.error(f"No se completó: {type(exc).__name__}: {exc}")
        finally:
            status.empty()
    if key not in cache:
        st.caption("Pulsa el botón para evaluar esta serie y descargar sus resultados.")
        return

    result = cache[key]
    info = result["summary"].iloc[0]
    kpis([
        ("Folds", int(info["Folds"]), "Ventana expansiva"),
        ("RMSE medio", f"{info['Promedio_RMSE_LSTM']:.2f}", "Pantalones por semana"),
        ("MAE medio", f"{info['Promedio_MAE_LSTM']:.2f}", "Pantalones por semana"),
        ("Dispersión RMSE", f"{info['Desviacion_RMSE_folds']:.2f}",
         "Diferencias entre folds"),
        ("Folds favorables", f"{int(info['Folds_superados_por_LSTM'])}/{int(info['Folds'])}",
         "Menor RMSE que referencias"),
        ("TEST", "Reservado", "No participa"),
    ])

    st.markdown("#### Rendimiento entre particiones")
    frame = result["folds"]
    cols = ["Fold", "Validación_desde", "Validación_hasta",
            "MAE_LSTM", "RMSE_LSTM", "R2_LSTM", "Sesgo_LSTM",
            "RMSE_Naive48", "RMSE_Naive52"]
    st.dataframe(frame[cols].round(3), hide_index=True, use_container_width=True)
    chart = frame[["Fold", "RMSE_LSTM", "RMSE_Naive48", "RMSE_Naive52"]].set_index("Fold")
    chart = chart.rename(columns={"RMSE_LSTM": "LSTM V9",
                                  "RMSE_Naive48": "Naive 48",
                                  "RMSE_Naive52": "Naive 52"})
    st.bar_chart(chart, use_container_width=True)
    with st.expander("Curvas de aprendizaje · loss y val_loss", expanded=True):
        curves = result["curves"]
        a, b, c = st.columns(3)
        with a:
            fold = st.selectbox("Fold", sorted(curves["Fold"].unique()),
                                key="cv_integrada_curva_fold")
        architectures = curves.loc[curves.Fold == fold,
                                    ["Ventana", "Unidades"]].drop_duplicates()
        options = [(int(row.Ventana), int(row.Unidades))
                   for row in architectures.itertuples()]
        with b:
            architecture = st.selectbox(
                "Arquitectura", options,
                format_func=lambda x: f"Ventana {x[0]} · unidades {x[1]}",
                key="cv_integrada_curva_modelo")
        seeds = sorted(curves.loc[
            (curves.Fold == fold) &
            (curves.Ventana == architecture[0]) &
            (curves.Unidades == architecture[1]), "Semilla"].unique())
        with c:
            seed = st.selectbox("Semilla", seeds, key="cv_integrada_curva_semilla")
        selected = curves.loc[
            (curves.Fold == fold) &
            (curves.Ventana == architecture[0]) &
            (curves.Unidades == architecture[1]) &
            (curves.Semilla == seed)]
        if not selected.empty:
            st.line_chart(
                selected.set_index("Época")[["loss_train",
                                            "loss_validacion_interna"]],
                use_container_width=True)
            st.caption(f"Mejor época del piloto: "
                       f"{int(selected['Época_óptima'].iloc[0])}. "
                       "El gráfico es diagnóstico, no modifica el motor principal.")

    with st.expander("Predicciones y selección de arquitecturas"):
        st.dataframe(result["predictions"], hide_index=True, use_container_width=True)
        st.dataframe(result["candidates"], hide_index=True, use_container_width=True)

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet, field in (("Resumen", "summary"), ("Folds", "folds"),
                             ("Predicciones", "predictions"),
                             ("Candidatas", "candidates"),
                             ("Curvas", "curves")):
            result[field].to_excel(writer, sheet_name=sheet, index=False)
    st.download_button(
        "Descargar auditoría de validación cruzada (Excel)",
        data=out.getvalue(),
        file_name=f"CV_V9_{'MQ' if name == 'Demanda de maquila' else 'CP'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True, key="cv_integrada_descarga")
    st.warning(
        "Las métricas pueden incorporar demanda reconstruida. La validación "
        "cruzada no garantiza ausencia de sobreajuste ni sustituye una "
        "evaluación independiente sobre ventas reales nuevas."
    )
