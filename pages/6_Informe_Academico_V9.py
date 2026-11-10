"""Informe visual para revisión académica: datos agregados históricos auditados."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_tesis import install_theme, hero_audit, kpis, note, ribbon


st.set_page_config(page_title="TESIS | Informe académico V9", page_icon="📊", layout="wide")
install_theme()
st.markdown("""
<div class="tesis-lead">TESIS · Informe técnico de validación</div>
""",unsafe_allow_html=True)
st.markdown("""
<section class="tesis-hero">
<div class="tesis-overline">Modelo de pronóstico de demanda · Revisión del tutor</div>
<h1>Diagnóstico de generalización de la LSTM V9</h1>
<p>Evidencia de validación cruzada temporal para dos modalidades productivas:
maquila y corte propio. Un resumen transparente de resultados,
alcances y limitaciones para sustentar el estudio.</p>
<div class="tesis-pills">
<span class="tesis-pill">Ventana expansiva</span>
<span class="tesis-pill">Horizonte fijo de 12 semanas</span>
<span class="tesis-pill">TEST fuera de la auditoría</span>
<span class="tesis-pill">Resultados reproducibles desde Excel</span>
</div>
</section>
""",unsafe_allow_html=True)

data_file = Path(__file__).resolve().parent.parent / "results" / "cv_v9_resumen_tutor.json"
if not data_file.exists():
    st.error("No se encontró el resumen verificado del estudio.")
    st.stop()
report = json.loads(data_file.read_text(encoding="utf-8"))
mq = report["modalidades"]["MQ"]
cp = report["modalidades"]["CP"]

st.caption(
    "Fuente: exportaciones CV_V9_MQ.xlsx y CV_V9_CP.xlsx del investigador "
    "· Corte del análisis: 10 de noviembre de 2026. "
    "Las cifras son un registro del experimento, NO métricas calculadas en directo "
    "con cualquier archivo que se cargue posteriormente."
)

kpis([
    ("Modalidades estudiadas", "2", "Maquila y corte propio"),
    ("Folds externos", "5", "3 MQ + 2 CP"),
    ("Predicciones", "60", "36 MQ + 24 CP; horizontes de 12 semanas"),
])

tab_results, tab_method, tab_caveats, tab_data = st.tabs([
    "01 · Resultados", "02 · Metodología",
    "03 · Sobreajuste y límites", "04 · Evidencia"
])

with tab_results:
    st.markdown('<div class="tesis-lead">Desempeño por modalidad</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.subheader("Maquila")
        kpis([
            ("MAE medio", f"{mq['MAE_medio']:.2f}", "Pantalones"),
            ("RMSE medio", f"{mq['RMSE_medio']:.2f}", "Pantalones"),
            ("Mejora frente a referencias", f"{mq['supera_ambas']}/{mq['n_folds']}", "Folds con menor RMSE"),
        ])
        note(
            "Conclusión · Maquila",
            "Se observan variaciones relevantes de desempeño entre los tres "
            "cortes temporales. La LSTM no supera sistemáticamente a ambos "
            "modelos estacionales. El sesgo de predicción en esos cortes es positivo.",
            warm=True,
        )
    with b:
        st.subheader("Corte propio")
        kpis([
            ("MAE medio", f"{cp['MAE_medio']:.2f}", "Pantalones"),
            ("RMSE medio", f"{cp['RMSE_medio']:.2f}", "Pantalones"),
            ("Mejora frente a referencias", f"{cp['supera_ambas']}/{cp['n_folds']}", "Folds con menor RMSE"),
        ])
        note(
            "Conclusión · Corte propio",
            "La LSTM presenta menor RMSE que ambas referencias en los dos "
            "cortes evaluados. La evidencia sigue siendo limitada por el número "
            "de folds y la procedencia de las observaciones."
        )

    selector = st.radio(
        "Explorar por modalidad", ["Maquila", "Corte propio"],
        horizontal=True, key="academic_cv_series"
    )
    view = mq if selector=="Maquila" else cp
    rows = view["folds"]
    frame = pd.DataFrame([
        {
            "Fold": row["fold"],
            "Intervalo": f"{row['desde']} / {row['hasta']}",
            "MAE LSTM": row["mae"],
            "RMSE LSTM": row["rmse"],
            "R²": row["r2"],
            "Sesgo": row["sesgo"],
            "RMSE Naive 48": row["naive48"],
            "RMSE Naive 52": row["naive52"],
        }
        for row in rows
    ])
    st.markdown("#### Comparación del RMSE por fold")
    plot=go.Figure()
    x=[f"Fold {r['fold']}" for r in rows]
    for name,key,color in [
        ("LSTM V9","rmse","#007D85"),
        ("Naive estacional 48","naive48","#A9BEC9"),
        ("Naive estacional 52","naive52","#D8A45A"),
    ]:
        plot.add_trace(go.Bar(
            name=name, x=x, y=[r[key] for r in rows],
            marker_color=color, marker_line_width=0,
            hovertemplate="%{x}<br>RMSE: %{y:.2f} pantalones<extra>"+name+"</extra>"
        ))
    plot.update_layout(
        barmode="group", height=365, plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=12,r=12,t=14,b=10),
        legend=dict(orientation="h",y=-.22,x=0),
        font=dict(family="Arial",color="#2F465C"),
        yaxis=dict(title="RMSE (pantalones)",showgrid=True,gridcolor="#ECF1F4",zeroline=False),
        xaxis=dict(title=None,showgrid=False)
    )
    st.plotly_chart(plot,use_container_width=True)
    st.dataframe(
        frame.style.format({
            "MAE LSTM":"{:.2f}","RMSE LSTM":"{:.2f}",
            "R²":"{:.3f}","Sesgo":"{:+.2f}",
            "RMSE Naive 48":"{:.2f}","RMSE Naive 52":"{:.2f}"
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Comparar MAE en valores absolutos entre MQ y CP no demuestra que una "
        "serie sea más predecible: tienen escalas de demanda diferentes."
    )

with tab_method:
    st.markdown("### Diseño de validación cruzada temporal")
    st.write(
        "Se aplicó una estrategia de **ventana expansiva** (*expanding-window*). "
        "Cada fold utiliza exclusivamente datos disponibles hasta su origen. "
        "La evaluación externa consiste en 12 semanas futuras consecutivas, "
        "sin incorporar las demandas reales intermedias durante el pronóstico."
    )
    st.markdown("""
| Elemento metodológico | Aplicación en la auditoría |
|---|---|
| Modalidades | Maquila y corte propio, evaluadas por separado |
| Horizontes | 12 semanas fijas por fold |
| Orígenes | 3 en maquila y 2 en corte propio |
| Estructura | Entrenamiento anidado, validación interna y evaluación externa |
| Ajuste | Tres semillas por arquitectura, Early Stopping, Dropout y L2 |
| Selección de arquitectura | RMSE y MAE de la validación interna |
| Benchmark | Naive estacional de 48 y 52 semanas |
| TEST | Reservado; la auditoría no lo consulta |
| Comparabilidad | Objetivos externos completos y disjuntos en cada modalidad |
""")
    note(
        "Importante sobre el procedimiento",
        "Los modelos de cada fold se vuelven a entrenar de manera independiente. "
        "La arquitectura que se selecciona dentro de un fold puede diferir de "
        "la arquitectura operativa actual de V9. Esta auditoría evalúa el "
        "procedimiento de aprendizaje y su generalización, sin modificar "
        "el pronóstico de producción."
    )
    st.markdown("### Relación con la planificación")
    st.write(
        "Los pronósticos de maquila y corte propio alimentan el **Plan Agregado "
        "de Producción (PAP)** y el **Programa Maestro de Producción (PMP)**. "
        "La validación cruzada caracteriza el riesgo de error antes de tomar "
        "decisiones de capacidad, horas extra e inventario."
    )

with tab_caveats:
    st.markdown("### Interpretación del sobreajuste")
    note(
        "Diagnóstico responsable",
        "Las curvas de algunas ejecuciones pueden reflejar divergencia entre "
        "las pérdidas de entrenamiento y validación, patrón compatible con "
        "sobreajuste. Early Stopping reduce ese riesgo, pero las métricas "
        "entre folds no permiten demostrar ausencia de sobreajuste.",
        warm=True,
    )
    st.markdown("#### Señales observadas")
    st.write(
        "**Maquila:** el modelo supera a las dos referencias solamente en uno "
        "de tres folds; dos folds presentan R² negativo. Se observa sesgo "
        "positivo en los tres períodos evaluados."
    )
    st.write(
        "**Corte propio:** el modelo alcanza menor RMSE que ambas referencias "
        "en dos folds, con R² positivo; no obstante, solo hay dos cortes "
        "externos y la dirección del sesgo cambia entre ellos."
    )
    st.markdown("#### Advertencias obligatorias")
    st.write(
        "Los libros exportados no identifican para cada predicción si la "
        "demanda observada proviene de un registro totalmente real, mixto "
        "o imputado. Por ello, las métricas no constituyen por sí solas "
        "validación independiente sobre ventas registradas."
    )
    st.write(
        "Aunque **esta auditoría** no utiliza TEST, el bloque TEST de la "
        "aplicación original ya había sido consultado en iteraciones anteriores. "
        "No se debe presentar como una evaluación ciega o completamente nueva."
    )
    st.write(
        "La comparación de resultados es retrospectiva. "
        "No se realizaron nuevos entrenamientos en esta página: sus valores "
        "proceden exclusivamente de los Excel de auditoría entregados."
    )
    st.info(
        "Para ejecutar nuevamente la auditoría con una base actualizada, "
        "abre «Validacion Cruzada V9» en el menú lateral de Streamlit."
    )

with tab_data:
    st.markdown("### Trazabilidad de los resultados")
    st.write(
        "Las tablas se transcribieron de los libros de auditoría "
        "**CV_V9_MQ.xlsx** y **CV_V9_CP.xlsx**, generados desde V9. "
        "Se conserva el horizonte de 12 semanas y el valor real de los "
        "indicadores, sin cambiar los errores para mejorar el desempeño."
    )
    st.dataframe(
        pd.DataFrame([
            {"Modalidad":v["nombre"],"Folds":v["n_folds"],
             "Semanas":v["semanas"],"MAE medio":v["MAE_medio"],
             "RMSE medio":v["RMSE_medio"],
             "Dispersión RMSE":v["desviacion_RMSE"],
             "LSTM mejor que ambas referencias":f"{v['supera_ambas']}/{v['n_folds']}"}
            for v in (mq,cp)
        ]),
        hide_index=True, use_container_width=True
    )
    st.markdown("#### Descargas")
    export_frame=pd.concat([
        pd.DataFrame([{"Modalidad":v["nombre"],"Fold":r["fold"],
            "Desde":r["desde"],"Hasta":r["hasta"],
            "MAE_LSTM":r["mae"],"RMSE_LSTM":r["rmse"],
            "R2_LSTM":r["r2"],"Sesgo":r["sesgo"],
            "RMSE_Naive48":r["naive48"],"RMSE_Naive52":r["naive52"]}
            for r in v["folds"]])
        for v in (mq,cp)
    ],ignore_index=True)
    st.download_button(
        "Descargar métricas de los cinco folds (CSV)",
        data=export_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="RESULTADOS_CV_V9_PARA_TUTOR.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption(
        "Para consultar curvas de aprendizaje y predicciones individuales, "
        "utilizar las exportaciones completas de cada modalidad en "
        "Validación Cruzada V9."
    )

st.divider()
st.caption(
    "TESIS · LSTM V9 | Informe de resultados para revisión académica | "
    "La auditoría es complementaria y no altera el motor original."
)
