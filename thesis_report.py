from io import BytesIO
import json
import pandas as pd


def export_thesis(result):
    if not result.get("evaluated"):
        raise ValueError("Evalúe TEST antes de exportar el informe final.")
    config = {**result["config"], **result["selected"], **result["training_summary"],
              "Lookback": result["selected"]["lookback"], "Neuronas": result["selected"]["units"],
              "Dropout": result["selected"]["dropout"], "Batch_size": result["config"]["batch_size"],
              "Epochs_ejecutadas": result["training_summary"]["Epocas_ejecutadas"],
              "Limite_epocas": result["config"]["epochs"], "Learning_rate": result["config"]["learning_rate"],
              "Configuracion_solicitada": json.dumps(result["config"], ensure_ascii=False),
              "Train_size": result["train_end"], "Validation_size": result["val_end"]-result["train_end"],
              "Test_size": result["test_origins"], "Ridge_alpha": result["ridge_alpha"],
              "Datos_SHA256": result["data_hash"], "Motor": result["engine_version"],
              "Scaler": json.dumps(result["scaler"]), "Preparacion": json.dumps(result.get("preparation", {}), ensure_ascii=False)}
    config.update(Referencia_validacion=result.get('reference_model', ''),
                  Scaler_produccion=json.dumps(result.get('production_scaler', {})),
                  Reajuste_produccion=json.dumps(result.get('production_training', {}), ensure_ascii=False),
                  Horizonte_backtest=result.get('backtest_horizon', 1))
    frames = {"Datos_procesados": result["processed"], "Predicciones": result["detail"], "Metricas": result["metrics"],
              "Configuracion": pd.DataFrame([config]), "Diagnostico": result["diagnostic"],
              "Metricas_solo_REAL": result["metrics_real"], "Metricas_descriptivas": result["metrics_all"],
              "Metricas_por_grupo": result["split_metrics"], "Particion_temporal": result["split_table"],
              "Perfil_demanda": result["data_profile"], "Busqueda_LSTM": result["search"], "Cortes_validacion": result["validation_folds"],
              "Predicciones_baselines": result["baseline_predictions"], "Pronostico_futuro": result["forecast"],
              "Loss": pd.DataFrame({"loss": result["history"]["loss"], "val_loss": result["history"]["val_loss"]}),
              "Metodologia": pd.DataFrame({"Nota": [
                  "TRAIN, VALIDATION y TEST cronológicos. Scaler y pesos de evaluación ajustados solo en TRAIN. El futuro puede usar un reajuste separado con todo el histórico posterior a TEST.",
                  "En v5: configuración seleccionada por RMSE de validación temporal; MAE desempata. TEST no selecciona hiperparámetros. VALIDATION no es una prueba independiente porque se usó para elegir épocas y configuración.",
                  "TEST mide un paso con observaciones anteriores conocidas; el futuro es recursivo y no entra en R².",
                  "SOLO_REAL excluye objetivos imputados. SIN_TRAZABILIDAD no certifica que los objetivos sean observados.",
                  "TODOS_DESCRIPTIVO incluye reconstrucciones y no constituye validación contra observaciones reales.",
                  "Un objetivo REAL puede usar entradas imputadas; compruebe que la reconstrucción original sea causal.",
                  "SOLO_REAL requiere al menos dos objetivos REAL; muestras reales pequeñas siguen siendo evidencia débil. La hoja solo_REAL siempre informa su N.",
                  "TRAIN se compara desde max(12, lookback), común a los modelos. La red se entrena desde su propio lookback.",
                  "La representación local resta la media de hasta 8 valores pasados y divide por la desviación de la ventana; el peso factor² conserva el MSE en la escala global de TRAIN.",
                  "Las predicciones negativas se limitan a cero para demanda no negativa. No se alteran los objetivos ni los R² negativos.",
                  "En v5: las variantes estacionales incorporan calendario y rezago causal del ciclo configurado. Compiten con redes sin esas entradas; no se fuerza estacionalidad.",
                  "El backtest recursivo mide trayectorias sin observaciones intermedias. Sus orígenes se solapan y no son muestras independientes. Horizonte_backtest indica el máximo evaluado.",
                  "Referencia_validacion se selecciona antes de TEST entre los modelos base. Su pronóstico se distingue de la LSTM.",
                  "R² negativo se conserva. Los diagnósticos son indicios por reglas, no pruebas causales ni garantías.",
                  "Después de consultar TEST no ajuste hiperparámetros con él; reserve datos nuevos para una nueva evaluación independiente."]})}
    for key, sheet in [('horizon_metrics', 'Error_por_horizonte'), ('recursive_detail', 'Trayectorias_recursivas'), ('validation_detail', 'Predicciones_validacion')]:
        if key in result:
            frames[sheet] = result[key]
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()
