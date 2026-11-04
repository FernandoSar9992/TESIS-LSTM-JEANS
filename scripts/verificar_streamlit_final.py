"""Recorrido real con Excel y TensorFlow, sin simular modelos."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from io import BytesIO
from unittest.mock import patch
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from streamlit.testing.v1 import AppTest

def check(app, stage):
    assert not app.exception, [(e.message) for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    print('OK',stage,flush=True)

upload=BytesIO(Path(sys.argv[1]).read_bytes());upload.name='04_BASE_DE_DATOS.xlsx'
with patch('streamlit.file_uploader', return_value=upload):
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=600).run()
    check(app,'carga real')
    next(c for c in app.checkbox if c.label.startswith('Confirmo')).check().run()
    next(b for b in app.button if b.label=='Entrenar modelos').click().run()
    check(app,'entrenamiento real')
    protocols=app.session_state['protocols']
    assert all(p['candidates'][p['selected']].model is not None for p in protocols.values())
    for n,p in protocols.items():
        if n.startswith('Demanda'):
            assert p['candidates'][p['selected']].model is app.session_state['fits'][n].model
    next(b for b in app.button if b.label=='Evaluar PRUEBA').click().run()
    check(app,'prueba')
    frozen_tests={n:p['test'].copy(deep=True) for n,p in protocols.items()}
    next(b for b in app.button if b.label=='Generar pronóstico').click().run()
    check(app,'pronostico y exportacion Excel')
    import pandas as pd
    for n,p in app.session_state['protocols'].items():
        pd.testing.assert_frame_equal(frozen_tests[n],p['test'])
    if len(sys.argv)>2:
        import json
        from pronostico_horizonte import forecast_lstm_verified
        from datetime import date
        out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
        forecast=app.session_state['forecast']
        forecast.to_csv(out/'pronostico_12_semanas.csv',index=False)
        for n,p in protocols.items():
            fit=app.session_state['fits'][n]
            if n.startswith('Demanda'):
                expected=forecast_lstm_verified(fit,date(2026,11,2),12)
                import numpy as np
                np.testing.assert_allclose(forecast[fit.target],expected[fit.target],rtol=1e-7)
            p['test'].to_csv(out/(fit.target+'_prueba.csv'),index=False)
            p['validation'].to_csv(out/(fit.target+'_validacion.csv'),index=False)
        (out/'seleccion.json').write_text(json.dumps({n:{'modelo':p['selected'],
            'epocas':p['candidates'][p['selected']].epochs} for n,p in protocols.items()},indent=2),encoding='utf-8')
    next(b for b in app.button if b.label=='Calcular plan preliminar').click().run()
    check(app,'plan y exportacion Excel')
    assert len(app.session_state['forecast'])==12
    assert len(app.session_state['plan_preliminar'])==12
    print('STREAMLIT COMPLETO',flush=True)
