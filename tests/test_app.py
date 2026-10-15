"""Interfaz con motor real y red simulada; entrenamiento real en validar_tesis.py."""
from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_thesis import simulated_network, make_series


def upload_csv(offset=0, provenance=False):
    data = make_series()
    data['demanda'] += offset
    if provenance:
        data['Demanda_Final'] = data.demanda + 2
        data['Origen'] = ['IMPUTADO']*142+['REAL']*34
    result = BytesIO(data.to_csv(index=False).encode())
    result.name = 'datos.csv'
    return result


class AppTests(unittest.TestCase):
    def test_empty_start(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, 'Pronóstico de la demanda')
        self.assertEqual(app.number_input(key='horizon').value, 8)

    def test_periodicity_changes_invalidate_trained_model(self):
        with patch('streamlit.file_uploader', return_value=upload_csv()), simulated_network():
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            app.button(key='run').click().run()
            self.assertIsNotNone(app.session_state['result'])
            app.number_input(key='periodicity').set_value(52).run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state['result'])

    def test_real_declaration_is_explicit_and_resets_with_new_content(self):
        upload = upload_csv()
        with patch('streamlit.file_uploader', return_value=upload), simulated_network():
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            app.checkbox(key='declared_real').check().run()
            app.button(key='run').click().run()
            self.assertTrue(app.session_state['result']['processed'].Origen.eq('REAL').all())
            self.assertTrue(app.session_state['result']['preparation']['declaracion_todos_reales'])
            upload.seek(0)
            upload.truncate()
            upload.write(upload_csv(5).getvalue())
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.checkbox(key='declared_real').value)
            self.assertIsNone(app.session_state['result'])

    def test_two_stages_export_no_repeat_and_invalidation(self):
        with patch('streamlit.file_uploader', return_value=upload_csv()), simulated_network():
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual([t.label for t in app.tabs], ['Pronóstico LSTM', 'Evaluación LSTM', 'Comparación', 'Diagnóstico del modelo', 'Datos'])
            app.button(key='run').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(app.session_state['result']['evaluated'])
            self.assertEqual(len(app.get('download_button')), 0)
            app.button(key='final_test').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(app.session_state['result']['evaluated'])
            self.assertEqual(len(app.get('download_button')), 1)
            previous_metrics = app.session_state['result']['metrics'].copy()
            app.number_input(key='horizon').set_value(4).run()
            self.assertTrue(app.session_state['result']['evaluated'])
            self.assertEqual(len(app.session_state['result']['forecast']), 4)
            self.assertTrue(app.session_state['result']['metrics'].equals(previous_metrics))
            app.number_input(key='epochs').set_value(200).run()
            self.assertIsNone(app.session_state['result'])
            self.assertEqual(len(app.get('download_button')), 0)
            app.button(key='run').click().run()
            self.assertTrue(app.button(key='final_test').disabled)
            app.checkbox(key='automatic').uncheck().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(w.label == 'Ventana histórica' for w in app.number_input))

    def test_same_filename_new_data_invalidates_result(self):
        upload = upload_csv()
        with patch('streamlit.file_uploader', return_value=upload), simulated_network():
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            app.button(key='run').click().run()
            self.assertFalse(app.exception)
            self.assertIsNotNone(app.session_state['result'])
            upload.seek(0)
            upload.truncate()
            upload.write(upload_csv(100).getvalue())
            app.run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state['result'])
            self.assertEqual(app.dataframe[0].value.Demanda.iloc[0], 200)

    def test_unknown_provenance_is_compact_and_never_relabelled_real(self):
        with patch('streamlit.file_uploader', return_value=upload_csv()), simulated_network():
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value.columns.tolist(), ['Período', 'Demanda'])
            self.assertFalse(any('SIN_DECLARAR' in str(frame.value) for frame in app.dataframe))
            # AppTest clasifica un expander con icono como Status en Streamlit 1.64.
            sections = list(app.expander) + list(app.status)
            self.assertTrue(any('Procedencia y preparación' in section.label for section in sections))
            self.assertFalse(any('sin declarar' in metric.label.lower() for metric in app.metric))
            app.button(key='run').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['result']['processed'].Origen.eq('SIN_DECLARAR').all())

    def test_detect_final_column_provenance_and_warning(self):
        with patch('streamlit.file_uploader', return_value=upload_csv(provenance=True)), simulated_network():
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='demand_col').value, 'Demanda_Final')
            self.assertTrue(any('50 %' in w.value for w in app.warning))
            app.button(key='run').click().run()
            app.button(key='final_test').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(set(app.session_state['result']['metrics'].Alcance), {'SOLO_REAL'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
