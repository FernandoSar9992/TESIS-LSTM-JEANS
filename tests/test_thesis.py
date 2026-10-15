"""Invariantes metodológicos; las redes reales se ejecutan en validar_tesis.py."""
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand_data import prepare_series, detect_column
from lstm_engine import scaled_windows, inverse_prediction, future_input
from thesis_engine import (ThesisConfig, run_thesis, finalize_test, candidates,
                           temporal_folds, split_points, table_metrics, metric_values)
from thesis_report import export_thesis


class FakeModel:
    def fit(self, x, target, **kw):
        assert not kw['shuffle']
        return SimpleNamespace(history={'loss': [.2, .1], 'val_loss': [.1, .2]})

    def predict(self, x, **kw):
        return np.zeros((len(x), 1))

    def __call__(self, x, **kw):
        return self.predict(x)

    def count_params(self):
        return 10


@contextmanager
def simulated_network():
    fake_tf = SimpleNamespace(keras=SimpleNamespace(
        backend=SimpleNamespace(clear_session=lambda: None),
        utils=SimpleNamespace(set_random_seed=lambda seed: None),
        callbacks=SimpleNamespace(EarlyStopping=lambda **kw: kw, TerminateOnNaN=lambda: None,
                                  ReduceLROnPlateau=lambda **kw: kw)))
    with patch.dict(sys.modules, {'tensorflow': fake_tf}), \
         patch('thesis_engine.tensorflow_status', return_value=(True, 'simulado')), \
         patch('thesis_engine.build_network', side_effect=lambda *args: FakeModel()):
        yield


def make_series(n=176):
    return pd.DataFrame({'periodo': [f'p{i:03d}' for i in range(n)],
                         'demanda': 100+np.arange(n)*.3+10*np.sin(np.arange(n)/3)})


class TraceabilityTests(unittest.TestCase):
    def test_original_final_and_missing_original_preserved(self):
        frame = pd.DataFrame({'Etiqueta de Semana': ['a', 'b', 'c'], 'Demanda': [10., np.nan, 30.],
                              'Demanda_Final': [10., 20., 30.], 'Origen': ['REAL', 'IMPUTADO', 'REAL'],
                              'Tipo_estimacion': ['', 'interpolado', '']})
        self.assertEqual(detect_column(frame.columns, 'Demanda_Final', 'Demanda'), 'Demanda_Final')
        data = prepare_series(frame, 'Etiqueta de Semana', 'Demanda_Final').data
        self.assertTrue(np.isnan(data.demanda_original.iloc[1]))
        self.assertEqual(data.demanda.tolist(), [10, 20, 30])
        self.assertEqual(data.origen.tolist(), frame.Origen.tolist())
        self.assertEqual(data.tipo_estimacion.iloc[1], 'interpolado')

    def test_aggregation_keeps_mixed_and_missing_zero_provenance(self):
        frame = pd.DataFrame({'fecha': ['2025-01-01', '2025-01-01', '2025-01-03'],
                              'Demanda': [10, 5, 20], 'Origen': ['REAL', 'IMPUTADO', 'REAL']})
        data = prepare_series(frame, 'fecha', 'Demanda', date_mode=True, frequency='D', fill_missing_zero=True).data
        self.assertEqual(data.origen.tolist(), ['MIXTO', 'IMPUTADO', 'REAL'])
        self.assertEqual(data.demanda.tolist(), [15, 0, 20])
        self.assertIn('Cero declarado', data.tipo_estimacion.iloc[1])

    def test_imputed_errors_excluded_from_primary_metric(self):
        data = make_series(3)
        data['origen'] = ['REAL', 'IMPUTADO', 'REAL']
        pred = data.demanda.to_numpy().copy()
        pred[1] += 99999
        primary, all_rows, real_rows = table_metrics(data, np.arange(3), {'LSTM': pred}, 'TEST')
        self.assertEqual(primary.MAE.iloc[0], 0)
        self.assertEqual(primary.N.iloc[0], 2)
        self.assertGreater(all_rows.MAE.iloc[0], 10000)
        pd.testing.assert_frame_equal(primary, real_rows)
        data['origen'] = 'IMPUTADO'
        primary, _, real_rows = table_metrics(data, np.arange(3), {'LSTM': pred}, 'TEST')
        self.assertEqual(primary.Alcance.iloc[0], 'TODOS_DESCRIPTIVO')
        self.assertEqual(real_rows.N.iloc[0], 0)
        self.assertTrue(np.isnan(real_rows.RMSE.iloc[0]))


class MethodologyTests(unittest.TestCase):
    def test_inverse_alignment_and_frozen_scale_all_representations(self):
        y = make_series().demanda.to_numpy()
        for mode in ('nivel', 'cambio', 'local'):
            x, targets, origins, anchors, center, scale = scaled_windows(y, 123, 8, 1, mode)
            model = SimpleNamespace(predict=lambda x, **kw: targets)
            recovered = inverse_prediction(model, x, anchors, center, scale).ravel()
            np.testing.assert_allclose(recovered, y[origins], rtol=1e-6)
            changed = y.copy()
            changed[123:] += 100000
            other = scaled_windows(changed, 123, 8, 1, mode)
            self.assertEqual((center, scale), other[-2:])
            np.testing.assert_array_equal(x[origins < 123], other[0][origins < 123])
            fx, anchor = future_input(y[:150], 8, mode, center, scale)
            np.testing.assert_allclose(fx[0], x[origins == 150][0])
            np.testing.assert_allclose(anchor, anchors[origins == 150][0])

    def test_fold_targets_disjoint_and_before_test(self):
        tr, va = split_points(176)
        self.assertEqual((tr, va), (123, 149))
        folds = temporal_folds(tr, va, 32)
        self.assertEqual(folds, [(71, 97), (97, 123), (123, 149)])
        options = candidates(ThesisConfig(), folds[0][0])
        self.assertEqual(len(options), 33)
        self.assertEqual({c['representation'] for c in options}, {'local', 'nivel', 'cambio'})
        fixed = candidates(ThesisConfig(auto_representation=False, seasonal_features=False), folds[0][0])
        self.assertEqual(len(fixed), 9)
        self.assertEqual({c['representation'] for c in fixed}, {'local'})
        self.assertEqual({c['lookback'] for c in options}, {4, 8, 12})
        self.assertEqual({c['units'] for c in options}, {8, 16, 32})
        self.assertEqual({c['lookback'] for c in candidates(ThesisConfig(extended_windows=True), 71)}, {4, 8, 12, 16, 24})

    def test_test_cannot_select_weights_scale_configuration_or_epoch(self):
        original = make_series()
        changed = original.copy()
        changed.loc[149:, 'demanda'] += 100000
        changed['origen'] = 'SIN_DECLARAR'
        changed.loc[149:, 'origen'] = 'REAL'
        with simulated_network():
            first = run_thesis(original, ThesisConfig(), evaluate_test=False)
            second = run_thesis(changed, ThesisConfig(), evaluate_test=False)
        for key in ('search', 'validation_folds', 'split_metrics'):
            pd.testing.assert_frame_equal(first[key], second[key])
        self.assertEqual(first['selected'], second['selected'])
        self.assertEqual(first['selected']['epochs'], 1)
        self.assertEqual(first['scaler'], second['scaler'])
        self.assertNotIn('metrics', first)
        self.assertNotIn('forecast', first)
        self.assertNotIn('TEST', first['split_metrics'].Grupo.tolist())
        with self.assertRaises(ValueError):
            export_thesis(first)

    def test_final_once_same_test_baselines_and_recursive_future(self):
        data = make_series()
        with simulated_network():
            result = run_thesis(data, ThesisConfig(automatic=False, representation='cambio'), evaluate_test=False)
            result = finalize_test(result)
        origins = np.arange(149, 176)
        self.assertEqual(result['detail'].Periodo.tolist(), data.periodo.iloc[149:].tolist())
        np.testing.assert_array_equal(result['baseline_predictions'].Naive, data.demanda.to_numpy()[origins-1])
        np.testing.assert_allclose(result['baseline_predictions']['Media móvil 4'],
                                   [data.demanda.iloc[t-4:t].mean() for t in origins])
        self.assertEqual(set(result['metrics'].N), {27})
        # Fake zero change gives constant recursive future at last observed value.
        np.testing.assert_allclose(result['forecast'].LSTM, data.demanda.iloc[-1])
        self.assertEqual(len(result['forecast']), 8)
        with self.assertRaisesRegex(ValueError, 'ya fue evaluado'):
            finalize_test(result)
        with pd.ExcelFile(BytesIO(export_thesis(result))) as book:
            self.assertEqual(book.sheet_names[:5], ['Datos_procesados', 'Predicciones', 'Metricas', 'Configuracion', 'Diagnostico'])
            self.assertEqual(len(pd.read_excel(book, 'Predicciones')), 27)
            self.assertIn('Origen', pd.read_excel(book, 'Datos_procesados'))

    def test_horizon_not_part_of_test_or_selection(self):
        with simulated_network():
            short = run_thesis(make_series(), ThesisConfig(automatic=False, horizon=4))
            long = run_thesis(make_series(), ThesisConfig(automatic=False, horizon=24))
        pd.testing.assert_frame_equal(short['metrics'], long['metrics'])
        pd.testing.assert_frame_equal(short['search'], long['search'])
        self.assertEqual(len(long['forecast']), 24)

    def test_alignment_finite_and_negative_r2_not_hidden(self):
        with self.assertRaises(AssertionError):
            metric_values([1, 2], [1])
        with self.assertRaises(AssertionError):
            metric_values([1, 2], [1, np.inf])
        self.assertLess(metric_values([1, 2, 3], [10, 10, 10])['R2'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
