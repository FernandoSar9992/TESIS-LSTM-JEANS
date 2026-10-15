"""Regresiones de causalidad estacional, reajuste y pronóstico recursivo."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand_data import prepare_series
from forecast_engine import period_positions
from lstm_engine import scaled_windows, future_input, inverse_prediction
from thesis_engine import ThesisConfig, run_thesis, finalize_test, generate_future, recursive_predictions
from test_thesis import simulated_network, make_series, FakeModel


class SeasonalIntegrityTests(unittest.TestCase):
    def test_future_encoding_identical_to_historical_and_no_lookahead(self):
        y = 100+np.arange(160)*.5+30*np.sin(np.arange(160)*2*np.pi/48)
        for mode in ('nivel', 'cambio', 'local'):
            x, targets, origins, anchors, center, scale = scaled_windows(y, 90, 8, 1, mode, 48)
            self.assertEqual(x.shape[-1], 6)
            for t in [20, 48, 90, 150]:
                xf, anchor = future_input(y[:t], 8, mode, center, scale, 48)
                i = np.flatnonzero(origins == t)[0]
                np.testing.assert_array_equal(xf[0], x[i])
                np.testing.assert_allclose(anchor, anchors[i])
            changed = y.copy()
            changed[100:] = 999999
            other = scaled_windows(changed, 90, 8, 1, mode, 48)
            np.testing.assert_array_equal(x[origins <= 100], other[0][origins <= 100])
            self.assertEqual((center, scale), other[-2:])
            np.testing.assert_array_equal(x[origins < 48, :, 5], 0)

    def test_annual_lag_reconstructs_seasonal_signal(self):
        y = 200+50*np.sin(np.arange(160)*2*np.pi/48)
        x, _, origins, anchors, center, scale = scaled_windows(y, 100, 8, 1, 'local', 48)
        # La quinta característica, último paso, es el objetivo del ciclo anterior
        # en la misma escala que la salida. Verifica columna y desfase exactos.
        model = type('AnnualLag', (), {'predict': lambda self, x, **kw: x[:, -1, 4:5]})()
        pred = inverse_prediction(model, x, anchors, center, scale).ravel()
        np.testing.assert_allclose(pred[origins >= 48], y[origins[origins >= 48]], atol=1e-5)

    def test_production_refit_never_changes_test_and_is_not_repeated(self):
        data = make_series()
        config = ThesisConfig(automatic=False, seasonal_features=False)
        with simulated_network():
            result = run_thesis(data, config, evaluate_test=False)
            old_scaler = result['scaler'].copy()
            models_before = list(result['_artifacts']['models'])
            finalize_test(result)
            self.assertEqual(result['production_scaler']['fit_end'], len(data))
            self.assertEqual(result['scaler'], old_scaler)
            self.assertEqual(result['_artifacts']['models'], models_before)
            scores = result['metrics'].copy()
            production = result['_artifacts']['production_models']
            result['config']['horizon'] = 12
            with patch('thesis_engine.fit_production', side_effect=AssertionError('No volver a entrenar')):
                generate_future(result)
            self.assertIs(result['_artifacts']['production_models'], production)
            pd.testing.assert_frame_equal(result['metrics'], scores)
            self.assertEqual(len(result['forecast']), 12)
            self.assertEqual(result['backtest_horizon'], 8)

    def test_recursive_paths_use_predictions_not_hidden_real_targets(self):
        data = make_series()
        with simulated_network():
            result = run_thesis(data, ThesisConfig(automatic=False, representation='cambio', seasonal_features=False))
        detail = result['recursive_detail'].query("Modelo == 'LSTM'")
        for start, group in detail.groupby('Indice_origen'):
            np.testing.assert_allclose(group.Prediccion, data.demanda.iloc[start-1], atol=1e-5)
        for _, row in result['horizon_metrics'].query("Modelo == 'LSTM'").iterrows():
            pairs = detail[(detail.Grupo == row.Grupo) & (detail.Horizonte == row.Horizonte)]
            self.assertEqual(row.N, len(pairs))
            self.assertAlmostEqual(row.RMSE, np.sqrt(np.mean((pairs.Real-pairs.Prediccion)**2)))

    def test_calendar_rejects_gaps_and_disorder(self):
        for labels in [['Enero 2023-Se1', 'Enero 2023-Se3'], ['Enero 2023-Se2', 'Enero 2023-Se1']]:
            with self.assertRaisesRegex(ValueError, 'saltos'):
                prepare_series(pd.DataFrame({'Periodo': labels, 'Demanda': [1, 2]}), 'Periodo', 'Demanda')

    def test_explicit_real_declaration_preserves_file_provenance(self):
        raw = pd.DataFrame({'Periodo': ['a', 'b'], 'Demanda': [1, 2]})
        self.assertTrue(prepare_series(raw, 'Periodo', 'Demanda', declared_real=True).data.origen.eq('REAL').all())
        self.assertTrue(prepare_series(raw, 'Periodo', 'Demanda').data.origen.eq('SIN_DECLARAR').all())
        raw['Origen'] = ['REAL', 'IMPUTADO']
        self.assertEqual(prepare_series(raw, 'Periodo', 'Demanda', declared_real=True).data.origen.tolist(), ['REAL', 'IMPUTADO'])


if __name__ == '__main__':
    unittest.main()
