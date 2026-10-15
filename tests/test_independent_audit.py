"""Falsación independiente de causalidad; no usa datos TEST del taller."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from workshop_engine import Encoder, WorkshopConfig, fit_ridge, predict, pairs, score, train_workshop, finalize_workshop
from workshop_planning import METHODS, client_predictions


class IndependentAuditTests(unittest.TestCase):
    def test_seasonal_training_start_and_saved_model_causality(self):
        from workshop_engine import fit_neural
        from auditar_taller import save_artifact
        from verify_workshop_artifacts import load_artifact
        from tempfile import TemporaryDirectory
        from pathlib import Path
        config = WorkshopConfig(epochs=1, members=1)
        candidate = {'family': 'LSTM', 'units': 8, 'dropout': .1,
                     'lookback': 12, 'residual': 'seasonal', 'train_start': 48}
        with patch('workshop_engine.pairs', wraps=pairs) as calls:
            model = fit_neural(self.data, 70, 70, config, candidate, None, fixed_epochs=1)
        calls.assert_called_once_with(48, 70, 8)
        origins, steps = np.full(8, 70), np.arange(1, 9)
        expected = predict(model, self.data, origins, steps)
        modified = self.data.copy()
        modified.loc[70:, ['demanda', 'precio_unitario', 'costo_unitario']] = 1e10
        modified.loc[70:, 'cliente'] = 'FUTURO'
        np.testing.assert_array_equal(expected, predict(model, modified, origins, steps))
        with TemporaryDirectory() as temp:
            prefix = Path(temp)/'candidate'
            save_artifact('experimental', model, prefix)
            _, restored = load_artifact(prefix.with_suffix('.json'), self.data)
            np.testing.assert_allclose(expected, predict(restored, self.data, origins, steps), atol=1e-5)

    def setUp(self):
        t = np.arange(190)
        self.data = pd.DataFrame({
            'periodo': [f'p{i}' for i in t],
            'demanda': 300 + 90*np.sin(t*2*np.pi/48) + t*.2,
            'cliente': np.where(t % 5 == 0, 'A', 'B'),
            'precio_unitario': 1.5 + .2*(t % 5 == 0),
            'costo_unitario': 1.34,
            'origen': 'REAL',
        })

    def test_adversarial_future_cannot_change_fitted_ridge_forecasts(self):
        config = WorkshopConfig(horizon=8)
        model = fit_ridge(self.data, 70, config, {'alpha': 10., 'history': True})
        for origin in (70, 96, 123, 149):
            with self.subTest(origin=origin):
                origins = np.full(48, origin)
                steps = np.arange(1, 49)
                before = predict(model, self.data, origins, steps)
                future = self.data.copy()
                future.loc[origin:, 'demanda'] = 1e12
                future.loc[origin:, 'cliente'] = 'FUTURO_NUEVO'
                future.loc[origin:, ['precio_unitario', 'costo_unitario']] = 1e9
                np.testing.assert_array_equal(before, predict(model, future, origins, steps))

    def test_encoder_statistics_ignore_every_row_after_fit_boundary(self):
        config = WorkshopConfig()
        changed = self.data.copy()
        changed.loc[70:, ['demanda', 'precio_unitario', 'costo_unitario']] = 1e10
        changed.loc[70:, 'cliente'] = 'NO_CONOCIDO_AL_AJUSTAR'
        first, second = Encoder(self.data, 70, config), Encoder(changed, 70, config)
        self.assertEqual(first.customers, second.customers)
        self.assertEqual(first.center, second.center)
        self.assertEqual(first.scale, second.scale)
        for attribute in ('mu', 'sd', 'active_numeric'):
            np.testing.assert_array_equal(getattr(first, attribute), getattr(second, attribute))

    def test_complete_trajectories_never_cross_any_split(self):
        for start, stop in ((71, 97), (97, 123), (123, 149), (149, 176)):
            for horizon in (1, 4, 8, 24):
                with self.subTest(start=start, horizon=horizon):
                    origins, steps = pairs(start, stop, horizon)
                    self.assertTrue(np.all(origins >= start))
                    self.assertTrue(np.all(origins + steps - 1 < stop))
                    for origin in np.unique(origins):
                        np.testing.assert_array_equal(steps[origins == origin], np.arange(1, horizon+1))

    def test_customer_prediction_is_invariant_to_future_labels_and_demands(self):
        for origin in (12, 70, 149):
            modified = self.data.copy()
            modified.loc[origin:, 'cliente'] = 'FUTURO_NUEVO'
            modified['demanda'] = 1e12
            for method in METHODS:
                targets = np.arange(origin, origin+48)
                self.assertEqual(client_predictions(self.data, origin, targets, 48, method),
                                 client_predictions(modified, origin, targets, 48, method))

    def test_mase_denominator_is_frozen_at_training_prefix(self):
        targets = np.arange(100, 108)
        pred = self.data.demanda.iloc[targets].to_numpy() + 10
        first = score(self.data, targets, pred, 70, 48)
        changed = self.data.copy()
        changed.loc[70:99, 'demanda'] = 1e12
        second = score(changed, targets, pred, 70, 48)
        expected = float(np.mean(np.abs(self.data.demanda.iloc[48:70].to_numpy()
                                        - self.data.demanda.iloc[:22].to_numpy())))
        self.assertAlmostEqual(first['Escala_MASE'], expected)
        self.assertAlmostEqual(first['MASE'], 10/expected)
        self.assertEqual(first['MASE'], second['MASE'])

    def test_seasonal_residual_zero_network_equals_only_available_annual_values(self):
        config = WorkshopConfig()
        artifact = {'encoder': Encoder(self.data, 70, config),
                    'candidate': {'family': 'LSTM', 'lookback': 24, 'residual': 'seasonal'},
                    'network': lambda inputs, **kwargs: np.zeros((len(inputs[0]), 1))}
        origin = 100
        steps = np.arange(1, 49)
        expected = self.data.demanda.iloc[origin-48:origin].to_numpy()
        actual = predict(artifact, self.data, np.full(48, origin), steps)
        np.testing.assert_array_equal(actual, expected)
        changed = self.data.copy()
        changed.loc[origin:, 'demanda'] = 1e12
        np.testing.assert_array_equal(actual, predict(artifact, changed, np.full(48, origin), steps))

    def test_failed_production_refit_leaves_evaluation_retryable_without_duplicates(self):
        from test_workshop import fake_training
        with fake_training():
            result = train_workshop(self.data, WorkshopConfig())
            before = {key: result[key].copy(deep=True) for key in ('split_metrics', 'detail', 'horizon_metrics')}
            with patch('workshop_engine.fit_neural', side_effect=RuntimeError('Fallo simulado de reajuste')):
                with self.assertRaisesRegex(RuntimeError, 'simulado'):
                    finalize_workshop(result)
            self.assertFalse(result['evaluated'])
            self.assertNotIn('production', result['_artifacts'])
            for key, frame in before.items():
                pd.testing.assert_frame_equal(result[key], frame)
            finalize_workshop(result)
        self.assertTrue(result['evaluated'])
        test_rows = result['split_metrics'].query("Grupo == 'TEST'")
        self.assertEqual(len(test_rows), len(result['_artifacts']['models']))
        self.assertFalse(test_rows.Modelo.duplicated().any())


if __name__ == '__main__':
    unittest.main()
