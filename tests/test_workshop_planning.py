"""Cliente estacional, causalidad y salida simplificada para planificación."""
import unittest
from io import BytesIO
import numpy as np
import pandas as pd
from workshop_planning import (client_predictions, select_client_model, evaluate_client_model,
                               planning_forecast, METHODS)
from workshop_app import export_workshop


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame({'cliente': np.tile(['A', 'B', 'B', 'B'], 44),
                                     'demanda': np.arange(176), 'precio_unitario': 1.5})

    def test_seasonality_without_demand_threshold(self):
        model = select_client_model(self.history, 123, 149, 8, 48)
        self.assertNotEqual(model['selected'], METHODS[0])
        evaluate_client_model(model, self.history, 149)
        metric = model['metrics'].query("Grupo == 'TEST'").iloc[0]
        self.assertEqual(metric.Exactitud_balanceada, 1)
        forecast = pd.DataFrame({'Periodo': ['f1', 'f2'], 'elegido': [999., 1.]})
        result = planning_forecast(forecast, self.history, 'elegido', model)
        self.assertEqual(list(result.columns), ['Periodo', 'Pronostico', 'Cliente'])
        self.assertEqual(result.Cliente.tolist(), ['A', 'B'])
        self.assertEqual(result.Pronostico.tolist(), [999., 1.])
        self.assertEqual(list(forecast.columns), ['Periodo', 'elegido'])

    def test_future_clients_cannot_change_selection_or_origin_predictions(self):
        changed = self.history.copy()
        changed.loc[149:, 'cliente'] = 'C'
        first = select_client_model(self.history, 123, 149, 8, 48)
        second = select_client_model(changed, 123, 149, 8, 48)
        self.assertEqual(first['selected'], second['selected'])
        pd.testing.assert_frame_equal(first['metrics'], second['metrics'])
        for method in METHODS:
            self.assertEqual(client_predictions(self.history, 149, [149, 170, 210], 48, method),
                             client_predictions(changed, 149, [149, 170, 210], 48, method))
        changed['demanda'] = 99999
        changed['precio_unitario'] = 99999
        self.assertEqual(client_predictions(self.history, 123, [123, 130], 48, first['selected']),
                         client_predictions(changed, 123, [123, 130], 48, first['selected']))

    def test_majority_fallback_and_repeated_cycles(self):
        short = pd.DataFrame({'cliente': ['A', 'B', 'B']})
        self.assertEqual(client_predictions(short, 3, [3, 4], 48, METHODS[1]), ['B', 'B'])
        self.assertEqual(client_predictions(self.history, 176, [176, 224, 272], 48, METHODS[1]), ['A']*3)

    def test_excel_contains_only_period_forecast_client(self):
        model = select_client_model(self.history, 123, 149, 8, 48)
        evaluate_client_model(model, self.history, 149)
        count = len(model['metrics'])
        evaluate_client_model(model, self.history, 149)
        self.assertEqual(len(model['metrics']), count)
        forecast = pd.DataFrame({'Periodo': ['f1', 'f2'], 'elegido': [318., 320.]})
        planned = planning_forecast(forecast, self.history, 'elegido', model)
        result = {key: pd.DataFrame({'dato': [1]}) for key in (
            'processed', 'split_metrics', 'horizon_metrics', 'detail', 'search', 'validation_folds', 'feature_audit')}
        result.update(evaluated=True, config={'mode': 'historico'}, selected='elegido',
                      data_hash='test', engine_version='test', forecast=forecast,
                      planning_forecast=planned, client_model=model)
        saved = pd.read_excel(BytesIO(export_workshop(result)), sheet_name='Pronostico')
        self.assertEqual(list(saved.columns), ['Periodo', 'Pronostico', 'Cliente'])
        self.assertEqual(saved.Cliente.tolist(), ['A', 'B'])


if __name__ == '__main__':
    unittest.main()
