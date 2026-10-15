"""Causalidad, separación de clientes, modelos y flujo del modo taller."""
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop_data import prepare_workshop, workshop_columns
from workshop_engine import WorkshopConfig, Encoder, pairs, train_workshop, finalize_workshop, forecast_workshop, fit_ridge, predict
from workshop_app import export_workshop


def raw_data(n=176):
    from forecast_engine import MONTHS
    t = np.arange(n)
    customers = np.where(t % 3 == 0, 'A', 'B')
    demand = 200+40*(customers == 'B')+25*np.sin(2*np.pi*t/48)
    price = np.where(customers == 'A', 1.7, 1.5)
    return pd.DataFrame({'Semana': [f'{MONTHS[(i%48)//4].capitalize()} {2023+i//48}-Se{i%4+1}' for i in t],
                         'Cliente': customers, 'Demanda (unidades)': demand, 'Precio Unitario ($/pantalón)': price,
                         'Ingreso Total ($)': demand*price, 'Costo Unitario ($)': 1.34,
                         'Costo Total ($)': demand*1.34, 'Utilidad Neta ($)': demand*(price-1.34)})


class ZeroResidual:
    def __call__(self, inputs, **kwargs):
        return np.zeros((len(inputs[0]), 1))


@contextmanager
def fake_training():
    def fake(data, fit_end, stop, config, candidate, ridge_candidate, fixed_epochs=None):
        artifact = fit_ridge(data, fit_end, config, ridge_candidate) if candidate['family'] == 'LSTM + Ridge' else {
            'encoder': Encoder(data, fit_end, config), 'ridge': None, 'ridge_candidate': ridge_candidate}
        return {**artifact, 'network': ZeroResidual(), 'candidate': candidate, 'epochs': fixed_epochs or 1,
                'history': {'loss': [1.], 'val_loss': [1.]}}
    with patch('workshop_engine.fit_neural', side_effect=fake):
        yield


class WorkshopTests(unittest.TestCase):
    def setUp(self):
        self.data = prepare_workshop(raw_data(), declared_real=True).data

    def test_mapping_source_and_excluded_totals(self):
        raw = raw_data()
        prepared = prepare_workshop(raw)
        self.assertEqual(workshop_columns(raw)['period'], 'Semana')
        self.assertEqual(prepared.data.demanda.tolist(), raw['Demanda (unidades)'].tolist())
        self.assertFalse(any('total' in c.lower() or 'utilidad' in c.lower() for c in prepared.data))
        modified = raw.copy()
        modified['Ingreso Total ($)'] = 9999999
        modified['Costo Total ($)'] = -9999
        pd.testing.assert_frame_equal(prepared.data, prepare_workshop(modified).data)

    def test_historical_mode_cannot_see_target_customer_price_cost_or_demand(self):
        enc = Encoder(self.data, 71, WorkshopConfig())
        origins, steps = np.full(8, 100), np.arange(1, 9)
        before = enc.encode(self.data, origins, steps)
        changed = self.data.copy()
        changed.loc[100:, ['demanda', 'precio_unitario', 'costo_unitario']] = 100000
        changed.loc[100:, 'cliente'] = 'DESCONOCIDO'
        after = enc.encode(changed, origins, steps)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        new = Encoder(changed, 71, WorkshopConfig())
        self.assertEqual(enc.customers, new.customers)
        self.assertEqual(enc.center, new.center)
        np.testing.assert_array_equal(enc.mu, new.mu)

    def test_known_plan_includes_covariates_but_not_future_demand(self):
        enc = Encoder(self.data, 71, WorkshopConfig(mode='plan'))
        before = enc.encode(self.data, [100], [3])
        changed = self.data.copy()
        changed.loc[100:, 'demanda'] = 100000
        for a, b in zip(before, enc.encode(changed, [100], [3])):
            np.testing.assert_array_equal(a, b)
        changed.loc[102, 'cliente'] = 'NUEVO'
        self.assertFalse(np.array_equal(before[1], enc.encode(changed, [100], [3])[1]))

    def test_ablation_removes_business_variables(self):
        config = WorkshopConfig(use_business=False, mode='plan')
        enc = Encoder(self.data, 71, config)
        changed = self.data.copy()
        changed['cliente'] = 'NUEVO'
        changed['precio_unitario'] = 999
        changed['costo_unitario'] = 500
        for a, b in zip(enc.encode(self.data, [100], [3]), enc.encode(changed, [100], [3])):
            np.testing.assert_array_equal(a, b)

    def test_constant_cost_has_no_unlearned_effect_on_quantity(self):
        enc = Encoder(self.data, 71, WorkshopConfig(mode='plan'))
        changed = self.data.copy()
        changed.loc[102, 'costo_unitario'] = 1000
        for a, b in zip(enc.encode(self.data, [100], [3]), enc.encode(changed, [100], [3])):
            np.testing.assert_array_equal(a, b)

    def test_pair_purge_and_real_future_matches_historical_encoding(self):
        origins, steps = pairs(71, 97, 8)
        self.assertEqual(len(origins), 19*8)
        self.assertTrue(np.all(origins >= 71))
        self.assertTrue(np.all(origins+steps-1 < 97))
        for mode in ('historico', 'plan'):
            enc = Encoder(self.data, 71, WorkshopConfig(mode=mode))
            expected = enc.encode(self.data, [100], [4])
            actual = enc.encode(self.data.iloc[:100], [100], [4], plan=self.data.iloc[100:108].reset_index(drop=True))
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)

    def test_test_values_do_not_change_selection(self):
        changed = self.data.copy()
        changed.loc[149:, ['demanda', 'precio_unitario', 'costo_unitario']] = 999999
        changed.loc[149:, 'cliente'] = 'TEST_NUEVO'
        with fake_training():
            first = train_workshop(self.data, WorkshopConfig())
            second = train_workshop(changed, WorkshopConfig())
        self.assertEqual(first['selected'], second['selected'])
        for key in ('search', 'validation_folds', 'split_metrics', 'horizon_metrics'):
            pd.testing.assert_frame_equal(first[key], second[key])
        self.assertFalse(first['evaluated'])

    def test_final_production_and_export(self):
        with fake_training():
            result = train_workshop(self.data, WorkshopConfig())
            old_enc = result['_artifacts']['models'][result['selected']]['encoder']
            finalize_workshop(result)
            scores = result['split_metrics'].copy()
            forecast_workshop(result, [f'f{i}' for i in range(8)])
        self.assertEqual(old_enc.fit_end, 123)
        self.assertEqual(result['_artifacts']['production'][result['selected']]['encoder'].fit_end, 176)
        pd.testing.assert_frame_equal(scores, result['split_metrics'])
        self.assertEqual(len(result['forecast']), 8)
        self.assertEqual(set(result['split_metrics'].query("Grupo == 'TEST'").N), {27})
        with self.assertRaisesRegex(ValueError, 'ya fue evaluado'):
            finalize_workshop(result)
        with pd.ExcelFile(BytesIO(export_workshop(result))) as book:
            self.assertIn('Error_por_horizonte', book.sheet_names)
            self.assertEqual(len(pd.read_excel(book, 'Pronostico')), 8)

    def test_future_plan_required_and_margins_are_outputs(self):
        with fake_training():
            result = train_workshop(self.data, WorkshopConfig(mode='plan'))
            finalize_workshop(result)
        labels = [f'f{i}' for i in range(8)]
        with self.assertRaisesRegex(ValueError, 'requiere'):
            forecast_workshop(result, labels)
        plan = pd.DataFrame({'cliente': ['A']*8, 'precio_unitario': [1.7]*8, 'costo_unitario': [1.34]*8})
        forecast_workshop(result, labels, plan)
        f = result['forecast']
        np.testing.assert_allclose(f.Ingreso_estimado, f[result['thesis_selected']]*1.7)
        np.testing.assert_allclose(f.Margen_estimado, f[result['thesis_selected']]*(1.7-1.34))

    def test_gapped_customer_filter_not_treated_as_consecutive_periods(self):
        raw = raw_data()
        with self.assertRaisesRegex(ValueError, 'saltos'):
            prepare_workshop(raw[raw.Cliente.eq('A')])


class WorkshopAppTests(unittest.TestCase):
    def test_route_train_final_and_invalidation(self):
        from streamlit.testing.v1 import AppTest
        upload = BytesIO(raw_data().to_csv(index=False).encode())
        upload.name = 'taller.csv'
        path = str(Path(__file__).resolve().parents[1]/'app.py')
        with patch('streamlit.file_uploader', return_value=upload), fake_training():
            app = AppTest.from_file(path, default_timeout=60)
            app.session_state['workshop_known'] = True
            app.run()
            self.assertFalse(app.exception)
            self.assertIn('Métricas', [tab.label for tab in app.tabs])
            self.assertNotIn('Cliente y valores unitarios se conocen antes de pronosticar', [item.label for item in app.checkbox])
            self.assertNotIn('Preparar la serie', [item.label for item in app.expander])
            self.assertNotIn('Usar cliente, precio y costo unitario', [item.label for item in app.checkbox])
            self.assertNotIn('Filtrar producto, tienda u otra serie', [item.label for item in app.selectbox])
            app.button(key='workshop_run').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(app.session_state['workshop_result']['evaluated'])
            processed = app.session_state['workshop_result']['processed']
            self.assertEqual(len(processed), len(raw_data()))
            self.assertEqual(set(processed.cliente), {'A', 'B'})
            self.assertTrue(processed.origen.eq('REAL').all())
            app.button(key='workshop_final').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(app.session_state['workshop_result']['evaluated'])
            forecast_tab = next(tab for tab in app.tabs if tab.label == 'Pronóstico del taller')
            metric_tab = next(tab for tab in app.tabs if tab.label == 'Métricas')
            result = app.session_state['workshop_result']
            expected = result['split_metrics'].query("Grupo == 'TEST'").set_index('Modelo').loc[result['thesis_selected']]
            self.assertEqual([item.value for item in forecast_tab.metric],
                             [f'{expected.MAE:.2f}', f'{expected.RMSE:.2f}'])
            displayed = metric_tab.dataframe[0].value
            self.assertEqual(displayed['Etapa'].tolist(), ['Entrenamiento', 'Validación', 'Prueba final'])
            self.assertEqual(list(displayed.columns), ['Etapa', 'Observaciones', 'MAE (unidades)',
                             'RMSE (unidades)', 'MAPE (%)', 'MSE (unidades²)', 'R²', 'MASE', 'RMSSE', 'Sesgo (unidades)'])
            np.testing.assert_allclose(displayed.iloc[-1]['RMSSE'], expected.RMSSE, equal_nan=True)
            np.testing.assert_allclose(displayed['MSE (unidades²)'], displayed['RMSE (unidades)']**2)
            self.assertAlmostEqual(displayed.iloc[-1]['MAPE (%)'], expected['MAPE_%'])
            self.assertAlmostEqual(displayed.iloc[-1]['R²'], expected.R2)
            self.assertEqual(len(app.session_state['workshop_result']['forecast']), 8)
            self.assertEqual(app.session_state['workshop_result']['config']['mode'], 'historico')
            self.assertEqual(list(app.session_state['workshop_result']['planning_forecast'].columns), ['Periodo', 'Pronostico', 'Cliente'])
            self.assertTrue(app.session_state['workshop_result']['client_model']['evaluated'])
            self.assertFalse(any('No necesitas completar un plan' in item.value for item in app.info))
            self.assertFalse(app.get('data_editor'))
            app.number_input(key='workshop_epochs').set_value(190).run()
            self.assertIsNone(app.session_state['workshop_result'])
            app.button(key='workshop_run').click().run()
            self.assertTrue(app.button(key='workshop_final').disabled)


if __name__ == '__main__':
    unittest.main()
