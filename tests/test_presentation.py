import unittest
import pandas as pd
from presentation import data_preview, readable_table


class PresentationTests(unittest.TestCase):
    def test_display_preserves_source_and_mixed_provenance(self):
        data = pd.DataFrame({'periodo': ['a', 'b', 'c'], 'demanda': [1., 2., 3.],
                             'origen': ['REAL', 'IMPUTADO', 'SIN_DECLARAR']})
        before = data.copy(deep=True)
        shown = data_preview(data)
        self.assertEqual(shown.Procedencia.tolist(), ['Observado', 'Reconstruido', 'No indicado en el archivo'])
        pd.testing.assert_frame_equal(before, data)
        unknown = data.assign(origen='SIN_DECLARAR')
        self.assertEqual(data_preview(unknown).columns.tolist(), ['Período', 'Demanda'])

    def test_metric_labels_do_not_change_values(self):
        data = pd.DataFrame({'Grupo': ['TEST'], 'Alcance': ['SIN_TRAZABILIDAD'], 'R2': [-.319]})
        shown = readable_table(data)
        self.assertEqual(shown['R²'].iloc[0], -.319)
        self.assertEqual(shown['Datos evaluados'].iloc[0], 'Procedencia no indicada')
        self.assertEqual(data.Alcance.iloc[0], 'SIN_TRAZABILIDAD')
