"""Regresiones de los intervalos guardados: ventanas de 12 y 24 períodos."""
import unittest
import pandas as pd
import numpy as np
from verify_workshop_artifacts import evaluation_origins


class ArtifactVerifierTests(unittest.TestCase):
    def rows(self, start, stop, horizon=3, group='TRAIN'):
        return pd.DataFrame([
            {'Modelo': 'LSTM', 'Grupo': group, 'Indice_origen': origin, 'Horizonte': h}
            for origin in range(start, stop-horizon+1)
            for h in range(1, horizon+1)
        ])

    def setUp(self):
        self.meta = {'train_end': 40, 'val_end': 50, 'config': {'horizon': 3}}

    def test_saved_training_interval_supports_both_windows(self):
        for start in (12, 24):
            with self.subTest(start=start):
                actual = evaluation_origins(self.meta, self.rows(start, 40), 'LSTM', 'TRAIN', 60)
                np.testing.assert_array_equal(actual, np.arange(start, 40))

    def test_validation_and_test_use_their_own_boundaries(self):
        for group, start, stop in [('VALIDATION', 40, 50), ('TEST', 50, 60)]:
            with self.subTest(group=group):
                rows = self.rows(start, stop, group=group)
                np.testing.assert_array_equal(
                    evaluation_origins(self.meta, rows, 'LSTM', group, 60), np.arange(start, stop))

    def test_missing_trajectory_is_rejected(self):
        rows = self.rows(24, 40).drop(index=4)
        with self.assertRaises(AssertionError):
            evaluation_origins(self.meta, rows, 'LSTM', 'TRAIN', 60)

    def test_shifted_test_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_origins(self.meta, self.rows(51, 60, group='TEST'), 'LSTM', 'TEST', 60)

    def test_missing_model_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_origins(self.meta, self.rows(24, 40), 'ausente', 'TRAIN', 60)


if __name__ == '__main__':
    unittest.main()
