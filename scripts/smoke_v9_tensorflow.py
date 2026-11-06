"""Smoke test real de TensorFlow para el motor V9; usa datos sintéticos."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from lstm_residual_estacional import _fit_member, _predict_members, _training_pairs

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

rng = np.arange(135)
history = np.asarray(
    160.0 + 0.15 * rng + 14.0 * np.sin(2.0 * np.pi * rng / 52),
    dtype=np.float64,
)
member = _fit_member(
    history, fit_end=111, lookback=4, units=4,
    seed=42, max_epochs=3, patience=2,
    dropout=0.10, l2_value=1e-4, batch_size=16,
)
origins = np.full(12, 111, dtype=int)
steps = np.arange(1, 13)
forecast = _predict_members([member], history, origins, steps)
assert forecast.shape == (12,), forecast.shape
assert np.isfinite(forecast).all()
assert (forecast >= 0).all()
changed = history.copy()
changed[111:] += 100_000.0
causal = _predict_members([member], changed, origins, steps)
np.testing.assert_allclose(forecast, causal, atol=1e-5, rtol=0.0)
print("TensorFlow:", tf.__version__, "| 12 pronósticos finitos, no negativos y causales.")
print("Épocas elegidas:", member["epochs"])
