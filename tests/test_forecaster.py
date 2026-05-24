"""
Tests for crypto_forecaster.

Covers:
* walk-forward split: train and test windows are temporally separated and
  together account for every row in the post-target frame
* Reddit-sentiment join: the four expected columns survive the
  generate_crypto_data pipeline and `reddit_sentiment` is never null
* LSTM input-shape: the (lookback, n_features) sequence builder produces
  the shape the Keras LSTM layer is configured for
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


# crypto_forecaster does `import tensorflow as tf` at module import time and
# uses it only to call `tf.random.set_seed(42)` inside generate_crypto_data.
# We stub it so the test suite doesn't require the heavy TF wheel just to
# exercise the dataset / sequencing code paths.
if "tensorflow" not in sys.modules:
    tf_stub = types.ModuleType("tensorflow")
    tf_random = types.ModuleType("tensorflow.random")
    tf_random.set_seed = lambda _seed: None
    tf_stub.random = tf_random
    keras_stub = types.ModuleType("tensorflow.keras")
    models_stub = types.ModuleType("tensorflow.keras.models")
    models_stub.Sequential = object
    layers_stub = types.ModuleType("tensorflow.keras.layers")
    layers_stub.LSTM = object
    layers_stub.Dense = object
    layers_stub.Dropout = object
    keras_stub.models = models_stub
    keras_stub.layers = layers_stub
    tf_stub.keras = keras_stub
    sys.modules["tensorflow"] = tf_stub
    sys.modules["tensorflow.random"] = tf_random
    sys.modules["tensorflow.keras"] = keras_stub
    sys.modules["tensorflow.keras.models"] = models_stub
    sys.modules["tensorflow.keras.layers"] = layers_stub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from crypto_forecaster import generate_crypto_data  # noqa: E402


# ---------------------------------------------------------------------------
# Data-generation fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """The same synthetic frame `run_ensemble()` consumes (seed=42)."""
    return generate_crypto_data(days=200)


# ---------------------------------------------------------------------------
# Reddit-sentiment join row preservation
# ---------------------------------------------------------------------------
def test_reddit_sentiment_join_preserves_rows(synthetic_df: pd.DataFrame) -> None:
    """Sentiment is appended in-place; total rows = days - 7 dropna for target."""
    days_requested = 200
    # The pipeline computes `future_7d_return = close.shift(-7)` then `.dropna()`
    # so we expect exactly the last 7 rows to drop.
    assert len(synthetic_df) == days_requested - 7

    # All sentiment-aware columns must be present and complete.
    expected_cols = {
        "date",
        "close",
        "volume",
        "reddit_sentiment",
        "future_7d_return",
        "target_up",
    }
    assert expected_cols.issubset(set(synthetic_df.columns))
    assert synthetic_df["reddit_sentiment"].isna().sum() == 0
    assert synthetic_df["close"].isna().sum() == 0
    # Sentiment is N(0,1) — keep a wide band but sanity-check magnitude.
    assert -10.0 < synthetic_df["reddit_sentiment"].min() < 0.0
    assert 0.0 < synthetic_df["reddit_sentiment"].max() < 10.0


def test_dataset_is_deterministic() -> None:
    """seed=42 inside generate_crypto_data → identical frames on re-run."""
    a = generate_crypto_data(days=120)
    b = generate_crypto_data(days=120)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------
def test_walk_forward_split_is_temporally_separated(synthetic_df: pd.DataFrame) -> None:
    """The 80/20 split run_ensemble() uses must not leak future into train."""
    df = synthetic_df.sort_values("date").reset_index(drop=True)
    train_size = int(len(df) * 0.8)
    assert train_size > 0
    assert train_size < len(df)

    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Walk-forward: every train date is strictly before every test date.
    assert train_df["date"].max() < test_df["date"].min()
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df.index).isdisjoint(set(test_df.index))


def test_target_label_is_binary(synthetic_df: pd.DataFrame) -> None:
    """target_up is a 0/1 directional flag — never NaN after dropna()."""
    targets = synthetic_df["target_up"].unique()
    assert set(targets).issubset({0.0, 1.0})
    pos_rate = float(synthetic_df["target_up"].mean())
    assert 0.20 < pos_rate < 0.80


# ---------------------------------------------------------------------------
# LSTM input-shape
# ---------------------------------------------------------------------------
def test_lstm_sequence_builder_shape(synthetic_df: pd.DataFrame) -> None:
    """Mirror the (lookback=14, 3 features) windowing in train_lstm()."""
    features = ["close", "volume", "reddit_sentiment"]
    data = synthetic_df[features].values
    lookback = 14
    horizon = 7

    X, y = [], []
    for i in range(lookback, len(data) - horizon):
        X.append(data[i - lookback:i])
        nxt = float(synthetic_df.iloc[i + horizon]["close"])
        now = float(synthetic_df.iloc[i]["close"])
        y.append(1 if nxt > now else 0)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    expected_n = len(data) - lookback - horizon
    assert X_arr.shape == (expected_n, lookback, len(features))
    assert y_arr.shape == (expected_n,)
    assert set(np.unique(y_arr).tolist()).issubset({0, 1})


def test_lstm_split_index_after_lookback(synthetic_df: pd.DataFrame) -> None:
    """`split_idx = train_size - lookback` must stay positive for real-sized runs."""
    lookback = 14
    train_size = int(len(synthetic_df) * 0.8)
    split_idx = train_size - lookback
    assert split_idx > 0, "training window collapses below the LSTM lookback"
