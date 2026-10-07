"""
Unit tests for sidescanpy.analysis.characterize.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np

from sidescanpy.analysis.characterize import characterize_values, infer_units
from tests.unit.synthetic import make_raw_dataset as _make_raw_dataset


class TestCharacterizeValues:
    def test_basic_metrics(self):
        bs = np.zeros((5, 2, 10), dtype=np.float32)
        bs[..., 5:] = 4.0  # half zeros, half value=4
        ds = _make_raw_dataset(n_pings=5, n_samples=10, backscatter=bs)

        result = characterize_values(ds)

        assert result.dtype == "float32"
        assert result.minimum == 0.0
        assert result.maximum == 4.0
        assert np.isclose(result.mean, 2.0)
        assert np.isclose(result.pct_zero, 50.0)
        assert result.pct_negative == 0.0

    def test_negative_values_detected(self):
        bs = np.full((3, 2, 10), -5.0, dtype=np.float32)
        ds = _make_raw_dataset(n_pings=3, n_samples=10, backscatter=bs)
        result = characterize_values(ds)
        assert result.pct_negative == 100.0


class TestInferUnits:
    def test_db_like_values_classified_as_db(self):
        s1 = characterize_values(
            _make_raw_dataset(backscatter=np.full((3, 2, 10), -40.0, dtype=np.float32))
        )
        result = infer_units(s1)
        assert result.units == "dB"

    def test_raw_counts_classified(self):
        bs = np.zeros((3, 2, 10), dtype=np.float32)
        bs[..., 5:] = 81.0
        s1 = characterize_values(_make_raw_dataset(backscatter=bs))
        result = infer_units(s1)
        assert result.units == "raw_counts"

    def test_normalized_classified(self):
        bs = np.full((3, 2, 10), 0.5, dtype=np.float32)
        s1 = characterize_values(_make_raw_dataset(backscatter=bs))
        result = infer_units(s1)
        assert result.units == "normalized"
