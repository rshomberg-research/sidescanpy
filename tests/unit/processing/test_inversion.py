"""
Unit tests for sidescanpy.processing.inversion.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np

from sidescanpy.processing import invert
from sidescanpy.processing.geometric import compute_geometric_ranges
from tests.unit.synthetic import make_raw_dataset as _make_raw_dataset


class TestInvertLegacyFile:
    def test_legacy_radiometric_file_restores_backscatter_raw_and_drops_geometry(self):
        ds = _make_raw_dataset()
        ds["backscatter_raw"] = ds["backscatter_intensity"].copy()
        ds = compute_geometric_ranges(ds)
        ds.attrs["processing_level"] = "radiometric"  # legacy: TVG known only from the attribute
        ds["backscatter_intensity"] = ds["backscatter_intensity"] * 0.0 + 999.0

        restored = invert(ds)

        assert restored.sidescan.processing_level == "raw"
        assert "processing_level" not in restored.attrs
        np.testing.assert_array_equal(
            restored["backscatter_intensity"].values, restored["backscatter_raw"].values
        )
        assert "ground_range_m" not in restored


class TestWithUncorrectedVar:
    def test_adds_uncorrected_equal_to_inverted_intensity(self):
        from sidescanpy.processing import UNCORRECTED_VAR, apply_tvg, with_uncorrected_var

        raw = _make_raw_dataset(backscatter=np.random.default_rng(0).uniform(1, 10, (5, 2, 40)).astype(np.float32))
        out = with_uncorrected_var(apply_tvg(raw))
        np.testing.assert_allclose(out[UNCORRECTED_VAR].values, raw["backscatter_intensity"].values, rtol=1e-4)
        assert UNCORRECTED_VAR not in raw

    def test_existing_variable_kept(self):
        from sidescanpy.processing import UNCORRECTED_VAR, with_uncorrected_var

        ds = _make_raw_dataset().assign({UNCORRECTED_VAR: (("ping", "channel", "sample"), np.zeros((20, 2, 200)))})
        assert with_uncorrected_var(ds) is ds
