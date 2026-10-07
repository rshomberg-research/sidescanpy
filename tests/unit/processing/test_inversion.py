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
