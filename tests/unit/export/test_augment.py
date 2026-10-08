"""
Unit tests for sidescanpy.export.augment.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest
from pyproj import Geod

from sidescanpy.core.target import Target
from sidescanpy.export.augment import (
    augmented_file_name,
    jitter_latlon,
    jitter_offset_m,
    random_flip,
)


def _t(sample, row):
    return Target(target_id="1", source_format="x", catalog_source="x", sample=sample, extra_attributes={"window_row": row})


class _FixedRng:
    """Stands in for a Generator whose next ``random()`` draws are known."""

    def __init__(self, draws):
        self.draws = list(draws)

    def random(self):
        return self.draws.pop(0)


class TestRandomFlip:
    @pytest.mark.parametrize("edge_coords, off", [(False, 1), (True, 0)])
    def test_both_flips_remap_targets(self, edge_coords, off):
        window = np.arange(12.0).reshape(3, 4)
        out, (t,), fh, fv = random_flip(window, [_t(1, 0)], _FixedRng([0.1, 0.2]), edge_coords=edge_coords)
        assert fh and fv
        np.testing.assert_array_equal(out, window[::-1, ::-1])
        assert t.sample == 4 - off - 1
        assert t.extra_attributes["window_row"] == 3 - off - 0

    def test_no_flip_leaves_window_and_targets(self):
        window = np.arange(6.0).reshape(2, 3)
        out, (t,), fh, fv = random_flip(window, [_t(2, 1)], _FixedRng([0.9, 0.6]), edge_coords=False)
        assert not fh and not fv
        np.testing.assert_array_equal(out, window)
        assert (t.sample, t.extra_attributes["window_row"]) == (2, 1)

    def test_output_is_contiguous_and_rng_advances_twice(self):
        rng = np.random.default_rng(3)
        out, _, _, _ = random_flip(np.zeros((4, 5)), [], rng, edge_coords=True)
        assert out.flags["C_CONTIGUOUS"]
        reference = np.random.default_rng(3)
        reference.random()
        reference.random()
        assert rng.random() == reference.random()


class TestJitter:
    def test_zero_jitter_does_not_draw(self):
        rng = np.random.default_rng(0)
        assert jitter_offset_m(0, 0.5, rng) == 0.0
        assert rng.random() == np.random.default_rng(0).random()

    def test_offset_is_whole_pixels_within_range(self):
        rng = np.random.default_rng(1)
        draws = [jitter_offset_m(3, 0.25, rng) for _ in range(200)]
        assert set(np.round(np.array(draws) / 0.25, 9)) <= set(range(-3, 4))
        assert min(draws) == -0.75 and max(draws) == 0.75

    def test_latlon_offset_is_the_requested_distance(self):
        lat, lon = jitter_latlon(40.0, -70.0, 30.0, -40.0)
        _az, _baz, dist = Geod(ellps="WGS84").inv(-70.0, 40.0, lon, lat)
        assert dist == pytest.approx(50.0, rel=1e-3)
        assert lat > 40.0 and lon < -70.0

    def test_latlon_zero_offset_is_identity(self):
        assert jitter_latlon(40.0, -70.0, 0.0, 0.0) == (40.0, -70.0)


def test_augmented_file_name():
    assert augmented_file_name("target_7", 0, 1) == "target_7.png"
    assert augmented_file_name("target_7", 2, 4) == "target_7_a2.png"
    assert augmented_file_name("bg_3", 0, 2) == "bg_3_a0.png"
