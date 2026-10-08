"""
2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.export.image import compute_reference_stats, from_uint8, to_uint8, to_uint16


class TestToUint8:
    def test_uint16_round_trip_within_tolerance(self):
        rng = np.random.default_rng(0)
        original = rng.integers(0, 65535, size=(50, 50), dtype=np.uint16)
        img, meta = to_uint8(original)
        assert img.dtype == np.uint8
        assert meta.kind == "uint16"
        recovered = from_uint8(img, meta)
        assert recovered.dtype == np.uint16
        # lossy (8-bit quantization) — bounded, not exact
        in_range = (original >= meta.p_min) & (original <= meta.p_max)
        assert np.abs(recovered[in_range].astype(np.float64) - original[in_range]).mean() < 600

    def test_uint32_excludes_zero_from_percentile(self):
        array = np.zeros((10, 10), dtype=np.uint32)
        array[0, 0] = 1000
        array[5, 5] = 2000
        _img, meta = to_uint8(array)
        # percentile computed only over the two non-zero values, not the
        # 98 zeros — if zero were included, p_min would collapse near 0
        assert meta.p_min > 500
        assert meta.p_max <= 2000

    def test_float_kind_recorded_and_reversible(self):
        array = np.array([[-40.0, -20.0], [-10.0, 0.0]], dtype=np.float32)
        img, meta = to_uint8(array)
        assert meta.kind == "float"
        recovered = from_uint8(img, meta)
        assert recovered.dtype == np.float32

    def test_gamma_default_is_noop_relative_to_gamma_1(self):
        array = np.linspace(0, 65535, 100).astype(np.uint16)
        img_default, _ = to_uint8(array)
        img_explicit, _ = to_uint8(array, gamma=1.0)
        assert np.array_equal(img_default, img_explicit)

    def test_gamma_changes_output(self):
        array = np.linspace(1, 65535, 100).astype(np.uint16)
        img_gamma1, _ = to_uint8(array, gamma=1.0)
        img_gamma2_5, _ = to_uint8(array, gamma=2.5)
        assert not np.array_equal(img_gamma1, img_gamma2_5)

    def test_gamma_above_1_darkens_matching_powernorm_convention(self):
        # Regression guard: to_uint8 must apply normalized ** gamma (not
        # normalized ** (1/gamma)), matching matplotlib.colors.PowerNorm
        # -- the convention viz.waterfall.plot_waterfall/viz.gridded
        # already use. Getting this backwards silently washes out/
        # over-brightens real exports at any gamma > 1 (found on real
        # data: gamma=2.5, which correctly darkened background/enhanced
        # contrast via PowerNorm elsewhere in this library, produced a
        # washed-out result via the inverted convention this test locks
        # against).
        array = np.linspace(0, 65535, 100).astype(np.uint16)
        img_gamma1, _ = to_uint8(array, gamma=1.0)
        img_gamma2_5, _ = to_uint8(array, gamma=2.5)
        mid = slice(20, 80)  # away from the 0/1 clamp endpoints
        assert (img_gamma2_5[mid].astype(int) <= img_gamma1[mid].astype(int)).all()
        assert (img_gamma2_5[mid] < img_gamma1[mid]).any()

    def test_reference_stats_overrides_per_image_percentiles(self):
        array = np.full((10, 10), 100.0, dtype=np.float32)
        stats = compute_reference_stats(np.array([0.0, 1000.0]))
        img, meta = to_uint8(array, reference_stats=stats)
        assert meta.p_min == stats.p_min
        assert meta.p_max == stats.p_max
        # a constant-100 array stretched against a much wider [0,1000]
        # reference lands well below full brightness
        assert img.max() < 255

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError):
            to_uint8(np.array([1, 2, 3], dtype=np.int8))


class TestComputeReferenceStats:
    def test_pools_multiple_arrays(self):
        a = np.array([0.0, 10.0])
        b = np.array([90.0, 100.0])
        stats = compute_reference_stats(a, b, pct_low=0.0, pct_high=100.0)
        assert stats.p_min == 0.0
        assert stats.p_max == 100.0

    def test_excludes_non_finite(self):
        a = np.array([np.nan, np.inf, 5.0, 10.0])
        stats = compute_reference_stats(a, pct_low=0.0, pct_high=100.0)
        assert stats.p_min == 5.0
        assert stats.p_max == 10.0

    def test_all_non_finite_returns_default_range(self):
        stats = compute_reference_stats(np.array([np.nan, np.nan]))
        assert stats.p_min == 0.0
        assert stats.p_max == 1.0


class TestToUint16:
    def test_same_stretch_as_uint8_at_finer_quantization(self):
        rng = np.random.default_rng(0)
        array = rng.gamma(2.0, 10.0, size=(64, 64)).astype(np.float32)
        img8, meta8 = to_uint8(array, gamma=2.0)
        img16, meta16 = to_uint16(array, gamma=2.0)
        assert img16.dtype == np.uint16
        assert meta16 == meta8
        assert np.abs((img16 >> 8).astype(int) - img8.astype(int)).max() <= 1
        assert len(np.unique(img16)) > len(np.unique(img8))

    def test_round_trips_through_from_uint8(self):
        array = np.linspace(0.0, 100.0, 1000, dtype=np.float32).reshape(10, 100)
        img16, meta = to_uint16(array, pct_low=0.0, pct_high=100.0)
        np.testing.assert_allclose(from_uint8(img16, meta), array, atol=0.01)
