"""
Unit tests for sidescanpy.processing.range_correction — the empirical,
data-driven range correction for a survey's residual TVG range-dependence.

2026 rshomberg@uri.edu
"""

import json

import numpy as np
import pytest

from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.radiometric import apply_tvg
from sidescanpy.processing.range_correction import (
    STEP_NAME,
    RangeCorrectionTable,
    apply_empirical_range_correction,
    fit_range_correction,
)
from tests.conftest import make_minimal_sidescan_dataset


def _make_radiometric_geometric_dataset(
    n_pings=20,
    n_samples=3000,
    freq_hz_start=1_495_000,
    altitude_m=0.05,
    sample_rate_hz=None,
):
    """
    Build a real tvg+geometric corrected dataset (via the actual library
    functions, not hand-built fields) with a small altitude so most
    samples fall outside the water column (finite ground_range_m) while
    a handful of near-in samples remain NaN-masked — needed to exercise
    both the "normal" and "water-column" code paths.
    """
    ds = make_minimal_sidescan_dataset(
        n_pings=n_pings,
        n_samples=n_samples,
        freq_hz_start=freq_hz_start,
        backscatter_val=1.0,
    )
    if sample_rate_hz is not None:
        ds.attrs["sample_rate_hz"] = sample_rate_hz
    ds = ds.assign(altitude_m=(["ping"], np.full(n_pings, altitude_m, dtype=np.float32)))
    ds = apply_tvg(ds)
    ds = compute_geometric_ranges(ds)
    return ds


def _inject_range_dependent_residual(ds, shape_fn):
    """Overwrite backscatter_intensity with a known function of
    ground_range_m, so fit_range_correction has a known target to recover."""
    range_m = ds["ground_range_m"]
    injected = shape_fn(range_m)
    return ds.assign(backscatter_intensity=injected)


class TestFitRangeCorrection:
    def test_requires_tvg(self):
        ds = make_minimal_sidescan_dataset()  # still "raw"
        ds = compute_geometric_ranges(ds)
        with pytest.raises(ValueError, match="TVG-corrected"):
            fit_range_correction(ds)

    def test_requires_ground_range_m(self):
        ds = apply_tvg(make_minimal_sidescan_dataset())  # no geometric step
        with pytest.raises(ValueError, match="ground_range_m"):
            fit_range_correction(ds)

    def test_flattens_known_bowl_shaped_residual(self):
        """A sharp, non-monotonic shape (deliberately not something a
        low-order polynomial fits well) — mirrors the real hz_1495000
        'bowl' residual shape found on real data. Uses a real ~60m range
        span (sample_rate_hz overridden down so the fixture doesn't need
        tens of thousands of samples to reach it) so the sin period is
        well-resolved by 2m bins — with only a handful of bins spanning a
        few meters, the piecewise-linear fit's inherent
        interpolation-smoothing error dominates and this assertion isn't
        meaningful; that's a property of the linear-interpolation
        approach (see RangeCorrectionTable.correction_db_at docstring),
        not a bug."""
        ds = _make_radiometric_geometric_dataset(
            n_pings=10, n_samples=400, sample_rate_hz=5000.0
        )
        ds = _inject_range_dependent_residual(
            ds, lambda r: 30.0 + 10.0 * np.sin(r.astype(np.float64) / 5.0)
        )

        range_m, correction_db, meta = fit_range_correction(
            ds, bin_width_m=2.0, exclude_below_db=None
        )
        assert len(range_m) == len(correction_db)
        assert meta["n_bins"] == len(range_m)
        assert meta["n_samples_total"] > 0

        table = RangeCorrectionTable()
        table.add_curve(1_495_000, range_m, correction_db, meta)
        corrected = apply_empirical_range_correction(ds, table=table)

        from sidescanpy.processing.profiles import backscatter_vs_range

        before = backscatter_vs_range(
            ds, "backscatter_intensity", bin_width_m=2.0, exclude_below_db=None
        )
        after = backscatter_vs_range(
            corrected, "backscatter_intensity", bin_width_m=2.0, exclude_below_db=None
        )

        before_spread = before["mean"].max() - before["mean"].min()
        after_spread = after["mean"].max() - after["mean"].min()
        assert after_spread < before_spread * 0.1

    def test_raises_when_nothing_survives_filtering(self):
        ds = _make_radiometric_geometric_dataset(n_pings=2, n_samples=50, altitude_m=5.0)
        # altitude_m=5.0 with only 50 samples means every sample is inside
        # the water column (see compute_geometric_ranges NaN masking) —
        # ground_range_m is entirely NaN, nothing survives binning.
        with pytest.raises(ValueError, match="No samples survived"):
            fit_range_correction(ds)


class TestRangeCorrectionTable:
    def test_correction_db_at_interpolates(self):
        table = RangeCorrectionTable()
        table.add_curve(1_495_000, [10.0, 20.0, 30.0], [0.0, 5.0, 0.0])
        result = table.correction_db_at(1_495_000, [10.0, 15.0, 20.0])
        np.testing.assert_allclose(result, [0.0, 2.5, 5.0])

    def test_correction_db_at_clamps_outside_range(self):
        table = RangeCorrectionTable()
        table.add_curve(1_495_000, [10.0, 20.0, 30.0], [1.0, 5.0, 2.0])
        result = table.correction_db_at(1_495_000, [0.0, 1000.0])
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_unmatched_frequency_returns_none(self):
        table = RangeCorrectionTable()
        table.add_curve(1_495_000, [10.0, 20.0], [1.0, 2.0])
        assert table.correction_db_at(547_500, [10.0]) is None
        assert table.matched_frequency(547_500) is None

    def test_frequency_match_within_tolerance(self):
        table = RangeCorrectionTable()
        table.add_curve(1_495_000, [10.0, 20.0], [1.0, 2.0])
        # Within the 1000 Hz default tolerance (e.g. float round-tripping)
        assert table.matched_frequency(1_495_000.4) == 1_495_000.0

    def test_save_load_roundtrip(self, tmp_path):
        table = RangeCorrectionTable()
        table.add_curve(
            1_495_000, [10.0, 20.0, 30.0], [1.0, 2.0, 3.0], fit_metadata={"note": "test"}
        )
        path = tmp_path / "table.json"
        table.save(str(path))

        loaded = RangeCorrectionTable.load(str(path))
        assert loaded.frequencies() == [1_495_000.0]
        np.testing.assert_allclose(
            loaded.correction_db_at(1_495_000, [10.0, 20.0, 30.0]), [1.0, 2.0, 3.0]
        )
        assert loaded.metadata_for(1_495_000) == {"note": "test"}


class TestApplyEmpiricalRangeCorrection:
    def _table_for_freq(self, freq_hz=1_495_000):
        table = RangeCorrectionTable()
        table.add_curve(freq_hz, [0.0, 100.0], [3.0, 3.0])  # flat +3dB shift
        return table

    def test_floor_clipped_samples_stay_at_floor(self):
        from sidescanpy.core.units import FLOOR_DB, mask_floor_clip
        from sidescanpy.processing import uncorrected_backscatter

        ds = make_minimal_sidescan_dataset(n_pings=5, n_samples=300, freq_hz_start=1_495_000, backscatter_val=1.0)
        ds = ds.assign(altitude_m=(["ping"], np.full(5, 0.05, dtype=np.float32)))
        ds["backscatter_intensity"].values[:, :, 100:110] = 0.0
        ds = compute_geometric_ranges(apply_tvg(ds))
        out = apply_empirical_range_correction(ds, self._table_for_freq())
        v = out["backscatter_intensity"].values
        assert np.all(v[:, :, 100:110] == FLOOR_DB)
        assert np.isnan(mask_floor_clip(v)[:, :, 100:110]).all()
        assert np.all(uncorrected_backscatter(out).values[:, :, 100:110] == 0.0)

    def test_requires_tvg(self):
        ds = compute_geometric_ranges(make_minimal_sidescan_dataset())  # still "raw"
        with pytest.raises(ValueError, match="TVG-corrected"):
            apply_empirical_range_correction(ds, table=self._table_for_freq())

    def test_requires_ground_range_m(self):
        ds = apply_tvg(make_minimal_sidescan_dataset())
        with pytest.raises(ValueError, match="ground_range_m"):
            apply_empirical_range_correction(ds, table=self._table_for_freq())

    def test_applies_flat_shift(self):
        ds = _make_radiometric_geometric_dataset(n_pings=2, n_samples=3000)
        before = ds["backscatter_intensity"].values.copy()
        result = apply_empirical_range_correction(ds, table=self._table_for_freq())

        valid = np.isfinite(ds["ground_range_m"].values)
        np.testing.assert_allclose(
            result["backscatter_intensity"].values[valid],
            before[valid] + 3.0,
            atol=1e-4,
        )

    def test_water_column_samples_get_zero_correction_not_nan(self):
        """Samples with NaN ground_range_m (masked water column) must not
        have NaN introduced into a previously-finite backscatter_intensity."""
        ds = _make_radiometric_geometric_dataset(n_pings=2, n_samples=3000, altitude_m=0.05)
        invalid = ~np.isfinite(ds["ground_range_m"].values)
        assert invalid.any(), "fixture should have some water-column samples"

        before = ds["backscatter_intensity"].values.copy()
        result = apply_empirical_range_correction(ds, table=self._table_for_freq())

        assert not np.isnan(result["backscatter_intensity"].values[invalid]).any()
        np.testing.assert_allclose(
            result["backscatter_intensity"].values[invalid],
            before[invalid],
            atol=1e-4,
        )

    def test_unrecognized_frequency_skips_unchanged(self, caplog):
        ds = _make_radiometric_geometric_dataset(
            n_pings=2, n_samples=3000, freq_hz_start=547_500
        )
        table = self._table_for_freq(freq_hz=1_495_000)  # only calibrated for 1495kHz

        with caplog.at_level("WARNING"):
            result = apply_empirical_range_correction(ds, table=table)

        np.testing.assert_array_equal(
            result["backscatter_intensity"].values, ds["backscatter_intensity"].values
        )
        history = json.loads(result.attrs["processing_history"])
        assert not any(h.get("step") == STEP_NAME for h in history)
        assert "No calibrated range-correction curve" in caplog.text

    def test_double_application_raises(self):
        ds = _make_radiometric_geometric_dataset(n_pings=2, n_samples=3000)
        once = apply_empirical_range_correction(ds, table=self._table_for_freq())
        with pytest.raises(ValueError, match="already applied"):
            apply_empirical_range_correction(once, table=self._table_for_freq())

    def test_appends_processing_history(self):
        ds = _make_radiometric_geometric_dataset(n_pings=2, n_samples=3000)
        result = apply_empirical_range_correction(ds, table=self._table_for_freq())
        history = json.loads(result.attrs["processing_history"])
        matching = [h for h in history if h.get("step") == STEP_NAME]
        assert len(matching) == 1
        assert matching[0]["function"] == (
            "sidescanpy.processing.range_correction.apply_empirical_range_correction"
        )

    def test_does_not_mutate_input(self):
        ds = _make_radiometric_geometric_dataset(n_pings=2, n_samples=3000)
        original = ds["backscatter_intensity"].values.copy()
        apply_empirical_range_correction(ds, table=self._table_for_freq())
        np.testing.assert_array_equal(ds["backscatter_intensity"].values, original)

    def test_table_is_required(self):
        ds = _make_radiometric_geometric_dataset(n_pings=2, n_samples=3000)
        with pytest.raises(TypeError, match="requires a `table`"):
            apply_empirical_range_correction(ds)
        with pytest.raises(TypeError, match="must be a RangeCorrectionTable or a path"):
            apply_empirical_range_correction(ds, table=42)

    def test_table_accepted_as_path(self, tmp_path):
        ds = _make_radiometric_geometric_dataset(n_pings=2, n_samples=3000)
        path = tmp_path / "table.json"
        self._table_for_freq().save(str(path))
        via_path = apply_empirical_range_correction(ds, table=path)
        direct = apply_empirical_range_correction(ds, table=self._table_for_freq())
        np.testing.assert_array_equal(
            via_path["backscatter_intensity"].values, direct["backscatter_intensity"].values
        )

    def test_library_ships_no_default_table(self):
        assert not hasattr(RangeCorrectionTable, "load_default")
        with pytest.raises(ImportError):
            __import__("sidescanpy.processing.calibration")
