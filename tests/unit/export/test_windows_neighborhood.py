"""
Unit tests for sidescanpy.export.windows.neighborhood.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.target import Target
from sidescanpy.export.windows.levels import nominal_sample_interval_m
from sidescanpy.export.windows.neighborhood import (
    SWATH_OVERSAMPLE,
    across_values,
    geo_decimation,
    load_geo_neighborhood,
    load_swath_neighborhood,
    neighborhood_ping_spacing_m,
)
from sidescanpy.io.hdf5 import read_survey


class TestDecimation:
    """Geometric/georef neighborhoods are block-averaged before resampling."""

    def test_factors_keep_two_cells_per_output_pixel(self, georef_ss_h5_dir):
        ds = read_survey(next(georef_ss_h5_dir.glob("*.ss.h5"))).get_frequency("hz_1495000").dataset
        mps = nominal_sample_interval_m(ds)
        half_span, fp, fs = geo_decimation(ds, 10, "dual-swath", size_px=64, jitter_px=0, resolution_m=0.1)
        assert fs == int(0.1 / (SWATH_OVERSAMPLE * mps))
        assert fs > 1
        assert fp >= 1

    def test_neighborhood_covers_the_window_in_meters(self, georef_ss_h5_dir):
        """Sized from size_px * resolution_m along-track, not from
        size_px pings -- a pings-sized neighborhood left real geometric
        chips short of data at top/bottom (griddata nearest-fill streaks)."""
        ds = read_survey(next(georef_ss_h5_dir.glob("*.ss.h5"))).get_frequency("hz_1495000").dataset
        spacing = neighborhood_ping_spacing_m(ds, 0, ds.sizes["ping"])
        half_span, _fp, _fs = geo_decimation(ds, 10, "chip", size_px=100, jitter_px=10, resolution_m=spacing)
        assert half_span * spacing >= (100 / 2 + 10) * spacing

    def test_loaded_neighborhood_is_averaged_and_rescaled(self, georef_ss_h5_dir):
        ds = read_survey(next(georef_ss_h5_dir.glob("*.ss.h5"))).get_frequency("hz_1495000").dataset
        out = load_swath_neighborhood(ds, 0, ds.sizes["ping"], 1, 10, with_db=True)
        assert out.sizes["sample"] == ds.sizes["sample"] // 10
        assert out.sizes["ping"] == ds.sizes["ping"]
        assert "backscatter_db" in out
        assert "slant_range_m" not in out
        assert nominal_sample_interval_m(out) == pytest.approx(10 * nominal_sample_interval_m(ds))


class TestAcrossAxis:
    def _target(self, n_pings=300):
        lat = float(np.linspace(40.0000, 40.0010, n_pings)[150])
        lon = float(np.linspace(-70.0000, -70.0010, n_pings)[150])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=15,
            position_confidence="verified",
        )

    def test_slant_across_axis_neighborhood_uses_signed_slant_range(self, tmp_path, swath_ss_h5_dir):
        """--across-axis slant swaps the loaded neighborhood's across_track_m
        for +/-slant_range_m (port negative), which every downstream
        geometric extraction/co-target lookup reads -- and keeps the
        water-column samples ground projection drops."""
        survey = read_survey(next(swath_ss_h5_dir.glob("*.ss.h5")))
        ds_full = survey.get_frequency("hz_1495000").dataset
        t = self._target()
        views = {
            axis: load_geo_neighborhood(
                ds_full, t, 150, {1: [(t, 150)]}, ["chip"], size_px=64, jitter_px=0,
                resolution_m=0.002, with_db=False, across_axis=axis,
            )
            for axis in ("ground", "slant")
        }
        slant = views["slant"].ds
        np.testing.assert_allclose(slant["across_track_m"].isel(channel=1).values, slant["slant_range_m"].isel(channel=1).values)
        np.testing.assert_allclose(slant["across_track_m"].isel(channel=0).values, -slant["slant_range_m"].isel(channel=0).values)
        ground_across = views["ground"].ds["across_track_m"].values
        assert np.isnan(ground_across).any() and np.isfinite(slant["across_track_m"].values).all()
        finite = np.isfinite(ground_across)
        assert (np.abs(slant["across_track_m"].values[finite]) >= np.abs(ground_across[finite]) - 1e-9).all()

    def test_across_values_slant_signs_port_negative(self):
        ds = xr.Dataset({"slant_range_m": (("ping", "channel", "sample"), np.ones((2, 2, 3)))})
        v = across_values(ds, "slant").values
        assert (v[:, 0] < 0).all() and (v[:, 1] > 0).all()
        with pytest.raises(ValueError):
            across_values(ds, "sideways")
