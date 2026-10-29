"""
Unit tests for sidescanpy.processing.layback (the ``"layback"`` step).

Covers the offset direction and size, the position-source rule (ship
moves; towfish / mixed / unknown never do), the cable-out fallback,
tow-point offsets, the history record, inversion and use through
``correct``.

2026 rshomberg@uri.edu
"""

import json
import logging

import numpy as np
import pyproj
import pytest

from sidescanpy.core.history import has_step, processing_level
from sidescanpy.processing import correct, invert
from sidescanpy.processing.layback import apply_layback
from tests.unit.synthetic import make_raw_dataset

GEOD = pyproj.Geod(ellps="WGS84")


def _ds(*, heading=90.0, layback=100.0, source="ship", cable=None, depth=None, n_pings=6):
    ds = make_raw_dataset(n_pings=n_pings, n_samples=50)
    ds["latitude"] = ds["latitude"].astype(np.float64)
    ds["longitude"] = ds["longitude"].astype(np.float64)
    ds["heading_deg"] = ("ping", np.full(n_pings, heading, dtype=np.float32))
    ds["layback_m"] = ("ping", np.broadcast_to(np.asarray(layback, dtype=np.float32), (n_pings,)).copy())
    if cable is not None:
        ds["cable_out_m"] = ("ping", np.full(n_pings, cable, dtype=np.float32))
    if depth is not None:
        ds["depth_m"] = ("ping", np.full(n_pings, depth, dtype=np.float32))
    if source is not None:
        ds.attrs["position_source"] = source
        ds.attrs["position_source_reason"] = "test"
    return ds


def _offset(before, after):
    """(distance m, azimuth deg) from the recorded to the moved position, per ping."""
    az, _, dist = GEOD.inv(
        before["longitude"].values, before["latitude"].values,
        after["longitude"].values, after["latitude"].values,
    )
    return dist, np.mod(az, 360.0)


@pytest.mark.unit
class TestOffset:
    @pytest.mark.parametrize("heading, astern", [(90.0, 270.0), (0.0, 180.0), (225.0, 45.0)])
    def test_moves_astern_by_layback_when_ship(self, heading, astern):
        ds = _ds(heading=heading, layback=100.0)
        out = apply_layback(ds)
        dist, az = _offset(ds, out)
        np.testing.assert_allclose(dist, 100.0, atol=1e-6)
        np.testing.assert_allclose(az, astern, atol=1e-6)

    def test_tow_point_offset_adds_aft_and_starboard(self):
        ds = _ds(heading=0.0, layback=50.0)
        ds.attrs["tow_point_aft_m"] = 10.0
        ds.attrs["tow_point_starboard_m"] = 3.0
        out = apply_layback(ds)
        # Heading north: 60 m south and 3 m east
        dist, _ = _offset(ds, out)
        np.testing.assert_allclose(dist, np.hypot(60.0, 3.0), atol=1e-3)
        assert np.all(out["longitude"].values > ds["longitude"].values)
        assert np.all(out["latitude"].values < ds["latitude"].values)

    def test_scalar_layback_overrides_variable(self):
        ds = _ds(layback=100.0)
        dist, _ = _offset(ds, apply_layback(ds, layback_m=25.0))
        np.testing.assert_allclose(dist, 25.0, atol=1e-6)

    def test_cable_out_fallback_when_layback_nan(self):
        ds = _ds(layback=np.nan, cable=50.0, depth=30.0)
        out = apply_layback(ds)
        dist, _ = _offset(ds, out)
        np.testing.assert_allclose(dist, 40.0, atol=1e-6)
        rec = json.loads(out.attrs["processing_history"])[-1]
        assert rec["params"]["n_pings_layback_from_cable_out"] == 6
        assert rec["params"]["n_pings_layback_recorded"] == 0

    def test_ping_without_layback_or_heading_not_moved(self, caplog):
        ds = _ds(layback=np.array([100.0, np.nan, 100.0, 100.0, 100.0, 100.0]))
        ds["heading_deg"].values[3] = np.nan
        with caplog.at_level(logging.WARNING, logger="sidescanpy.processing.layback"):
            out = apply_layback(ds, use_cable_out=False)
        assert "2 of 6 pings" in caplog.text
        dist, _ = _offset(ds, out)
        np.testing.assert_allclose(dist[[1, 3]], 0.0, atol=1e-9)
        np.testing.assert_allclose(dist[[0, 2, 4, 5]], 100.0, atol=1e-6)


@pytest.mark.unit
class TestNoDoubleApply:
    @pytest.mark.parametrize("source", ["towfish", "mixed", "unknown", None])
    def test_unchanged_with_warning_when_not_ship(self, source, caplog):
        ds = _ds(source=source)
        with caplog.at_level(logging.WARNING, logger="sidescanpy.processing.layback"):
            out = apply_layback(ds)
        assert out is ds
        assert "layback not applied" in caplog.text
        assert not has_step(out, "layback")

    def test_caller_can_assert_ship(self):
        ds = _ds(source="unknown")
        out = apply_layback(ds, position_source="ship")
        assert out.attrs["position_source"] == "towfish"
        dist, _ = _offset(ds, out)
        np.testing.assert_allclose(dist, 100.0, atol=1e-6)

    def test_second_application_raises(self):
        out = apply_layback(_ds())
        with pytest.raises(ValueError, match="already applied"):
            apply_layback(out, position_source="ship")

    def test_raises_when_georef_already_applied(self):
        ds = correct(_ds(), ["geometric", "georef"])
        with pytest.raises(ValueError, match="invert georef first"):
            apply_layback(ds)

    def test_unknown_position_source_rejected(self):
        with pytest.raises(ValueError, match="not one of"):
            apply_layback(_ds(), position_source="antenna")


@pytest.mark.unit
class TestRecordAndInversion:
    def test_history_record_and_kept_positions(self):
        ds = _ds()
        out = apply_layback(ds)
        rec = json.loads(out.attrs["processing_history"])[-1]
        assert rec["step"] == "layback"
        assert rec["params"]["position_source"] == "ship"
        assert rec["params"]["model"] == "fixed_offset_astern"
        assert rec["params"]["layback_m_median"] == pytest.approx(100.0)
        np.testing.assert_array_equal(out["latitude_before_layback"].values, ds["latitude"].values)
        assert processing_level(out) == "raw"

    def test_invert_restores_positions_and_source(self):
        ds = _ds()
        back = invert(apply_layback(ds), ["layback"])
        np.testing.assert_array_equal(back["latitude"].values, ds["latitude"].values)
        np.testing.assert_array_equal(back["longitude"].values, ds["longitude"].values)
        assert "latitude_before_layback" not in back
        assert back.attrs["position_source"] == "ship"
        assert back.attrs["position_source_reason"] == "test"
        assert not has_step(back, "layback")

    def test_invert_layback_also_drops_georef(self):
        ds = correct(_ds(), ["layback", "geometric", "georef"])
        back = invert(ds, ["layback"])
        assert "latitude_per_sample" not in back
        assert "across_track_m" in back

    def test_invert_all_includes_layback(self):
        ds = _ds()
        back = invert(correct(ds, ["layback", "geometric"]))
        np.testing.assert_array_equal(back["latitude"].values, ds["latitude"].values)


@pytest.mark.unit
class TestThroughCorrect:
    def test_georef_uses_laid_back_positions(self):
        ds = _ds(heading=0.0, layback=100.0)
        plain = correct(ds, ["geometric", "georef"])
        laid = correct(ds, ["layback", "geometric", "georef"])
        _, _, dist = GEOD.inv(
            plain["longitude_per_sample"].values, plain["latitude_per_sample"].values,
            laid["longitude_per_sample"].values, laid["latitude_per_sample"].values,
        )
        finite = np.isfinite(dist)
        np.testing.assert_allclose(dist[finite], 100.0, atol=0.05)

    def test_idempotent_without_force(self):
        once = correct(_ds(), ["layback"])
        twice = correct(once, ["layback"])
        np.testing.assert_array_equal(once["latitude"].values, twice["latitude"].values)

    def test_force_reapplies_from_recorded_positions(self):
        ds = _ds()
        once = correct(ds, ["layback"])
        again = correct(once, ["layback"], {"layback": {"layback_m": 30.0}}, force=True)
        dist, _ = _offset(ds, again)
        np.testing.assert_allclose(dist, 30.0, atol=1e-6)
        steps = [r["step"] for r in json.loads(again.attrs["processing_history"])]
        assert steps.count("layback") == 1

    def test_force_without_georef_in_steps_raises_when_georef_applied(self):
        ds = correct(_ds(), ["layback", "geometric", "georef"])
        with pytest.raises(ValueError, match="also list 'georef'"):
            correct(ds, ["layback"], force=True)
