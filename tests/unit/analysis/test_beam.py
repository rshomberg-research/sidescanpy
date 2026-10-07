"""
Unit tests for sidescanpy.analysis.beam.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.analysis.beam import estimate_beam_geometry
from sidescanpy.analysis.nadir import NadirResult, analyze_nadir
from sidescanpy.analysis.usable_range import UsableRangeResult
from sidescanpy.processing.geometric import compute_geometric_ranges
from tests.unit.synthetic import make_raw_dataset as _make_raw_dataset


class TestEstimateBeamGeometry:
    def test_near_and_far_ray_match_known_geometry(self):
        """Isolates stage2_5's own angle math from
        detect_first_bottom_return's inherent smoothing imprecision
        (covered separately in TestDetectFirstBottomReturn) by feeding
        a hand-built boundary array with an exact, known sample,
        rather than relying on detection to find it. altitude=10.0
        (fixture default); sample 27 with sound_vel=1500,
        sample_rate=1000 -> slant_range = 27*0.75 = 20.25m ->
        near_angle_from_vertical = arccos(10/20.25) ~= 60.4deg.
        usable_range_m=50.0 -> far_angle_from_vertical = arctan(50/10)
        ~= 78.7deg."""
        n_pings, n_samples = 20, 100
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        fake_boundary = np.full((n_pings, 2), 27.0)
        fake_stage2 = NadirResult(first_bottom_return_sample=fake_boundary)
        fake_stage3 = UsableRangeResult(usable_range_m=50.0)
        result = estimate_beam_geometry(ds, fake_stage2.first_bottom_return_sample, fake_stage3.usable_range_m)

        for entry in result.near_ray.values():
            assert entry["observed_sample"] == 27
            assert entry["angle_from_vertical_deg"] == pytest.approx(60.4, abs=0.5)
        for label, entry in result.far_ray.items():
            assert entry["angle_from_vertical_deg"] == pytest.approx(78.7, abs=0.5)
            expected_sign = -1.0 if label == "port" else 1.0
            assert entry["predicted_across_track_m"] == pytest.approx(expected_sign * 50.0)
            # slant range is a true distance (hypotenuse), never signed
            assert entry["predicted_slant_range_m"] == pytest.approx(np.hypot(50.0, 10.0))
        assert result.beam_width_deg == pytest.approx(78.7 - 60.4, abs=1.0)

    def test_far_ray_angle_at_45_degrees_when_usable_range_equals_altitude(self):
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = analyze_nadir(ds, nadir_n_samples=5)
        stage3 = UsableRangeResult(usable_range_m=10.0)  # == altitude -> 45deg

        result = estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)

        for label, entry in result.far_ray.items():
            assert entry["angle_from_vertical_deg"] == pytest.approx(45.0, abs=1e-6)
            expected_sign = -1.0 if label == "port" else 1.0
            assert entry["predicted_across_track_m"] == pytest.approx(expected_sign * 10.0)

    def test_far_ray_port_and_starboard_are_oppositely_signed(self):
        """Regression test: usable_range_m is a single survey-aggregate
        magnitude, not per-channel -- an earlier version stored it
        unsigned for both "port" and "starboard" far_ray entries, which
        meant the beam-geometry diagram only ever drew a far ray toward
        the starboard side (both entries carried the same positive
        value), never port. across_track_m's convention (port negative,
        starboard positive) must be applied here the same way it's
        already inherent in near_ray's real per-channel values."""
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = analyze_nadir(ds, nadir_n_samples=5)
        stage3 = UsableRangeResult(usable_range_m=25.0)

        result = estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)

        assert result.far_ray["port"]["predicted_across_track_m"] == pytest.approx(-25.0)
        assert result.far_ray["starboard"]["predicted_across_track_m"] == pytest.approx(25.0)

    def test_far_ray_none_when_usable_range_none(self):
        """A real, documented outcome on some hardware (see
        estimate_usable_range's docstring) -- the far edge, beam width,
        and center angle should come back None with an explanatory
        note, not raise or silently omit."""
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = analyze_nadir(ds, nadir_n_samples=5)
        stage3 = UsableRangeResult(usable_range_m=None)

        result = estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)

        assert result.far_angle_deg is None
        assert result.beam_width_deg is None
        assert result.beam_center_angle_deg is None
        assert any("usable_range_m is None" in n for n in result.notes)

    def test_missing_required_var_raises(self):
        ds = _make_raw_dataset()  # no geometric ranges computed
        stage2 = analyze_nadir(ds, nadir_n_samples=5)
        stage3 = UsableRangeResult()
        with pytest.raises(ValueError, match="across_track_m"):
            estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)

    def test_impossible_geometry_flagged(self):
        """A boundary sample whose slant range is less than altitude
        (fixture default 10.0; sample 5 -> slant_range=3.75m) is
        geometrically impossible for any real beam angle -- should be
        flagged, not silently produce a bogus angle."""
        n_pings, n_samples = 20, 100
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        fake_boundary = np.full((n_pings, 2), 5.0)
        fake_stage2 = NadirResult(first_bottom_return_sample=fake_boundary)
        fake_stage3 = UsableRangeResult()
        result = estimate_beam_geometry(ds, fake_stage2.first_bottom_return_sample, fake_stage3.usable_range_m)

        for entry in result.near_ray.values():
            assert entry["angle_from_vertical_deg"] is None
        assert any("geometrically" in n for n in result.notes)

    @staticmethod
    def _make_angle_ds(angle_deg_per_ping: np.ndarray, *, altitude: float = 10.0) -> tuple[xr.Dataset, NadirResult, UsableRangeResult]:
        """Directly construct slant_range_m so the near-edge angle at
        boundary sample 0 is exactly `angle_deg_per_ping[p]` for every
        ping/channel -- bypasses compute_geometric_ranges' coarse
        (0.75m/sample) discretization, which can't hit small target
        angles at an integer sample boundary. Returns a stage3 with no
        usable_range_m -- these tests only exercise the near-edge
        classification, which doesn't depend on the far edge."""
        n_pings = angle_deg_per_ping.shape[0]
        n_channels, n_samples = 2, 5
        slant_at_zero = altitude / np.cos(np.deg2rad(angle_deg_per_ping))
        slant = np.ones((n_pings, n_channels, n_samples), dtype=np.float64) * 1000.0
        slant[:, :, 0] = slant_at_zero[:, None]
        across = np.zeros((n_pings, n_channels, n_samples), dtype=np.float64)
        ds = xr.Dataset(
            {
                "slant_range_m": (("ping", "channel", "sample"), slant),
                "across_track_m": (("ping", "channel", "sample"), across),
                "altitude_m": (("ping",), np.full(n_pings, altitude)),
            }
        )
        boundary = np.zeros((n_pings, n_channels))
        stage2 = NadirResult(first_bottom_return_sample=boundary)
        stage3 = UsableRangeResult()
        return ds, stage2, stage3

    def test_classification_determined(self):
        angles = np.full(40, 30.0)  # consistent, clearly nonzero tilt
        ds, stage2, stage3 = self._make_angle_ds(angles)

        result = estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)

        assert result.confidence == "determined"
        assert result.near_angle_deg_median == pytest.approx(30.0, abs=0.1)
        assert result.near_angle_valid_fraction == pytest.approx(1.0)

    def test_classification_determined_even_at_near_zero_angle(self):
        """An earlier version of this stage treated a near-zero implied
        angle as a different physical category ("near-grazing weak
        backscatter" vs. "water column") -- that distinction was wrong:
        the near-nadir span is never illuminated regardless of how
        small the beam's near angle is. classification now reports
        measurement confidence only, not a choice between two physical
        stories -- a small but consistent angle is still `determined`."""
        angles = np.full(40, 2.0)
        ds, stage2, stage3 = self._make_angle_ds(angles)

        result = estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)

        assert result.confidence == "determined"
        assert result.near_angle_deg_median == pytest.approx(2.0, abs=0.1)

    def test_classification_ambiguous_when_inconsistent(self):
        rng = np.random.default_rng(11)
        angles = rng.uniform(5.0, 85.0, size=60)  # wide spread -> high std
        ds, stage2, stage3 = self._make_angle_ds(angles)

        result = estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)

        assert result.confidence == "ambiguous"
        assert result.near_angle_deg_std > 15.0

    def test_classification_undetermined_when_mostly_invalid(self):
        n_pings, n_channels, n_samples = 20, 2, 5
        # slant_range_m < altitude everywhere -- every entry geometrically impossible.
        ds = xr.Dataset(
            {
                "slant_range_m": (("ping", "channel", "sample"), np.full((n_pings, n_channels, n_samples), 1.0)),
                "across_track_m": (("ping", "channel", "sample"), np.zeros((n_pings, n_channels, n_samples))),
                "altitude_m": (("ping",), np.full(n_pings, 10.0)),
            }
        )
        boundary = np.zeros((n_pings, n_channels))
        stage2 = NadirResult(first_bottom_return_sample=boundary)
        stage3 = UsableRangeResult()

        result = estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)

        assert result.confidence == "undetermined"
        assert result.near_angle_valid_fraction == pytest.approx(0.0)
