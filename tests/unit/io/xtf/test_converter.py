"""
Unit tests for sidescanpy.io.xtf.converter on synthetic XTF files.

Files are built with the shared builders in ``conftest.py`` and read with
the real reader, so these tests cover read + convert: two frequencies,
pings missing the high-frequency pair, NaN padding, sound-velocity
halving detection, sample rate and range, a single-frequency file,
multi-file merging, navigation units and the ``.ss.h5`` round trip.

2026 rshomberg@uri.edu
"""

import json
import logging

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.io.xtf import convert, read_xtf
from sidescanpy.processing.sample_axis import sample_interval_m
from tests.unit.io.xtf.conftest import (
    PORT,
    STBD,
    chan_info,
    dual_frequency_infos,
    file_header,
    notes_packet,
    ramp,
    sonar_packet,
    write_xtf,
)

N_SAMP = 64


def _ping(i, n=N_SAMP, chans=(0, 1, 2, 3), **kw):
    return sonar_packet([(c, ramp(n, np.uint16, i * 4 + c)) for c in chans], ping_number=i, **kw)


def _dual(tmp_path, pings, name="dual.xtf", **header_fields):
    hdr = file_header(dual_frequency_infos(), **header_fields)
    return write_xtf(tmp_path / name, hdr, [notes_packet()] + pings)


@pytest.fixture
def dual_survey(tmp_path):
    path = _dual(tmp_path, [_ping(i) for i in range(6)])
    return path, convert(read_xtf(path))


# ============================================================
# STRUCTURE
# ============================================================

@pytest.mark.unit
class TestDualFrequency:
    def test_one_channel_per_frequency_in_hz(self, dual_survey):
        _, survey = dual_survey
        assert isinstance(survey, SideScanSurvey)
        assert survey.frequencies == ["hz_100000", "hz_400000"]
        for fk, f in (("hz_100000", 100_000), ("hz_400000", 400_000)):
            ds = survey.get_frequency(fk).dataset
            assert ds.attrs["freq_hz_start"] == f and ds.attrs["freq_hz_end"] == f
            ds.sidescan.validate()

    def test_survey_metadata(self, dual_survey):
        path, survey = dual_survey
        assert survey.source_format == "xtf"
        assert survey.source_files == [path]
        assert survey.survey_name == "dual"

    def test_backscatter_layout_port_then_starboard(self, dual_survey):
        _, survey = dual_survey
        for fk, (port, stbd) in (("hz_100000", (0, 1)), ("hz_400000", (2, 3))):
            ds = survey.get_frequency(fk).dataset
            bs = ds["backscatter_intensity"]
            assert bs.dims == ("ping", "channel", "sample")
            assert bs.dtype == np.float32
            assert bs.shape == (6, 2, N_SAMP)
            assert list(ds["channel_id"].values) == ["port", "starboard"]
            for i in range(6):
                # XTF stores port far -> near; the converter puts the near sample first.
                np.testing.assert_array_equal(bs.values[i, 0], ramp(N_SAMP, np.uint16, i * 4 + port)[::-1])
                np.testing.assert_array_equal(bs.values[i, 1], ramp(N_SAMP, np.uint16, i * 4 + stbd))
            np.testing.assert_array_equal(ds["sample"].values, np.arange(N_SAMP))
            assert (ds["valid_samples"].values == N_SAMP).all()

    def test_per_ping_fields(self, dual_survey):
        path, survey = dual_survey
        data = read_xtf(path)
        h = data.ping_headers
        ds = survey.get_frequency("hz_100000").dataset
        np.testing.assert_array_equal(ds["latitude"].values, h["SensorYcoordinate"])
        np.testing.assert_array_equal(ds["longitude"].values, h["SensorXcoordinate"])
        assert ds["latitude"].dtype == np.float64
        np.testing.assert_array_equal(ds["heading_deg"].values, h["SensorHeading"])
        np.testing.assert_array_equal(ds["depth_m"].values, h["SensorDepth"])
        np.testing.assert_array_equal(ds["altitude_m"].values, h["SensorPrimaryAltitude"])
        np.testing.assert_array_equal(ds["fathometer_reading_m"].values, ds["altitude_m"].values)
        np.testing.assert_array_equal(ds["time"].values, data.times)
        np.testing.assert_array_equal(ds["ping_number"].values, np.arange(6))

    def test_aux_records(self, dual_survey):
        path, survey = dual_survey
        h = read_xtf(path).ping_headers
        nav = survey.navigation
        assert nav.attrs["source"] == "extracted_from_xtf_ping_headers"
        np.testing.assert_array_equal(nav["latitude"].values, h["SensorYcoordinate"])
        np.testing.assert_array_equal(nav["longitude"].values, h["SensorXcoordinate"])
        assert "speed_knots" in nav and "heading_deg" in nav
        assert survey.attitude.sizes["obs"] == 6
        assert survey.has_dvl is False

    def test_provenance_attrs(self, dual_survey):
        _, survey = dual_survey
        ds = survey.get_frequency("hz_400000").dataset
        assert ds.attrs["xtf_recording_program"] == "synth"
        assert ds.attrs["xtf_sonar_name"] == "synthetic"
        assert ds.attrs["xtf_channel_numbers"] == "2,3"
        assert ds.attrs["xtf_channel_names"] == "PORT_HI,STBD_HI"
        assert ds.attrs["xtf_packets_non_sonar"] == 1
        assert ds.attrs["xtf_packets_fewer_channels"] == 0
        assert ds.attrs["xtf_pings_absent"] == 0
        assert ds.attrs["pulse_length_us"] == 0.0
        assert json.loads(ds.attrs["processing_history"]) == []


# ============================================================
# ABSENT CHANNELS AND PADDING
# ============================================================

@pytest.mark.unit
class TestAbsentAndPadding:
    def test_pings_without_a_frequency_are_dropped_from_it(self, tmp_path):
        pk = [_ping(i, chans=(0, 1), num_chans_to_follow=4) for i in range(3)]
        pk += [_ping(i) for i in range(3, 7)]
        survey = convert(read_xtf(_dual(tmp_path, pk)))
        lo = survey.get_frequency("hz_100000").dataset
        hi = survey.get_frequency("hz_400000").dataset
        assert lo.sizes["ping"] == 7 and hi.sizes["ping"] == 4
        np.testing.assert_array_equal(hi["ping_number"].values, [3, 4, 5, 6])
        np.testing.assert_array_equal(hi["time"].values, lo["time"].values[3:])
        assert not np.isnan(hi["backscatter_intensity"].values).any()
        assert hi.attrs["xtf_pings_absent"] == 3 and lo.attrs["xtf_pings_absent"] == 0
        assert hi.attrs["xtf_packets_fewer_channels"] == 3
        # Navigation keeps every ping.
        assert survey.navigation.sizes["obs"] == 7

    def test_one_side_absent_is_nan(self, tmp_path, caplog):
        pk = [_ping(0), _ping(1, chans=(0, 2, 3)), _ping(2)]
        with caplog.at_level(logging.WARNING):
            survey = convert(read_xtf(_dual(tmp_path, pk)))
        lo = survey.get_frequency("hz_100000").dataset
        assert lo.sizes["ping"] == 3
        assert np.isnan(lo["backscatter_intensity"].values[1, 1]).all()
        assert not np.isnan(lo["backscatter_intensity"].values[1, 0]).any()
        assert list(lo["valid_samples"].values[1]) == [N_SAMP, 0]
        assert any("only one side" in r.getMessage() for r in caplog.records)

    def test_short_pings_are_nan_padded(self, tmp_path):
        counts = [N_SAMP, 40, N_SAMP, 17]
        pk = [_ping(i, n=n) for i, n in enumerate(counts)]
        survey = convert(read_xtf(_dual(tmp_path, pk)))
        ds = survey.get_frequency("hz_100000").dataset
        bs = ds["backscatter_intensity"].values
        assert bs.shape == (4, 2, N_SAMP)
        for i, n in enumerate(counts):
            assert not np.isnan(bs[i, :, :n]).any()
            assert np.isnan(bs[i, :, n:]).all()
            np.testing.assert_array_equal(bs[i, 0, :n], ramp(n, np.uint16, i * 4)[::-1])
        np.testing.assert_array_equal(ds["valid_samples"].values[:, 0], counts)
        assert ds["valid_samples"].dtype == np.int32

    def test_single_frequency_file(self, tmp_path):
        infos = [chan_info(PORT, 120.0, name="PORT"), chan_info(STBD, 120.0, name="STBD")]
        pk = [_ping(i, chans=(0, 1)) for i in range(4)]
        path = write_xtf(tmp_path / "single.xtf", file_header(infos), pk)
        survey = convert(read_xtf(path))
        assert survey.frequencies == ["hz_120000"]
        assert survey.get_frequency("hz_120000").n_pings == 4

    def test_float_and_byte_samples(self, tmp_path):
        infos = [chan_info(PORT, 120.0, 4, 5), chan_info(STBD, 120.0, 4, 5)]
        vals = np.linspace(0.5, 9.5, N_SAMP).astype(np.float32)
        pk = [sonar_packet([(0, vals), (1, vals * 2)], ping_number=i) for i in range(2)]
        path = write_xtf(tmp_path / "f4.xtf", file_header(infos), pk)
        ds = convert(read_xtf(path)).get_frequency("hz_120000").dataset
        np.testing.assert_array_equal(ds["backscatter_intensity"].values[1, 1], vals * 2)

    def test_samples_scaled_by_weight_per_ping(self, tmp_path):
        # Weight changes from ping to ping, as in EdgeTech Discover files.
        weights = [5, 6, 0, 3]
        pk = [_ping(i, chan_fields={"Weight": w}) for i, w in enumerate(weights)]
        ds = convert(read_xtf(_dual(tmp_path, pk))).get_frequency("hz_100000").dataset
        bs = ds["backscatter_intensity"].values
        for i, w in enumerate(weights):
            expected = ramp(N_SAMP, np.uint16, i * 4 + 1).astype(np.float32) * 2.0 ** -w
            np.testing.assert_array_equal(bs[i, 1], expected)
        assert ds.attrs["xtf_weight_min"] == 0
        assert ds.attrs["xtf_weight_max"] == 6

    def test_implausible_weight_ignored_with_warning(self, tmp_path, caplog):
        """Some older Klein-written files carry garbage Weight (e.g. -8408);
        2**-Weight would overflow every sample. Such blocks stay unscaled."""
        weights = [-8408, 3, -4584, 0]
        pk = [_ping(i, chan_fields={"Weight": w}) for i, w in enumerate(weights)]
        with caplog.at_level("WARNING", logger="sidescanpy"):
            ds = convert(read_xtf(_dual(tmp_path, pk))).get_frequency("hz_100000").dataset
        bs = ds["backscatter_intensity"].values
        assert np.isfinite(bs[:, 1]).all()
        np.testing.assert_array_equal(bs[0, 1], ramp(N_SAMP, np.uint16, 1).astype(np.float32))
        np.testing.assert_array_equal(bs[1, 1], ramp(N_SAMP, np.uint16, 5).astype(np.float32) * 2.0 ** -3)
        assert ds.attrs["xtf_weight_ignored"] > 0
        assert ds.attrs["xtf_weight_min"] == -8408
        assert any("Weight" in r.getMessage() for r in caplog.records)

    def test_zero_weight_leaves_samples_unscaled(self, dual_survey):
        _, survey = dual_survey
        ds = survey.get_frequency("hz_100000").dataset
        np.testing.assert_array_equal(ds["backscatter_intensity"].values[0, 1], ramp(N_SAMP, np.uint16, 1))
        assert ds.attrs["xtf_weight_min"] == ds.attrs["xtf_weight_max"] == 0


# ============================================================
# SOUND VELOCITY, SAMPLE RATE, RANGE
# ============================================================

@pytest.mark.unit
class TestSoundVelocityAndRange:
    def test_halved_sound_velocity_is_doubled(self, dual_survey, caplog):
        path, survey = dual_survey
        ds = survey.get_frequency("hz_100000").dataset
        assert (ds["sound_vel_ms"].values == 1500.0).all()
        assert ds["sound_vel_ms"].dtype == np.float32
        assert ds.attrs["xtf_sound_velocity_doubled"] == 1
        with caplog.at_level(logging.DEBUG, logger="sidescanpy.io.xtf.converter"):
            convert(read_xtf(path))
        assert any("doubled" in r.getMessage() for r in caplog.records)

    def test_plausible_sound_velocity_is_kept(self, tmp_path, caplog):
        # SlantRange/TimeDuration = c/2 = 750 is consistent with 1500 m/s.
        pk = [_ping(i, SoundVelocity=1500.0) for i in range(3)]
        with caplog.at_level(logging.WARNING):
            ds = convert(read_xtf(_dual(tmp_path, pk))).get_frequency("hz_100000").dataset
        assert (ds["sound_vel_ms"].values == 1500.0).all()
        assert ds.attrs["xtf_sound_velocity_doubled"] == 0
        assert not caplog.records

    def test_sound_velocity_inconsistent_with_range_warns(self, tmp_path, caplog):
        pk = [_ping(i, SoundVelocity=1450.0, chan_fields={"SlantRange": 150.0}) for i in range(3)]
        with caplog.at_level(logging.WARNING):
            ds = convert(read_xtf(_dual(tmp_path, pk))).get_frequency("hz_100000").dataset
        assert (ds["sound_vel_ms"].values == 1450.0).all()
        assert any("disagrees" in r.getMessage() for r in caplog.records)

    def test_zero_sound_velocity_is_nan(self, tmp_path):
        pk = [_ping(i, SoundVelocity=0.0) for i in range(3)]
        ds = convert(read_xtf(_dual(tmp_path, pk))).get_frequency("hz_100000").dataset
        assert np.isnan(ds["sound_vel_ms"].values).all()

    def test_sample_rate_and_range_match_time_duration(self, dual_survey):
        _, survey = dual_survey
        ds = survey.get_frequency("hz_100000").dataset
        duration = 100.0 / 750.0  # conftest's TimeDuration
        assert ds.attrs["sample_rate_hz"] == pytest.approx(N_SAMP / np.float32(duration))
        max_range = sample_interval_m(ds) * N_SAMP
        np.testing.assert_allclose(max_range, np.float32(duration) * 1500.0 / 2, rtol=1e-6)
        np.testing.assert_allclose(max_range, 100.0, rtol=1e-6)

    def test_sample_rate_from_slant_range_without_duration(self, tmp_path):
        pk = [_ping(i, chan_fields={"TimeDuration": 0.0, "SlantRange": 150.0}) for i in range(3)]
        ds = convert(read_xtf(_dual(tmp_path, pk))).get_frequency("hz_100000").dataset
        assert ds.attrs["sample_rate_hz"] == pytest.approx(N_SAMP * 1500.0 / 300.0)

    def test_range_change_uses_modal_rate_and_warns(self, tmp_path, caplog):
        pk = [_ping(i) for i in range(5)]
        pk += [_ping(i, chan_fields={"SlantRange": 200.0, "TimeDuration": 200.0 / 750.0})
               for i in range(5, 7)]
        with caplog.at_level(logging.WARNING):
            ds = convert(read_xtf(_dual(tmp_path, pk))).get_frequency("hz_100000").dataset
        assert ds.attrs["sample_rate_hz"] == pytest.approx(N_SAMP * 750.0 / 100.0)
        assert ds.attrs["xtf_pings_other_sample_rate"] == 2
        assert any("sample rate changes" in r.getMessage() for r in caplog.records)

    def test_zero_altitude_is_nan(self, tmp_path):
        pk = [_ping(0), _ping(1, SensorPrimaryAltitude=0.0), _ping(2)]
        ds = convert(read_xtf(_dual(tmp_path, pk))).get_frequency("hz_100000").dataset
        assert np.isnan(ds["altitude_m"].values[1])
        assert np.isnan(ds["fathometer_reading_m"].values[1])
        assert ds["altitude_m"].values[0] == 8.5


# ============================================================
# NAVIGATION UNITS, FREQUENCY UNITS, ERRORS
# ============================================================

@pytest.mark.unit
class TestUnitsAndErrors:
    def test_projected_nav_units_without_nav_crs_raises(self, tmp_path):
        path = _dual(tmp_path, [_ping(0)], nav_units=0)
        with pytest.raises(ValueError, match="nav_crs"):
            convert(read_xtf(path))

    def test_undefined_nav_units_not_implemented(self, tmp_path):
        path = _dual(tmp_path, [_ping(0)], nav_units=1)
        with pytest.raises(NotImplementedError, match="NavUnits 1"):
            convert(read_xtf(path))

    def test_frequency_already_in_hz(self, tmp_path):
        infos = [chan_info(PORT, 455_000.0), chan_info(STBD, 455_000.0)]
        path = write_xtf(tmp_path / "hz.xtf", file_header(infos), [_ping(0, chans=(0, 1))])
        assert convert(read_xtf(path)).frequencies == ["hz_455000"]

    def test_no_pings_raises(self, tmp_path):
        path = write_xtf(tmp_path / "empty.xtf", file_header(dual_frequency_infos()),
                         [notes_packet()])
        with pytest.raises(ValueError, match="no sonar pings"):
            convert(read_xtf(path))

    def test_survey_name_and_source_files_override(self, dual_survey):
        path, _ = dual_survey
        survey = convert(read_xtf(path), source_files=[path], survey_name="named")
        assert survey.survey_name == "named"


# ============================================================
# MULTIPLE FILES
# ============================================================

@pytest.mark.unit
class TestMultiFile:
    def test_files_are_concatenated(self, tmp_path):
        a = _dual(tmp_path, [_ping(i) for i in range(3)], name="a.xtf")
        b = _dual(tmp_path, [_ping(i, n=48) for i in range(3, 6)], name="b.xtf")
        survey = convert([read_xtf(a), read_xtf(b)])
        assert survey.source_files == [a, b]
        assert survey.survey_name == "a"
        ds = survey.get_frequency("hz_100000").dataset
        np.testing.assert_array_equal(ds["ping_number"].values, np.arange(6))
        assert np.isnan(ds["backscatter_intensity"].values[3:, :, 48:]).all()
        assert ds.attrs["xtf_packets_non_sonar"] == 2

    def test_files_out_of_time_order_are_sorted(self, tmp_path):
        a = _dual(tmp_path, [_ping(i) for i in range(3)], name="a.xtf")
        b = _dual(tmp_path, [_ping(i) for i in range(3, 6)], name="b.xtf")
        survey = convert([read_xtf(b), read_xtf(a)])
        ds = survey.get_frequency("hz_400000").dataset
        np.testing.assert_array_equal(ds["ping_number"].values, np.arange(6))
        assert np.all(np.diff(ds["time"].values.astype("int64")) > 0)
        np.testing.assert_array_equal(
            ds["backscatter_intensity"].values[0, 0], ramp(N_SAMP, np.uint16, 2)[::-1]
        )
        assert np.all(np.diff(survey.navigation["time"].values.astype("int64")) > 0)


# ============================================================
# .ss.h5 ROUND TRIP
# ============================================================

@pytest.mark.unit
def test_ss_h5_round_trip(tmp_path):
    pk = [_ping(i, chans=(0, 1), num_chans_to_follow=4) for i in range(2)]
    pk += [_ping(i, n=N_SAMP if i % 2 else 50) for i in range(2, 6)]
    survey = convert(read_xtf(_dual(tmp_path, pk)))
    out = tmp_path / "x.ss.h5"
    write_survey(survey, out)
    loaded = read_survey(out)
    assert loaded.source_format == "xtf"
    assert loaded.frequencies == survey.frequencies
    for fk in survey.frequencies:
        xr.testing.assert_identical(
            loaded.get_frequency(fk).dataset, survey.get_frequency(fk).dataset
        )
    xr.testing.assert_identical(loaded.navigation, survey.navigation)
    xr.testing.assert_identical(loaded.attitude, survey.attitude)


# ============================================================
# PROJECTED NAVIGATION (NavUnits 0)
# ============================================================

# Two positions in UTM zone 19N (EPSG:32619) and their WGS84 lon/lat.
_UTM19N = 32619
_LONLAT = [(-71.40, 41.50), (-71.39, 41.501)]


def _projected_pings(n=4):
    from pyproj import Transformer

    fwd = Transformer.from_crs(4326, _UTM19N, always_xy=True)
    lons = np.linspace(_LONLAT[0][0], _LONLAT[1][0], n)
    lats = np.linspace(_LONLAT[0][1], _LONLAT[1][1], n)
    east, north = fwd.transform(lons, lats)
    pk = [_ping(i, SensorXcoordinate=east[i], SensorYcoordinate=north[i],
                ShipXcoordinate=east[i] + 5.0, ShipYcoordinate=north[i] - 5.0) for i in range(n)]
    return pk, lons, lats


@pytest.mark.unit
class TestProjectedNavigation:
    @pytest.mark.parametrize("crs", [_UTM19N, "EPSG:32619"])
    def test_nav_crs_converts_easting_northing_to_lonlat(self, tmp_path, crs):
        pk, lons, lats = _projected_pings()
        path = _dual(tmp_path, pk, nav_units=0)
        survey = convert(read_xtf(path), nav_crs=crs)
        ds = survey.get_frequency(survey.frequencies[0]).dataset
        np.testing.assert_allclose(ds["longitude"].values, lons, atol=1e-8)
        np.testing.assert_allclose(ds["latitude"].values, lats, atol=1e-8)
        np.testing.assert_allclose(survey.navigation["longitude"].values, lons, atol=1e-8)
        np.testing.assert_allclose(survey.navigation["latitude"].values, lats, atol=1e-8)
        assert "32619" in ds.attrs["xtf_nav_crs"]

    def test_nav_crs_accepts_pyproj_crs(self, tmp_path):
        from pyproj import CRS

        pk, lons, _ = _projected_pings()
        path = _dual(tmp_path, pk, nav_units=0)
        survey = convert(read_xtf(path), nav_crs=CRS.from_epsg(_UTM19N))
        ds = survey.get_frequency(survey.frequencies[0]).dataset
        np.testing.assert_allclose(ds["longitude"].values, lons, atol=1e-8)

    def test_zero_easting_northing_is_nan(self, tmp_path):
        pk, _, _ = _projected_pings(3)
        pk.append(_ping(3, SensorXcoordinate=0.0, SensorYcoordinate=0.0))
        path = _dual(tmp_path, pk, nav_units=0)
        survey = convert(read_xtf(path), nav_crs=_UTM19N)
        ds = survey.get_frequency(survey.frequencies[0]).dataset
        assert np.isnan(ds["latitude"].values[3]) and np.isnan(ds["longitude"].values[3])
        assert np.isfinite(ds["latitude"].values[:3]).all()

    def test_geographic_nav_crs_rejected(self, tmp_path):
        pk, _, _ = _projected_pings()
        path = _dual(tmp_path, pk, nav_units=0)
        with pytest.raises(ValueError, match="projected"):
            convert(read_xtf(path), nav_crs=4326)

    def test_nav_crs_ignored_for_latlon_files(self, dual_survey, tmp_path):
        path, expected = dual_survey
        survey = convert(read_xtf(path), nav_crs=_UTM19N)
        fk = expected.frequencies[0]
        np.testing.assert_array_equal(
            survey.get_frequency(fk).dataset["latitude"].values,
            expected.get_frequency(fk).dataset["latitude"].values,
        )
        assert "xtf_nav_crs" not in survey.get_frequency(fk).dataset.attrs

    def test_read_and_convert_passes_nav_crs(self, tmp_path):
        from sidescanpy.io import read_and_convert, read_and_convert_segments

        pk, lons, _ = _projected_pings()
        path = _dual(tmp_path, pk, nav_units=0)
        for survey in (read_and_convert([path], "xtf", nav_crs=_UTM19N),
                       read_and_convert_segments(path, nav_crs=_UTM19N, min_pings=1)[0][1]):
            ds = survey.get_frequency(survey.frequencies[0]).dataset
            np.testing.assert_allclose(ds["longitude"].values, lons, atol=1e-8)



# ============================================================
# LAYBACK AND POSITION SOURCE
# ============================================================

def _geo_pings(n=4, ship_offset_deg=None, **kw):
    """Pings at the default Sensor positions; Ship = Sensor + offset (None: no Ship)."""
    out = []
    for i in range(n):
        fields = dict(kw)
        if ship_offset_deg is not None:
            fields["ShipXcoordinate"] = -73.7 + i * 1e-5 + ship_offset_deg
            fields["ShipYcoordinate"] = -46.3 - i * 1e-5
        out.append(_ping(i, **fields))
    return out


@pytest.mark.unit
class TestLaybackAndPositionSource:
    def _ds(self, tmp_path, pings, **conv):
        survey = convert(read_xtf(_dual(tmp_path, pings)), **conv)
        return survey.get_frequency(survey.frequencies[0]).dataset

    def test_layback_and_cable_out_carried_per_ping(self, tmp_path):
        ds = self._ds(tmp_path, _geo_pings(Layback=29.5, CableOut=30, CableOutHundredths=25))
        np.testing.assert_allclose(ds["layback_m"].values, 29.5)
        np.testing.assert_allclose(ds["cable_out_m"].values, 30.25, rtol=1e-6)

    def test_position_source_towfish_when_sensor_away_from_ship(self, tmp_path):
        # 5e-4 deg of longitude at 46 S is about 39 m
        ds = self._ds(tmp_path, _geo_pings(ship_offset_deg=5e-4))
        assert ds.attrs["position_source"] == "towfish"
        assert "median 38." in ds.attrs["position_source_reason"]

    def test_position_source_unknown_when_ship_equals_sensor(self, tmp_path):
        ds = self._ds(tmp_path, _geo_pings(ship_offset_deg=0.0, Layback=126.0))
        assert ds.attrs["position_source"] == "unknown"

    def test_position_source_unknown_without_ship_position(self, tmp_path):
        ds = self._ds(tmp_path, _geo_pings(Layback=300.0))
        assert ds.attrs["position_source"] == "unknown"

    def test_isis_with_layback_and_no_ship_hinted_not_assumed(self, tmp_path):
        path = _dual(tmp_path, _geo_pings(Layback=300.0), RecordingProgramName=b"Isis")
        ds = convert(read_xtf(path)).get_frequency("hz_100000").dataset
        assert ds.attrs["position_source"] == "unknown"
        assert "(Isis)" in ds.attrs["position_source_reason"]
        assert "probably the ship's position" in ds.attrs["position_source_reason"]

    @pytest.mark.parametrize("pings, program", [
        (lambda: _geo_pings(), b"Isis"),                                    # no layback recorded
        (lambda: _geo_pings(ship_offset_deg=0.0, Layback=300.0), b"Isis"),  # Ship written
        (lambda: _geo_pings(Layback=300.0), b"synth"),                     # other writer
    ])
    def test_no_ship_hint_otherwise(self, tmp_path, pings, program):
        path = _dual(tmp_path, pings(), RecordingProgramName=program)
        ds = convert(read_xtf(path)).get_frequency("hz_100000").dataset
        assert ds.attrs["position_source"] == "unknown"
        assert "probably the ship" not in ds.attrs["position_source_reason"]

    def test_position_source_unknown_when_only_some_pings_apart(self, tmp_path):
        pk = _geo_pings(n=10, ship_offset_deg=0.0)
        pk[3] = _ping(3, ShipXcoordinate=-73.7 + 3e-5 + 5e-5, ShipYcoordinate=-46.3 - 3e-5)
        ds = self._ds(tmp_path, pk)
        assert ds.attrs["position_source"] == "unknown"

    def test_position_source_projected_compares_in_metres(self, tmp_path):
        pk, _, _ = _projected_pings()  # Ship 7 m from Sensor
        survey = convert(read_xtf(_dual(tmp_path, pk, nav_units=0)), nav_crs=_UTM19N)
        ds = survey.get_frequency(survey.frequencies[0]).dataset
        assert ds.attrs["position_source"] == "towfish"
        assert "median 7.07" in ds.attrs["position_source_reason"]
