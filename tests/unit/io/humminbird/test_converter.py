"""
Unit tests for sidescanpy.io.humminbird.converter on synthetic recordings.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.io.humminbird import HUMMINBIRD_SAMPLE_RATE_HZ, convert, read_humminbird
from tests.unit.io.humminbird.conftest import START_UNIX, write_recording


def _convert(tmp_path, **kw):
    sample_rate = kw.pop("sample_rate_hz", None)
    return convert(read_humminbird(write_recording(tmp_path, **kw)), sample_rate_hz=sample_rate)


@pytest.mark.unit
class TestConvert:
    def test_one_frequency_channel_validates(self, tmp_path):
        survey = _convert(tmp_path)
        assert survey.frequencies == ["hz_455000"]
        assert survey.source_format == "humminbird"
        ds = survey.get_frequency("hz_455000").dataset
        ds.sidescan.validate()
        assert dict(ds.sizes) == {"ping": 5, "channel": 2, "sample": 100}

    def test_port_and_starboard_near_to_far_unreversed(self, tmp_path):
        ds = _convert(tmp_path).get_frequency("hz_455000").dataset
        bs = ds.backscatter_intensity.values
        np.testing.assert_array_equal(bs[2, 0, :3], [2, 3, 4])       # port, as stored
        np.testing.assert_array_equal(bs[2, 1, :3], [102, 103, 104])  # starboard, as stored

    def test_units_and_missing_fields(self, tmp_path):
        ds = _convert(tmp_path).get_frequency("hz_455000").dataset
        np.testing.assert_allclose(ds.altitude_m.values, [3.0, 3.1, 3.2, 3.3, 3.4], rtol=1e-6)
        np.testing.assert_allclose(ds.fathometer_reading_m.values, ds.altitude_m.values)
        np.testing.assert_allclose(ds.heading_deg.values, 90.0)
        for name in ("depth_m", "roll_deg", "pitch_deg", "heave_m", "sound_vel_ms"):
            assert np.isnan(ds[name].values).all(), name
        assert ds.latitude.values[0] == pytest.approx(64.8, abs=1e-5)
        assert ds.longitude.values[0] == pytest.approx(-147.9, abs=1e-5)

    def test_times_from_dat_start_plus_ms(self, tmp_path):
        ds = _convert(tmp_path).get_frequency("hz_455000").dataset
        start = np.datetime64(START_UNIX, "s").astype("datetime64[ns]")
        assert ds.time.values[0] == start
        assert ds.time.values[3] == start + np.timedelta64(150, "ms")

    def test_default_and_given_sample_rate(self, tmp_path):
        ds = _convert(tmp_path / "a").get_frequency("hz_455000").dataset
        assert ds.attrs["sample_rate_hz"] == pytest.approx(HUMMINBIRD_SAMPLE_RATE_HZ)
        assert 1500 / (2 * HUMMINBIRD_SAMPLE_RATE_HZ) == pytest.approx(0.0194, abs=1e-4)
        assert ds.attrs["humminbird_sample_rate_source"].startswith("assumed")
        ds = _convert(tmp_path / "b", sample_rate_hz=40_000.0).get_frequency("hz_455000").dataset
        assert ds.attrs["sample_rate_hz"] == 40_000.0
        assert ds.attrs["humminbird_sample_rate_source"] == "given by caller"

    def test_bad_sample_rate_raises(self, tmp_path):
        with pytest.raises(ValueError, match="positive"):
            _convert(tmp_path, sample_rate_hz=0.0)

    def test_starboard_one_ms_late_still_paired(self, tmp_path):
        ds = _convert(tmp_path, stbd_time_offset_ms=1).get_frequency("hz_455000").dataset
        assert ds.sizes["ping"] == 5
        assert ds.attrs["humminbird_pings_one_side"] == 0
        assert (ds.valid_samples.values == 100).all()

    def test_one_sided_ping_kept_with_nan(self, tmp_path):
        from tests.unit.io.humminbird.conftest import frame, ramp
        extra = frame(beam=2, record=99, time_ms=10_000, samples=ramp(100))
        ds = _convert(tmp_path, extra_frames={2: extra}).get_frequency("hz_455000").dataset
        assert ds.sizes["ping"] == 6
        assert ds.attrs["humminbird_pings_one_side"] == 1
        assert ds.valid_samples.values[-1].tolist() == [100, 0]
        assert np.isnan(ds.backscatter_intensity.values[-1, 1]).all()

    def test_varying_sample_counts_nan_padded(self, tmp_path):
        ds = _convert(tmp_path, n_samples=[90, 100, 95, 100, 90]).get_frequency("hz_455000").dataset
        assert ds.sizes["sample"] == 100
        assert ds.valid_samples.values[:, 0].tolist() == [90, 100, 95, 100, 90]
        assert np.isnan(ds.backscatter_intensity.values[0, :, 90:]).all()

    def test_down_beams_skipped_with_warning(self, tmp_path, caplog):
        survey = _convert(tmp_path, beams=(1, 2, 3, 4))
        ds = survey.get_frequency("hz_455000").dataset
        assert ds.attrs["humminbird_beams_skipped"] == "1,4"
        assert "not converted" in caplog.text

    def test_no_side_beams_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no side imaging"):
            _convert(tmp_path, beams=(1,))

    def test_position_source_unknown(self, tmp_path):
        ds = _convert(tmp_path).get_frequency("hz_455000").dataset
        assert ds.attrs["position_source"] == "unknown"

    def test_navigation_record(self, tmp_path):
        nav = _convert(tmp_path).navigation
        assert nav.sizes["obs"] == 5
        np.testing.assert_allclose(nav.speed_knots.values, 1.5 * 3600 / 1852, rtol=1e-5)

    def test_two_recordings_merged_in_time_order(self, tmp_path):
        a = read_humminbird(write_recording(tmp_path / "a", "R00002"))
        b = read_humminbird(write_recording(tmp_path / "b", "R00001"))
        ds = convert([a, b]).get_frequency("hz_455000").dataset
        assert ds.sizes["ping"] == 10
        assert np.all(np.diff(ds.time.values.astype("int64")) >= 0)

    def test_round_trip_through_ss_h5(self, tmp_path):
        survey = _convert(tmp_path)
        out = tmp_path / "out.ss.h5"
        write_survey(survey, out)
        back = read_survey(out).get_frequency("hz_455000").dataset
        np.testing.assert_array_equal(
            back.backscatter_intensity.values,
            survey.get_frequency("hz_455000").dataset.backscatter_intensity.values,
        )
        assert back.attrs["humminbird_header_sizes"] == "72"
