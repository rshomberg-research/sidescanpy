"""
Unit tests for sidescanpy.io.humminbird.reader on synthetic recordings.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.io.humminbird import (
    humminbird_to_lonlat,
    lonlat_to_humminbird,
    read_dat,
    read_humminbird,
    read_son,
    son_directory,
)
from tests.unit.io.humminbird.conftest import (
    START_UNIX,
    TAGS_67,
    dat_bytes,
    frame,
    ramp,
    write_recording,
)


@pytest.mark.unit
class TestDat:
    def test_reads_helix_header(self, tmp_path):
        p = tmp_path / "R00001.DAT"
        p.write_bytes(dat_bytes(name="R00001.SON", n_records=12, ms=550))
        dat = read_dat(p)
        assert (dat.size, dat.unix_time, dat.n_records, dat.recording_ms) == (64, START_UNIX, 12, 550)
        assert dat.name == "R00001.SON"
        assert dat.water_type == "fresh"

    def test_water_code_mapping_helix(self, tmp_path):
        p = tmp_path / "R.DAT"
        p.write_bytes(dat_bytes(water_code=1))
        assert read_dat(p).water_type == "deep salt"

    def test_onix_xml_header_not_read(self, tmp_path):
        p = tmp_path / "R.DAT"
        p.write_bytes(b"header<NumberOfPings=10><TotalTimeMs=500>" + b" " * 300)
        with pytest.raises(NotImplementedError, match="XML"):
            read_dat(p)


@pytest.mark.unit
class TestSon:
    def test_decodes_tagged_header_fields(self, tmp_path):
        p = tmp_path / "B002.SON"
        p.write_bytes(frame(beam=2, record=7, time_ms=1234, samples=ramp(20), depth_dm=42,
                            heading_ddeg=1805, speed_dms=12))
        beam = read_son(p)
        f = beam.fields
        assert beam.beam == 2 and beam.n_pings == 1
        assert (f["record_number"][0], f["time_ms"][0], f["depth_dm"][0]) == (7, 1234, 42)
        assert (f["heading_ddeg"][0], f["speed_dms"][0], f["gps_flag_1"][0]) == (1805, 12, 1)
        assert f["frequency_hz"][0] == 455_000
        assert "tag_134" in f  # unknown tags kept under their number
        assert beam.header_sizes == {72: 1}

    def test_samples_padded_with_nan(self, tmp_path):
        p = tmp_path / "B002.SON"
        p.write_bytes(frame(beam=2, record=0, time_ms=0, samples=ramp(5))
                      + frame(beam=2, record=1, time_ms=50, samples=ramp(3, 10)))
        s = read_son(p).samples()
        assert s.shape == (2, 5)
        np.testing.assert_array_equal(s[0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(s[1, :3], [10, 11, 12])
        assert np.isnan(s[1, 3:]).all()

    def test_67_byte_header(self, tmp_path):
        p = tmp_path / "B002.SON"
        p.write_bytes(frame(beam=2, record=0, time_ms=0, samples=ramp(4), tags=TAGS_67))
        beam = read_son(p)
        assert beam.header_sizes == {67: 1}
        assert "tag_134" not in beam.fields

    def test_signed_easting(self, tmp_path):
        p = tmp_path / "B002.SON"
        p.write_bytes(frame(beam=2, record=0, time_ms=0, samples=ramp(4)))
        assert read_son(p).fields["easting"][0] < 0  # western hemisphere

    def test_truncated_final_frame_dropped(self, tmp_path):
        p = tmp_path / "B002.SON"
        whole = frame(beam=2, record=0, time_ms=0, samples=ramp(10))
        p.write_bytes(whole + whole[:-4])
        beam = read_son(p)
        assert beam.n_pings == 1 and beam.n_truncated == 1

    def test_garbage_between_frames_skipped(self, tmp_path, caplog):
        p = tmp_path / "B002.SON"
        a = frame(beam=2, record=0, time_ms=0, samples=ramp(10))
        b = frame(beam=2, record=1, time_ms=50, samples=ramp(10))
        p.write_bytes(a + b"\x00\x01\x02" + b)
        beam = read_son(p)
        assert beam.n_pings == 2 and beam.n_resyncs == 1
        assert "no ping frame" in caplog.text

    def test_not_a_son_file(self, tmp_path):
        p = tmp_path / "B002.SON"
        p.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="no Humminbird ping frames"):
            read_son(p)


@pytest.mark.unit
class TestRecording:
    def test_finds_son_directory_any_case(self, tmp_path):
        dat = write_recording(tmp_path, "R00001")
        (tmp_path / "R00001").rename(tmp_path / "r00001")
        assert son_directory(dat).name == "r00001"

    def test_missing_son_directory(self, tmp_path):
        p = tmp_path / "R00009.DAT"
        p.write_bytes(dat_bytes())
        with pytest.raises(FileNotFoundError, match="SON directory"):
            read_humminbird(p)

    def test_reads_every_beam(self, tmp_path):
        dat = write_recording(tmp_path, beams=(1, 2, 3, 4))
        rec = read_humminbird(dat)
        assert sorted(rec.beams) == [1, 2, 3, 4]
        assert rec.beams[3].n_pings == 5

    def test_reads_selected_beams(self, tmp_path):
        dat = write_recording(tmp_path, beams=(1, 2, 3, 4))
        assert sorted(read_humminbird(dat, beams=(2, 3)).beams) == [2, 3]


@pytest.mark.unit
class TestCoordinates:
    def test_round_trip(self):
        lon = np.array([-147.9, -111.5, 10.0, 151.2])
        lat = np.array([64.8, 36.9, -33.9, -33.9])
        e, n = lonlat_to_humminbird(lon, lat)
        lon2, lat2 = humminbird_to_lonlat(e, n)
        np.testing.assert_allclose(lon2, lon, atol=1e-9)
        np.testing.assert_allclose(lat2, lat, atol=1e-9)

    def test_known_point(self):
        # A recorded Humminbird position on the Chena River, Fairbanks, Alaska.
        lon, lat = humminbird_to_lonlat(-16466182, 9516885)
        assert lon == pytest.approx(-147.9124, abs=1e-4)
        assert lat == pytest.approx(64.7982, abs=1e-4)
