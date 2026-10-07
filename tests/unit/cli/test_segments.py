"""
Unit tests for sidescanpy.cli._segments (segment file output).
"""

import json

import numpy as np

from sidescanpy.cli._segments import write_segments, write_segments_report
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.processing.segments import Segment
from tests.conftest import make_minimal_sidescan_dataset


def _uniform_track_dataset(n_pings=200, n_samples=50):
    """A clean baseline: perfectly uniform lat/lon spacing (linspace ->
    constant along-track step), constant sound_vel_ms, nonzero random
    backscatter, constant altitude/pitch. Every detector should stay
    quiet against this on its own."""
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples)
    ds["latitude"] = ("ping", np.linspace(41.0000, 41.0100, n_pings))
    ds["longitude"] = ("ping", np.linspace(-71.0000, -71.0000, n_pings))
    return ds



class TestWriteSegments:
    def _survey(self, n_pings=200):
        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="survey_a")
        survey.add_frequency("hz_1495000", _uniform_track_dataset(n_pings=n_pings))
        return survey

    def test_writes_only_clean_segments(self, tmp_path):
        survey = self._survey()
        segments = [
            Segment(label="section1", start=0, stop=80, status="clean", reasons=["clean"]),
            Segment(label="section2", start=80, stop=120, status="excluded", reasons=["garbage_run"]),
            Segment(label="section3", start=120, stop=200, status="clean", reasons=["clean"]),
        ]
        written = write_segments(survey, segments, tmp_path / "sections")
        names = {p.name for p in written}
        assert names == {"section1.ss.h5", "section3.ss.h5"}
        assert not (tmp_path / "sections" / "section2.ss.h5").exists()

    def test_written_segment_has_correct_ping_range(self, tmp_path):
        survey = self._survey()
        segments = [Segment(label="section1", start=50, stop=150, status="clean", reasons=["clean"])]
        [out_path] = write_segments(survey, segments, tmp_path / "sections")
        reread = read_survey(out_path)
        assert reread.get_frequency("hz_1495000").dataset.sizes["ping"] == 100

    def test_every_frequency_channel_sliced(self, tmp_path):
        survey = self._survey()
        survey.add_frequency("hz_547500", _uniform_track_dataset(n_pings=200))
        segments = [Segment(label="section1", start=0, stop=40, status="clean", reasons=["clean"])]
        [out_path] = write_segments(survey, segments, tmp_path / "sections")
        reread = read_survey(out_path)
        assert reread.get_frequency("hz_1495000").dataset.sizes["ping"] == 40
        assert reread.get_frequency("hz_547500").dataset.sizes["ping"] == 40


class TestWriteSegmentsReport:
    def test_report_round_trips(self, tmp_path):
        segments = [
            Segment(label="section1", start=0, stop=100, status="clean", reasons=["clean"]),
            Segment(label="section2", start=100, stop=150, status="excluded", reasons=["gap", "garbage_run"]),
        ]
        path = tmp_path / "segments.json"
        write_segments_report(segments, path)
        data = json.loads(path.read_text())
        assert len(data["segments"]) == 2
        assert data["segments"][1]["status"] == "excluded"
        assert data["segments"][1]["reasons"] == ["gap", "garbage_run"]
