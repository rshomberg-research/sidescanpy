"""
Unit tests for sidescan-correct — the consolidated far-range-crop / TVG /
geometric / georef in-place correction CLI, replacing the old per-project
02/04/06/08 numbered scripts.

2026 rshomberg@uri.edu
"""

import json

import pytest

from sidescanpy.cli import correct
from sidescanpy.core.history import has_step
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.processing import correct_survey


def _all_tvg(survey):
    return all(has_step(ch.dataset, "tvg") for ch in survey.freq_channels.values())


def _has_var(survey, var):
    return all(var in ch.dataset.data_vars for ch in survey.freq_channels.values())


def _crop_history_present(survey):
    for ch in survey.freq_channels.values():
        history = json.loads(ch.dataset.attrs.get("processing_history", "[]"))
        if not any(h.get("step") == "crop_farrange" for h in history):
            return False
    return True


@pytest.fixture
def ss_h5_path(tmp_path, sample_multi_frequency_survey):
    path = tmp_path / "survey_a.ss.h5"
    write_survey(sample_multi_frequency_survey, path)
    return path


class TestArgValidation:
    def test_requires_at_least_one_step(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["sidescan-correct", "--ss-h5-dir", str(tmp_path)])
        with pytest.raises(SystemExit):
            correct.main()

    def test_tvg_override_requires_tvg_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(tmp_path), "--geometric", "--no-spreading"],
        )
        with pytest.raises(SystemExit):
            correct.main()

    def test_no_spreading_and_explicit_value_mutually_exclusive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "sidescan-correct", "--ss-h5-dir", str(tmp_path), "--tvg",
                "--no-spreading", "--spreading-loss-db", "10",
            ],
        )
        with pytest.raises(SystemExit):
            correct.main()

    def test_no_files_found_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["sidescan-correct", "--ss-h5-dir", str(tmp_path), "--tvg"])
        assert correct.main() == 1

    def test_ping_chunk_size_accepts_crop_farrange(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--crop-farrange", "--ping-chunk-size", "4"],
        )
        assert correct.main() == 0
        assert _crop_history_present(read_survey(ss_h5_path))

    def test_ping_chunk_size_rejects_non_positive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(tmp_path), "--tvg", "--ping-chunk-size", "0"],
        )
        with pytest.raises(SystemExit):
            correct.main()


class TestPingChunkSize:
    """--ping-chunk-size routes through correct_file(chunk_pings=...) --
    the numerical-equivalence proof is in tests/unit/processing/test_correct.py;
    these confirm the CLI wiring."""

    def test_tvg_only_chunked(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--tvg", "--ping-chunk-size", "4"],
        )
        assert correct.main() == 0

        result = read_survey(ss_h5_path)
        assert _all_tvg(result)

    def test_all_steps_chunked(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent),
                "--tvg", "--geometric", "--georef", "--ping-chunk-size", "3",
            ],
        )
        assert correct.main() == 0

        result = read_survey(ss_h5_path)
        assert _has_var(result, "latitude_per_sample")

    def test_already_done_chunked_run_is_skipped(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--tvg", "--ping-chunk-size", "4"],
        )
        assert correct.main() == 0
        assert correct.main() == 0  # second run: already done, still exits 0


class TestSingleStep:
    def test_tvg_only(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--tvg"])
        assert correct.main() == 0

        result = read_survey(ss_h5_path)
        assert _all_tvg(result)
        assert not _has_var(result, "ground_range_m")

    def test_georef_alone(self, tmp_path, sample_multi_frequency_survey, monkeypatch):
        """Matches the old standalone-stage-script behavior — a single
        step still works exactly like before, not just combined runs.
        Requires geometric to already have run (needs across_track_m/
        along_track_m), same precondition the old 08_georef.py documented."""
        pre_geometric = correct_survey(sample_multi_frequency_survey, ["interpolate_aux", "geometric"])
        path = tmp_path / "survey_geo.ss.h5"
        write_survey(pre_geometric, path)

        monkeypatch.setattr("sys.argv", ["sidescan-correct", "--ss-h5-dir", str(tmp_path), "--georef"])
        assert correct.main() == 0

        result = read_survey(path)
        assert _has_var(result, "latitude_per_sample")


class TestCombinedSteps:
    def test_all_four_in_one_pass(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent),
                "--crop-farrange", "--tvg", "--geometric", "--georef",
            ],
        )
        assert correct.main() == 0

        result = read_survey(ss_h5_path)
        assert _all_tvg(result)
        assert _has_var(result, "ground_range_m")
        assert _has_var(result, "latitude_per_sample")
        assert _crop_history_present(result)

    def test_range_correction_included_in_one_pass(self, ss_h5_path, monkeypatch):
        # sample_multi_frequency_survey's two channels (hz_1495000/hz_547500)
        # are exactly the bundled default calibration's own scope.
        monkeypatch.setattr(
            "sys.argv",
            [
                "sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent),
                "--tvg", "--geometric", "--range-correction",
            ],
        )
        assert correct.main() == 0

        result = read_survey(ss_h5_path)
        for ch in result.freq_channels.values():
            history = json.loads(ch.dataset.attrs["processing_history"])
            assert any(h.get("step") == "range_correction" for h in history)

    def test_tvg_override_applied(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--tvg", "--no-spreading", "--no-absorption"],
        )
        assert correct.main() == 0
        result = read_survey(ss_h5_path)
        assert _all_tvg(result)
        for ch in result.freq_channels.values():
            tvg = [h for h in json.loads(ch.dataset.attrs["processing_history"]) if h["step"] == "tvg"][0]
            assert tvg["params"]["spreading_loss_db"] == 0.0
            assert tvg["params"]["absorption_db_per_m"] == 0.0


class TestConfigFile:
    def test_config_seeds_tvg_override(self, ss_h5_path, tmp_path, monkeypatch):
        import json
        config_path = tmp_path / "survey.json"
        config_path.write_text(json.dumps({"no_spreading": True}))

        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--tvg", "--config", str(config_path)],
        )
        assert correct.main() == 0
        result = read_survey(ss_h5_path)
        assert _all_tvg(result)

    def test_config_can_seed_a_step_flag_not_typed_on_the_command_line(self, ss_h5_path, tmp_path, monkeypatch):
        import json
        config_path = tmp_path / "survey.json"
        config_path.write_text(json.dumps({"crop_farrange": True}))

        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--tvg", "--config", str(config_path)],
        )
        assert correct.main() == 0
        result = read_survey(ss_h5_path)
        assert _crop_history_present(result)  # crop_farrange applied even though only --tvg was typed


class TestIdempotency:
    def test_rerun_skips_completed_step(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--tvg"])
        assert correct.main() == 0
        assert correct.main() == 0  # would raise inside apply_tvg if double-applied

    def test_range_correction_rerun_skips_completed_step(self, ss_h5_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-correct", "--ss-h5-dir", str(ss_h5_path.parent), "--tvg", "--geometric", "--range-correction"],
        )
        assert correct.main() == 0
        assert correct.main() == 0  # would raise inside apply_empirical_range_correction if double-applied

    def test_range_correction_added_in_later_invocation(self, tmp_path, sample_multi_frequency_survey):
        """--range-correction must work standalone once geometric was
        already applied in an earlier, separate sidescan-correct run --
        idempotency here is judged from the file's own recorded state,
        not from which flags happen to be typed on this invocation."""
        pre_corrected = correct_survey(sample_multi_frequency_survey, ["interpolate_aux", "tvg", "geometric"])
        path = tmp_path / "survey_d.ss.h5"
        write_survey(pre_corrected, path)

        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["sidescan-correct", "--ss-h5-dir", str(tmp_path), "--range-correction"]
            assert correct.main() == 0
        finally:
            sys.argv = old_argv

        result = read_survey(path)
        for ch in result.freq_channels.values():
            history = json.loads(ch.dataset.attrs["processing_history"])
            assert any(h.get("step") == "range_correction" for h in history)

    def test_pending_subset_only_applies_missing_step(self, tmp_path, sample_multi_frequency_survey):
        """A file already through --tvg, then run with --tvg --geometric,
        must only apply geometric -- re-running tvg on an already-radiometric
        survey raises by design (prevents double-TVG)."""
        pre_corrected = correct_survey(sample_multi_frequency_survey, ["interpolate_aux", "tvg"])
        path = tmp_path / "survey_b.ss.h5"
        write_survey(pre_corrected, path)

        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["sidescan-correct", "--ss-h5-dir", str(tmp_path), "--tvg", "--geometric"]
            assert correct.main() == 0
        finally:
            sys.argv = old_argv

        result = read_survey(path)
        assert _all_tvg(result)
        assert _has_var(result, "ground_range_m")

    def test_all_steps_already_done_is_skipped(self, tmp_path, sample_multi_frequency_survey):
        pre_corrected = correct_survey(sample_multi_frequency_survey, ["interpolate_aux", "tvg", "geometric"])
        path = tmp_path / "survey_c.ss.h5"
        write_survey(pre_corrected, path)
        mtime_before = path.stat().st_mtime_ns

        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["sidescan-correct", "--ss-h5-dir", str(tmp_path), "--tvg", "--geometric"]
            assert correct.main() == 0
        finally:
            sys.argv = old_argv

        assert path.stat().st_mtime_ns == mtime_before
