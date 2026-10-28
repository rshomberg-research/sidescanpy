"""
Tests for the one correction API: correct / correct_survey / correct_file,
invert / uncorrected_backscatter, and the derived processing level.

2026 rshomberg@uri.edu
"""

import json
import logging

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core import SideScanSurvey
from sidescanpy.core.history import applied_steps, processing_level, read_history
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.processing import (
    DEFAULT_STEPS,
    Step,
    correct,
    correct_file,
    correct_survey,
    estimate_correct_disk_bytes,
    francois_garrison_absorption_db_per_m,
    invert,
    uncorrected_backscatter,
)
from sidescanpy.processing.range_correction import RangeCorrectionTable
from tests.conftest import (
    make_minimal_attitude_record,
    make_minimal_navigation_record,
    make_minimal_sidescan_dataset,
)


def _ds(n_pings=40, n_samples=400, seed=0):
    """Channel whose swath reaches past the 5 m altitude (geometry is finite)."""
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples)
    rng = np.random.default_rng(seed)
    bs = rng.uniform(0.5, 80.0, size=ds["backscatter_intensity"].shape).astype(np.float32)
    bs[:, :, :5] = 0.0  # exact-zero blanking run, clipped to the dB floor by TVG
    ds["backscatter_intensity"].values[...] = bs
    ds.attrs["sample_rate_hz"] = 20_000.0  # 0.0375 m/sample -> 15 m swath
    return ds


def _survey(n_pings=40):
    survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
    hf = _ds(n_pings, seed=1)
    lf = _ds(n_pings, seed=2)
    lf.attrs["freq_hz_start"] = lf.attrs["freq_hz_end"] = 547_500
    survey.add_frequency("hz_1495000", hf)
    survey.add_frequency("hz_547500", lf)
    survey.navigation = make_minimal_navigation_record(n_obs=n_pings)
    survey.attitude = make_minimal_attitude_record(n_obs=n_pings)
    return survey


def _rc_table(freqs=(1_495_000, 547_500)):
    """Small synthetic range-correction table (flat +2 dB, ramping to +4 dB)."""
    table = RangeCorrectionTable()
    for f in freqs:
        table.add_curve(f, [0.0, 5.0, 20.0], [2.0, 3.0, 4.0])
    return table


def _steps(ds):
    return [h["step"] for h in read_history(ds)]


# ============================================================
# correct()
# ============================================================

class TestCorrectSteps:
    def test_steps_applied_in_given_order_and_recorded(self):
        out = correct(_ds(), ["tvg", "geometric", "georef"])
        assert _steps(out) == ["tvg", "geometric", "georef"]
        assert "latitude_per_sample" in out

    def test_history_record_format(self):
        out = correct(_ds(), ["tvg"])
        rec = read_history(out)[0]
        assert set(rec) == {"step", "params", "function", "version", "timestamp_utc"}
        # actual params used, defaults resolved
        assert rec["params"] == {
            "spreading_loss_db": 30.0, "absorption_db_per_m": pytest.approx(francois_garrison_absorption_db_per_m(1_495_000)),
            "min_range_m": 0.5, "sound_velocity_ms": 1500.0,
            "temperature_c": 10.0, "salinity_psu": 35.0, "depth_m": 10.0, "ph": 8.0,
        }

    def test_skip_by_leaving_out(self):
        out = correct(_ds(), ["geometric"])
        assert _steps(out) == ["geometric"]
        np.testing.assert_array_equal(out["backscatter_intensity"].values, _ds()["backscatter_intensity"].values)

    def test_params_override(self):
        out = correct(_ds(), ["tvg"], {"tvg": {"spreading_loss_db": 0.0, "absorption_db_per_m": 0.0}})
        np.testing.assert_allclose(
            out["backscatter_intensity"].values[:, :, 10:],
            20 * np.log10(_ds()["backscatter_intensity"].values[:, :, 10:]), rtol=1e-5,
        )

    def test_default_steps_skip_interpolate_aux_without_aux(self, caplog):
        with caplog.at_level("DEBUG"):
            out = correct(_ds())
        assert _steps(out) == [s for s in DEFAULT_STEPS if s != "interpolate_aux"]
        assert "interpolate_aux skipped: no aux given" in caplog.text

    def test_skipped_interpolate_aux_warns_when_no_aux(self, caplog):
        with caplog.at_level("WARNING", logger="sidescanpy"):
            correct(_ds())
        msgs = [r for r in caplog.records if "interpolate_aux skipped" in r.getMessage()]
        assert len(msgs) == 1
        assert msgs[0].levelname == "WARNING"
        assert "aux=" in msgs[0].getMessage()

    def test_skipped_interpolate_aux_no_warning_when_already_applied(self, caplog):
        survey = _survey()
        done = correct(survey.get_frequency("hz_1495000").dataset, ["interpolate_aux"], aux=survey)
        with caplog.at_level("WARNING", logger="sidescanpy"):
            correct(done, ["interpolate_aux", "tvg"])
        assert "interpolate_aux skipped" not in caplog.text

    def test_no_skip_warning_when_step_not_requested(self, caplog):
        with caplog.at_level("WARNING", logger="sidescanpy"):
            correct(_ds(), ["crop_farrange"])
        assert "interpolate_aux skipped" not in caplog.text

    def test_explicit_interpolate_aux_params_need_aux(self):
        with pytest.raises(ValueError, match="aux"):
            correct(_ds(), ["interpolate_aux", "tvg"], {"interpolate_aux": {"method": "linear"}})

    def test_default_steps_with_aux(self):
        survey = _survey()
        out = correct(survey.get_frequency("hz_1495000").dataset, aux=survey)
        assert _steps(out) == list(DEFAULT_STEPS)

    def test_input_not_modified(self):
        ds = _ds()
        before = ds["backscatter_intensity"].values.copy()
        correct(ds, ["tvg", "geometric"])
        np.testing.assert_array_equal(ds["backscatter_intensity"].values, before)
        assert read_history(ds) == []


class TestCustomSteps:
    def test_callable_recorded_under_qualname_with_bound_params(self):
        def add_gain(ds, gain=2.0):
            return ds.assign(backscatter_intensity=ds["backscatter_intensity"] + gain)

        out = correct(_ds(), ["tvg", add_gain], {"add_gain": {"gain": 3.0}})
        rec = read_history(out)[-1]
        assert rec["step"] == "add_gain"
        assert rec["params"] == {"gain": 3.0}

    def test_step_dataclass_names_the_record_and_defaults_recorded(self):
        def fn(ds, offset=1.5):
            return ds
        out = correct(_ds(), [Step("my_offset", fn)])
        assert read_history(out)[-1]["step"] == "my_offset"
        assert read_history(out)[-1]["params"] == {"offset": 1.5}

    def test_custom_step_replaces_builtin(self):
        def my_tvg(ds):
            return ds.assign(backscatter_intensity=ds["backscatter_intensity"] * 0 + 7.0)
        out = correct(_ds(), [my_tvg, "geometric"])
        assert _steps(out) == ["my_tvg", "geometric"]
        assert processing_level(out) == "geometric"

    def test_lambda_rejected(self):
        with pytest.raises(ValueError, match="Step"):
            correct(_ds(), [lambda ds: ds])

    def test_builtin_name_collision_rejected(self):
        with pytest.raises(ValueError, match="collides"):
            correct(_ds(), [Step("tvg", lambda ds: ds)])

    def test_non_dataset_return_raises(self):
        def bad(ds):
            return 1
        with pytest.raises(TypeError, match="xr.Dataset"):
            correct(_ds(), [bad])


class TestErrors:
    def test_unknown_step_lists_valid(self):
        with pytest.raises(ValueError, match="Built-in steps"):
            correct(_ds(), ["tgv"])

    def test_unknown_param_typeerror_before_anything_runs(self):
        with pytest.raises(TypeError, match="Accepted params"):
            correct(_ds(), ["geometric", "tvg"], {"tvg": {"spreading": 1.0}})

    def test_params_for_step_not_in_steps(self):
        with pytest.raises(ValueError, match="not in steps"):
            correct(_ds(), ["geometric"], {"tvg": {"spreading_loss_db": 0.0}})

    def test_params_none_value_rejected(self):
        with pytest.raises(TypeError, match="mapping"):
            correct(_ds(), ["tvg"], {"tvg": None})

    def test_duplicate_step(self):
        with pytest.raises(ValueError, match="twice"):
            correct(_ds(), ["tvg", "tvg"])

    def test_aux_not_a_param(self):
        with pytest.raises(TypeError, match="caller API"):
            correct(_ds(), ["interpolate_aux"], {"interpolate_aux": {"aux": None}})


class TestIdempotency:
    def test_second_run_is_noop(self):
        once = correct(_ds(), ["tvg", "geometric", "georef"])
        twice = correct(once, ["tvg", "geometric", "georef"])
        assert twice.attrs["processing_history"] == once.attrs["processing_history"]
        np.testing.assert_array_equal(twice["backscatter_intensity"].values, once["backscatter_intensity"].values)

    def test_custom_step_matched_by_name(self):
        calls = []

        def counted(ds):
            calls.append(1)
            return ds
        correct(correct(_ds(), [counted]), [counted])
        assert len(calls) == 1

    def test_tvg_water_conditions_drive_default_absorption_and_are_recorded(self):
        warm = correct(_ds(), ["tvg"], {"tvg": {"temperature_c": 25.0, "salinity_psu": 30.0}})
        rec = read_history(warm)[0]["params"]
        assert rec["temperature_c"] == 25.0 and rec["salinity_psu"] == 30.0
        assert rec["depth_m"] == 10.0 and rec["ph"] == 8.0
        assert rec["absorption_db_per_m"] == pytest.approx(
            francois_garrison_absorption_db_per_m(1_495_000, temperature_c=25.0, salinity_psu=30.0))

    def test_tvg_explicit_absorption_ignores_water_conditions(self):
        out = correct(_ds(), ["tvg"], {"tvg": {"absorption_db_per_m": 0.1, "temperature_c": 25.0}})
        rec = read_history(out)[0]["params"]
        assert rec["absorption_db_per_m"] == 0.1
        assert "temperature_c" not in rec

    def test_force_reapplies_tvg_with_new_params_not_stacked(self):
        once = correct(_ds(), ["tvg"])
        redo = correct(once, ["tvg"], {"tvg": {"spreading_loss_db": 0.0}}, force=True)
        direct = correct(_ds(), ["tvg"], {"tvg": {"spreading_loss_db": 0.0}})
        np.testing.assert_allclose(redo["backscatter_intensity"].values,
                                   direct["backscatter_intensity"].values, atol=1e-3)
        assert _steps(redo) == ["tvg"]

    def test_force_tvg_requires_later_radiometric_steps_listed(self):
        ds = correct(_ds(), ["tvg", "geometric"])
        rc = {"range_correction": {"table": _rc_table()}}
        ds = correct(ds, ["range_correction"], rc)
        with pytest.raises(ValueError, match="range_correction"):
            correct(ds, ["tvg"], force=True)
        redo = correct(ds, ["tvg", "range_correction"], rc, force=True)
        assert _steps(redo).count("tvg") == 1

    def test_geometric_done_if_vars_present_without_record(self):
        ds = correct(_ds(), ["geometric"])
        ds.attrs["processing_history"] = "[]"
        assert _steps(correct(ds, ["geometric"])) == []

    def test_geometric_not_done_if_all_nan(self):
        ds = correct(_ds(), ["geometric"])
        ds.attrs["processing_history"] = "[]"
        ds["slant_range_m"] = ds["slant_range_m"] * np.nan
        assert _steps(correct(ds, ["geometric"])) == ["geometric"]


# ============================================================
# invert / uncorrected_backscatter
# ============================================================

class TestInvert:
    def test_round_trip_tvg_geometric_georef(self):
        ds = _ds()
        back = invert(correct(ds, ["tvg", "geometric", "georef"]))
        np.testing.assert_allclose(back["backscatter_intensity"].values,
                                   ds["backscatter_intensity"].values, rtol=1e-5, atol=1e-6)
        assert np.all(back["backscatter_intensity"].values[:, :, :5] == 0.0)  # floor -> exact 0
        assert "ground_range_m" not in back and "latitude_per_sample" not in back
        assert read_history(back) == []
        assert processing_level(back) == "raw"

    def test_round_trip_with_range_correction(self):
        ds = _ds()
        corrected = correct(ds, ["tvg", "geometric", "range_correction"], {"range_correction": {"table": _rc_table()}})
        assert "range_correction" in applied_steps(corrected)
        np.testing.assert_allclose(uncorrected_backscatter(corrected).values,
                                   ds["backscatter_intensity"].values, rtol=1e-5, atol=1e-6)

    def test_partial_invert_keeps_geometry(self):
        corrected = correct(_ds(), ["tvg", "geometric"])
        back = invert(corrected, ["tvg"])
        assert "ground_range_m" in back
        assert _steps(back) == ["geometric"]

    def test_cannot_invert_tvg_under_later_range_correction(self):
        corrected = correct(_ds(), ["tvg", "geometric", "range_correction"], {"range_correction": {"table": _rc_table()}})
        with pytest.raises(ValueError, match="range_correction"):
            invert(corrected, ["tvg"])

    def test_range_correction_without_table_is_a_clear_error(self):
        with pytest.raises(TypeError, match="requires.*table"):
            correct(_ds(), ["tvg", "geometric", "range_correction"])

    def test_non_invertible_step(self):
        with pytest.raises(ValueError, match="cannot invert"):
            invert(_ds(), ["crop_farrange"])

    def test_noop_on_uncorrected(self):
        ds = _ds()
        xr.testing.assert_identical(uncorrected_backscatter(ds), ds["backscatter_intensity"])

    def test_legacy_tvg_record_name_inverts(self):
        ds = _ds()
        corrected = correct(ds, ["tvg"])
        hist = read_history(corrected)
        hist[0]["step"] = "tvg_correction"
        hist[0]["params"]["sound_velocity_used_ms"] = hist[0]["params"].pop("sound_velocity_ms")
        corrected.attrs["processing_history"] = json.dumps(hist)
        np.testing.assert_allclose(uncorrected_backscatter(corrected).values,
                                   ds["backscatter_intensity"].values, rtol=1e-5, atol=1e-6)


class TestProcessingLevel:
    def test_derived_from_history(self):
        ds = _ds()
        assert processing_level(ds) == "raw"
        assert processing_level(correct(ds, ["tvg"])) == "tvg"
        assert processing_level(correct(ds, ["tvg", "geometric", "georef"])) == "georef"

    def test_legacy_attribute_fallback(self):
        ds = _ds()
        ds.attrs["processing_level"] = "radiometric"
        assert processing_level(ds) == "tvg"
        ds.attrs["processing_level"] = "raw"
        assert processing_level(ds) == "raw"

    def test_legacy_attribute_dropped_once_history_written(self):
        ds = _ds()
        ds.attrs["processing_level"] = "raw"
        assert "processing_level" not in correct(ds, ["geometric"]).attrs

    def test_legacy_history_names(self):
        ds = _ds()
        ds.attrs["processing_history"] = json.dumps([{"step": "tvg_correction", "params": {}}])
        assert processing_level(ds) == "tvg"


# ============================================================
# correct_survey / correct_file
# ============================================================

class TestCorrectSurvey:
    def test_every_channel_and_aux_used(self):
        survey = _survey()
        out = correct_survey(survey, progress=None)
        for ch in out.freq_channels.values():
            assert _steps(ch.dataset) == list(DEFAULT_STEPS)
        # input untouched
        assert all(_steps(ch.dataset) == [] for ch in survey.freq_channels.values())

    def test_progress_called(self):
        messages = []
        correct_survey(_survey(), ["tvg"], progress=messages.append)
        assert messages == ["hz_1495000: tvg", "hz_547500: tvg"]


@pytest.fixture
def ss_h5(tmp_path):
    path = tmp_path / "a.ss.h5"
    write_survey(_survey(n_pings=45), path)
    return path


class TestCorrectFile:
    def test_in_place_atomic_write(self, ss_h5):
        written = correct_file(ss_h5)
        assert written == ss_h5
        assert not list(ss_h5.parent.glob("*.tmp"))
        for ch in read_survey(ss_h5).freq_channels.values():
            assert _steps(ch.dataset) == list(DEFAULT_STEPS)

    def test_out_path(self, ss_h5, tmp_path):
        out = tmp_path / "out" / "b.ss.h5"
        assert correct_file(ss_h5, ["tvg"], out_path=out) == out
        assert _steps(read_survey(ss_h5).get_frequency("hz_1495000").dataset) == []
        assert _steps(read_survey(out).get_frequency("hz_1495000").dataset) == ["tvg"]

    def test_already_done_not_rewritten(self, ss_h5):
        correct_file(ss_h5, ["tvg"])
        mtime = ss_h5.stat().st_mtime_ns
        correct_file(ss_h5, ["tvg"])
        assert ss_h5.stat().st_mtime_ns == mtime

    @pytest.mark.parametrize("steps", [
        DEFAULT_STEPS,
        ("crop_farrange", "interpolate_aux", "tvg", "geometric", "range_correction", "georef"),
        ("interpolate_aux", "bottom_track", "tvg", "geometric", "georef"),
    ])
    def test_chunked_equals_unchunked(self, ss_h5, tmp_path, steps):
        params = {"range_correction": {"table": _rc_table()}} if "range_correction" in steps else None
        eager = correct_file(ss_h5, steps, params, out_path=tmp_path / "eager.ss.h5")
        chunked = correct_file(ss_h5, steps, params, out_path=tmp_path / "chunked.ss.h5", chunk_pings=7)
        a, b = read_survey(eager), read_survey(chunked)
        for fk in a.frequencies:
            da, db = a.get_frequency(fk).dataset.load(), b.get_frequency(fk).dataset.load()
            assert set(da.data_vars) == set(db.data_vars)
            for var in da.data_vars:
                np.testing.assert_array_equal(da[var].values, db[var].values, err_msg=var)
            assert _steps(da) == _steps(db)
            for ra, rb in zip(read_history(da), read_history(db), strict=True):
                assert ra["params"] == rb["params"]

    def test_invalid_sound_velocity_warns_once_per_channel_when_chunked(self, tmp_path, caplog):
        survey = _survey(n_pings=45)
        for fk in survey.frequencies:
            survey.get_frequency(fk).dataset["sound_vel_ms"].values[:10] = np.nan
        path = tmp_path / "sv.ss.h5"
        write_survey(survey, path)
        with caplog.at_level("WARNING", logger="sidescanpy"):
            correct_file(path, ["interpolate_aux", "geometric"], out_path=tmp_path / "o.ss.h5", chunk_pings=7)
        msgs = [r.getMessage() for r in caplog.records if "invalid sound_vel_ms" in r.getMessage()]
        assert len(msgs) == len(survey.frequencies)  # once per channel, not once per chunk
        assert all("10 of 45 pings" in m for m in msgs)

    def test_chunked_correct_file_does_not_warn_about_skipped_aux(self, tmp_path, caplog):
        survey = _survey(n_pings=45)
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)
        with caplog.at_level("WARNING", logger="sidescanpy"):
            correct_file(path, ["tvg", "geometric"], out_path=tmp_path / "o.ss.h5", chunk_pings=7)
        assert "interpolate_aux skipped" not in caplog.text  # the file supplies aux

    def test_chunked_crop_detects_on_whole_channel(self, tmp_path):
        survey = _survey(n_pings=30)
        for fk in survey.frequencies:
            ds = survey.get_frequency(fk).dataset
            db = np.linspace(30, -30, 400)
            db[250:] = np.linspace(-30, 0, 150)  # decay then rise -> crop
            ds["backscatter_intensity"].values[...] = (10 ** (db / 20)).astype(np.float32)[None, None, :]
        path = tmp_path / "c.ss.h5"
        write_survey(survey, path)
        eager = correct_file(path, ["crop_farrange"], out_path=tmp_path / "e.ss.h5")
        chunked = correct_file(path, ["crop_farrange"], out_path=tmp_path / "k.ss.h5", chunk_pings=4)
        ne = read_survey(eager).get_frequency("hz_1495000").dataset.sizes["sample"]
        nk = read_survey(chunked).get_frequency("hz_1495000").dataset.sizes["sample"]
        assert ne == nk < 400

    def test_legacy_backscatter_raw_read_then_dropped(self, tmp_path):
        survey = _survey()
        for fk in survey.frequencies:
            ds = survey.get_frequency(fk).dataset
            ds["backscatter_raw"] = ds["backscatter_intensity"].copy()
            ds.attrs["processing_level"] = "raw"
        path = tmp_path / "legacy.ss.h5"
        write_survey(survey, path)
        legacy = read_survey(path)
        assert "backscatter_raw" in legacy.get_frequency("hz_1495000").dataset
        correct_file(path, ["tvg"])
        out = read_survey(path).get_frequency("hz_1495000").dataset
        assert "backscatter_raw" not in out
        assert "processing_level" not in out.attrs
        assert processing_level(out) == "tvg"

    def test_disk_space_check(self, ss_h5, mocker):
        mocker.patch("shutil.disk_usage", return_value=mocker.Mock(free=10))
        with pytest.raises(OSError, match="insufficient free space"):
            correct_file(ss_h5)

    def test_estimate_scales_with_steps(self, ss_h5):
        size = ss_h5.stat().st_size
        assert estimate_correct_disk_bytes(ss_h5, ["crop_farrange"]) == int(size * 1.2)
        assert estimate_correct_disk_bytes(ss_h5, ["tvg", "geometric", "georef"]) > 4 * size
        assert estimate_correct_disk_bytes(ss_h5, ["tvg"], chunk_pings=10) > estimate_correct_disk_bytes(ss_h5, ["tvg"])


def test_preparation_steps_are_not_levels():
    from sidescanpy.core.history import PROCESSING_LEVELS

    assert "crop_farrange" not in PROCESSING_LEVELS
    assert "interpolate_aux" not in PROCESSING_LEVELS
    assert "bottom_track" not in PROCESSING_LEVELS
    assert PROCESSING_LEVELS[0] == "raw"


class TestCorrectionSteps:
    def test_canonical_order_and_aux(self):
        from sidescanpy.processing import correction_steps

        assert correction_steps(["georef", "tvg"]) == ["interpolate_aux", "tvg", "georef"]

    def test_crop_alone_needs_no_aux(self):
        from sidescanpy.processing import correction_steps

        assert correction_steps({"crop_farrange"}) == ["crop_farrange"]

    def test_every_step(self):
        from sidescanpy.core.history import BUILTIN_STEPS
        from sidescanpy.processing import correction_steps

        assert correction_steps(reversed(BUILTIN_STEPS)) == list(BUILTIN_STEPS)

    def test_empty_and_unknown(self):
        from sidescanpy.processing import correction_steps

        assert correction_steps([]) == []
        with pytest.raises(ValueError, match="unknown correction step"):
            correction_steps(["tvgg"])


class TestFrequencyParams:
    """Per-frequency overrides of ``params`` (e.g. absorption per frequency)."""

    def _absorption(self, ds):
        return next(h for h in read_history(ds) if h["step"] == "tvg")["params"]["absorption_db_per_m"]

    def test_correct_survey_applies_each_frequency_override(self):
        out = correct_survey(
            _survey(), ["tvg"], params={"tvg": {"spreading_loss_db": 20.0}},
            frequency_params={"hz_1495000": {"tvg": {"absorption_db_per_m": 0.2}},
                              "hz_547500": {"tvg": {"absorption_db_per_m": 0.1}}},
        )
        assert self._absorption(out.get_frequency("hz_1495000").dataset) == pytest.approx(0.2)
        assert self._absorption(out.get_frequency("hz_547500").dataset) == pytest.approx(0.1)

    def test_override_merges_with_shared_params(self):
        out = correct_survey(
            _survey(), ["tvg"], params={"tvg": {"spreading_loss_db": 30.0}},
            frequency_params={"hz_547500": {"tvg": {"absorption_db_per_m": 0.1}}},
        )
        lf = next(h for h in read_history(out.get_frequency("hz_547500").dataset) if h["step"] == "tvg")["params"]
        assert lf["spreading_loss_db"] == 30.0 and lf["absorption_db_per_m"] == pytest.approx(0.1)
        hf = next(h for h in read_history(out.get_frequency("hz_1495000").dataset) if h["step"] == "tvg")["params"]
        assert hf["spreading_loss_db"] == 30.0 and hf["absorption_db_per_m"] != pytest.approx(0.1)

    def test_unknown_frequency_warns_not_raises(self, caplog):
        with caplog.at_level("WARNING", logger="sidescanpy.processing.correct"):
            correct_survey(_survey(), ["tvg"], frequency_params={"hz_999": {"tvg": {"absorption_db_per_m": 0.1}}})
        assert "hz_999" in caplog.text

    def test_override_for_step_not_requested_raises(self):
        with pytest.raises(ValueError):
            correct_survey(_survey(), ["tvg"], frequency_params={"hz_547500": {"georef": {}}})

    @pytest.mark.parametrize("chunk_pings", [None, 16])
    def test_correct_file_eager_and_chunked(self, ss_h5, chunk_pings):
        correct_file(ss_h5, ["tvg"], frequency_params={"hz_547500": {"tvg": {"absorption_db_per_m": 0.1}}},
                     chunk_pings=chunk_pings)
        survey = read_survey(ss_h5)
        assert self._absorption(survey.get_frequency("hz_547500").dataset) == pytest.approx(0.1)
        assert self._absorption(survey.get_frequency("hz_1495000").dataset) != pytest.approx(0.1)


class TestChunkedWarningsOncePerChannel:
    def test_interpolate_warning_logged_once_per_channel_not_per_chunk(self, tmp_path, caplog):
        survey = _survey(n_pings=45)
        att = survey.attitude.copy()
        att["pitch_deg"] = att["pitch_deg"] * np.nan  # every chunk would warn about it
        survey.attitude = att
        path = tmp_path / "nan_pitch.ss.h5"
        write_survey(survey, path)
        with caplog.at_level("WARNING", logger="sidescanpy.processing.interpolate"):
            correct_file(  # 5 chunks per channel; no per-ping fallback, so pitch warns
                path, ["interpolate_aux", "tvg"], {"interpolate_aux": {"per_ping_fallback": False}}, chunk_pings=10,
            )
        pitch = [r for r in caplog.records if "pitch_deg" in r.getMessage()]
        assert len(pitch) == 2  # once per frequency channel
        # the filter is removed afterwards
        assert not logging.getLogger("sidescanpy.processing.interpolate").filters
