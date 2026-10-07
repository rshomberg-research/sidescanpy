"""
Unit tests for sidescanpy.processing.farrange — far-range noise
detection and cropping (Known Issue #6).

2026 rshomberg@uri.edu
"""

import json

import numpy as np

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.processing.farrange import (
    crop_far_range_noise,
    detect_far_range_cutoff_sample,
)
from tests.conftest import make_minimal_sidescan_dataset


def _db_to_linear(db: np.ndarray) -> np.ndarray:
    return 10 ** (db / 20.0)


def _make_profile_dataset(db_profile: np.ndarray, n_pings: int = 50) -> "xr.Dataset":  # noqa: F821
    """Build a dataset whose backscatter_raw follows db_profile (shape
    (n_samples,)) identically across every ping/channel, plus small noise."""
    n_samples = len(db_profile)
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples)
    rng = np.random.default_rng(0)
    noise_db = rng.normal(0, 0.3, size=(n_pings, 2, n_samples))
    db = db_profile[None, None, :] + noise_db
    linear = _db_to_linear(db).astype(np.float32)
    ds["backscatter_raw"].values[...] = linear
    ds["backscatter_intensity"].values[...] = linear
    return ds


def _decay_then_rise_profile(n_samples: int, floor_at: int, floor_db: float = -30.0, peak_db: float = 10.0, rise_to_db: float = -10.0) -> np.ndarray:
    """dB profile: linear decay from peak_db to floor_db by floor_at,
    then flat at floor_db, then rises to rise_to_db by the end."""
    profile = np.empty(n_samples)
    decay = np.linspace(peak_db, floor_db, floor_at)
    profile[:floor_at] = decay
    remaining = n_samples - floor_at
    rise = np.linspace(floor_db, rise_to_db, remaining)
    profile[floor_at:] = rise
    return profile


def _clean_decay_profile(n_samples: int, floor_at: int, floor_db: float = -30.0, peak_db: float = 10.0) -> np.ndarray:
    """dB profile: decays to floor_db and stays flat — no rise."""
    profile = np.full(n_samples, floor_db)
    profile[:floor_at] = np.linspace(peak_db, floor_db, floor_at)
    return profile


def _ramp_then_decay_profile(
    n_samples: int, ramp_at: int, decay_at: int,
    ramp_start_db: float = -35.0, peak_db: float = 8.0, floor_db: float = -20.0,
) -> np.ndarray:
    """dB profile with a legitimate near-nadir ramp-up (weak returns very
    close to nadir, lower than the eventual real floor) before reaching
    peak, then a normal decay to floor_db with no rise -- the real shape
    seen on one real channel that originally fooled a
    global-minimum floor search."""
    profile = np.empty(n_samples)
    profile[:ramp_at] = np.linspace(ramp_start_db, peak_db, ramp_at)
    remaining = n_samples - ramp_at
    decay_len = min(decay_at - ramp_at, remaining)
    profile[ramp_at:ramp_at + decay_len] = np.linspace(peak_db, floor_db, decay_len)
    if ramp_at + decay_len < n_samples:
        profile[ramp_at + decay_len:] = floor_db
    return profile


class TestDetectFarRangeCutoffSample:
    def test_no_rise_returns_none(self):
        n_samples = 2000
        profile = _clean_decay_profile(n_samples, floor_at=800)
        ds = _make_profile_dataset(profile)
        assert detect_far_range_cutoff_sample(ds) is None

    def test_rise_past_floor_detected(self):
        n_samples = 2000
        floor_at = 800
        profile = _decay_then_rise_profile(n_samples, floor_at=floor_at)
        ds = _make_profile_dataset(profile)
        cutoff = detect_far_range_cutoff_sample(ds)
        assert cutoff is not None
        # cutoff should land near the floor point, well before the end,
        # and strictly less than n_samples (something was actually cropped)
        assert floor_at * 0.5 < cutoff < n_samples

    def test_small_rise_below_threshold_not_detected(self):
        n_samples = 2000
        floor_at = 800
        # Rise of only ~1dB, below the default 3dB threshold
        profile = _decay_then_rise_profile(n_samples, floor_at=floor_at, floor_db=-30.0, rise_to_db=-29.0)
        ds = _make_profile_dataset(profile)
        assert detect_far_range_cutoff_sample(ds) is None

    def test_large_rise_above_threshold_detected_with_custom_threshold(self):
        n_samples = 2000
        floor_at = 800
        profile = _decay_then_rise_profile(n_samples, floor_at=floor_at, floor_db=-30.0, rise_to_db=-25.0)
        # Default 3dB threshold: rise of 5dB should be detected
        cutoff = detect_far_range_cutoff_sample(_make_profile_dataset(profile))
        assert cutoff is not None
        # With a stricter 10dB threshold, the same 5dB rise should NOT be detected
        cutoff_strict = detect_far_range_cutoff_sample(
            _make_profile_dataset(profile), rise_threshold_db=10.0
        )
        assert cutoff_strict is None

    def test_near_nadir_ramp_up_not_mistaken_for_the_floor(self):
        # Regression test for a real bug found on real data: a legitimate near-nadir
        # ramp-up (weak returns very close to nadir, ordinary geometry/
        # beam-pattern behavior, lower than the eventual real far-range
        # floor) was picked as "the floor" by a global-minimum search --
        # the normal rise from there up to the real peak then read as a
        # huge spurious "rise," cropping almost the entire array on a
        # channel with no far-range noise problem at all.
        n_samples = 2000
        profile = _ramp_then_decay_profile(n_samples, ramp_at=100, decay_at=900)
        ds = _make_profile_dataset(profile)
        assert detect_far_range_cutoff_sample(ds) is None

    def test_leading_floor_clip_run_not_mistaken_for_the_floor(self):
        # Regression test for a real bug found 2026-08-06 on real data
        # (unknown/20241017181822.ss.h5, hz_557500): Known Issue #5's
        # leading run of exact-zero raw amplitude (blanking/ring-down)
        # converts to exactly -120dB, which is *lower* than the real
        # far-range floor -- left unmasked, the bin-minimum search picks
        # this artifact as "the floor," and the real (much stronger)
        # near-nadir signal in the very next bin then reads as a huge
        # spurious "rise," producing a false-positive crop right at the
        # start of a channel that has no far-range noise problem at all.
        n_samples = 2000
        profile = _clean_decay_profile(n_samples, floor_at=800)  # no rise anywhere
        ds = _make_profile_dataset(profile, n_pings=50)
        # Inject a leading exact-zero run (~half of the first bin) on
        # every ping/channel, matching the real leading-blanking-run shape.
        bin_size = n_samples // 40
        blank_len = bin_size // 2
        ds["backscatter_raw"].values[:, :, :blank_len] = 0.0
        ds["backscatter_intensity"].values[:, :, :blank_len] = 0.0

        assert detect_far_range_cutoff_sample(ds) is None

    def test_too_few_samples_returns_none(self):
        # n_samples=2 with bin_size floored at 1 gives only 2 bins (< 3
        # required), regardless of n_bins requested.
        ds = make_minimal_sidescan_dataset(n_pings=10, n_samples=2)
        assert detect_far_range_cutoff_sample(ds, n_bins=40) is None


class TestCropFarRangeNoise:
    def test_cropped_channel_has_reduced_samples(self):
        n_samples = 2000
        floor_at = 800
        profile = _decay_then_rise_profile(n_samples, floor_at=floor_at)
        ds = _make_profile_dataset(profile)
        ds.attrs["freq_hz_start"] = 1_000_000
        ds.attrs["freq_hz_end"] = 1_000_000

        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey.add_frequency("hz_1000000", ds)

        survey, cutoffs = crop_far_range_noise(survey)

        assert cutoffs["hz_1000000"] is not None
        new_n_samples = survey.get_frequency("hz_1000000").n_samples
        assert new_n_samples == cutoffs["hz_1000000"]
        assert new_n_samples < n_samples

    def test_clean_channel_left_unchanged(self):
        n_samples = 2000
        profile = _clean_decay_profile(n_samples, floor_at=800)
        ds = _make_profile_dataset(profile)
        ds.attrs["freq_hz_start"] = 1_000_000
        ds.attrs["freq_hz_end"] = 1_000_000

        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey.add_frequency("hz_1000000", ds)

        survey, cutoffs = crop_far_range_noise(survey)

        assert cutoffs["hz_1000000"] is None
        assert survey.get_frequency("hz_1000000").n_samples == n_samples

    def test_clean_channel_still_gets_a_history_marker(self):
        # A channel needing no crop must still record that detection
        # ran (params={"cropped": False}) -- otherwise a caller checking
        # "has this already been through crop_far_range_noise?" (e.g. a
        # pipeline script's skip-on-rerun check) can't tell "never
        # checked" apart from "checked, nothing to crop," and keeps
        # re-scanning a channel that will never need cropping.
        n_samples = 2000
        profile = _clean_decay_profile(n_samples, floor_at=800)
        ds = _make_profile_dataset(profile)
        ds.attrs["freq_hz_start"] = 1_000_000
        ds.attrs["freq_hz_end"] = 1_000_000

        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey.add_frequency("hz_1000000", ds)

        survey, _cutoffs = crop_far_range_noise(survey)

        history = json.loads(survey.get_frequency("hz_1000000").dataset.attrs["processing_history"])
        matching = [h for h in history if h["step"] == "crop_far_range_noise"]
        assert len(matching) == 1
        assert matching[0]["params"]["cropped"] is False

    def test_mixed_survey_crops_only_the_affected_channel(self):
        n_samples = 2000
        dirty_ds = _make_profile_dataset(_decay_then_rise_profile(n_samples, floor_at=800))
        dirty_ds.attrs["freq_hz_start"] = 1_495_000
        dirty_ds.attrs["freq_hz_end"] = 1_495_000

        clean_ds = _make_profile_dataset(_clean_decay_profile(n_samples, floor_at=800))
        clean_ds.attrs["freq_hz_start"] = 557_500
        clean_ds.attrs["freq_hz_end"] = 557_500

        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey.add_frequency("hz_1495000", dirty_ds)
        survey.add_frequency("hz_557500", clean_ds)

        survey, cutoffs = crop_far_range_noise(survey)

        assert cutoffs["hz_1495000"] is not None
        assert cutoffs["hz_557500"] is None
        assert survey.get_frequency("hz_1495000").n_samples < n_samples
        assert survey.get_frequency("hz_557500").n_samples == n_samples

    def test_processing_history_records_the_crop(self):
        n_samples = 2000
        profile = _decay_then_rise_profile(n_samples, floor_at=800)
        ds = _make_profile_dataset(profile)
        ds.attrs["freq_hz_start"] = 1_000_000
        ds.attrs["freq_hz_end"] = 1_000_000

        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey.add_frequency("hz_1000000", ds)

        survey, _cutoffs = crop_far_range_noise(survey)

        history = json.loads(survey.get_frequency("hz_1000000").dataset.attrs["processing_history"])
        steps = [h["step"] for h in history]
        assert "crop_far_range_noise" in steps

    def test_processing_level_unchanged(self):
        n_samples = 2000
        profile = _decay_then_rise_profile(n_samples, floor_at=800)
        ds = _make_profile_dataset(profile)
        ds.attrs["freq_hz_start"] = 1_000_000
        ds.attrs["freq_hz_end"] = 1_000_000
        assert ds.attrs["processing_level"] == "raw"

        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey.add_frequency("hz_1000000", ds)

        survey, _cutoffs = crop_far_range_noise(survey)

        assert survey.get_frequency("hz_1000000").dataset.attrs["processing_level"] == "raw"
