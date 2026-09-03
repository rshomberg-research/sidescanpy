"""Tests for `sidescanpy.investigation.run_investigation()`'s top-level
orchestration -- previously untested directly (only the individual stage
functions in `analysis/investigation.py` and the plot functions in
`viz/investigation.py` had coverage). Added 2026-09-02 alongside the new
`channel` parameter -- a real gap found from a live report: Stage 3/4's
own binning (`mean_backscatter_vs_range`) pools every dimension via
`.ravel()`, including the physical (port/starboard) `channel` dim, so
every `run_investigation()` call before this parameter existed silently
blended both transducers into one fitted curve with nothing in the
output signaling that a blend happened.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.core import SideScanSurvey
from sidescanpy.investigation import run_investigation
from sidescanpy.io.hdf5.writer import write_survey
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.radiometric import TVGParams, absorption_for_frequency, tvg_gain_db


def _make_two_channel_survey_path(tmp_path, *, n_pings=40, n_samples=300):
    """Port and starboard deliberately follow DIFFERENT decay shapes --
    port carries the full uncompensated physics curve (spreading +
    absorption loss), starboard is flat (already fully compensated,
    scale=1.0 everywhere). Pooled together (channel=None), this produces
    neither curve cleanly; separated, each side should classify as its
    own distinct, correct TVG status."""
    n_channels = 2
    time = pd.date_range("2026-01-01", periods=n_pings, freq="1s").values.astype("datetime64[ns]")
    freq_hz = 1_495_000
    base = xr.Dataset(
        {
            "backscatter_raw": (["ping", "channel", "sample"], np.ones((n_pings, n_channels, n_samples), dtype=np.float32)),
            "backscatter_intensity": (["ping", "channel", "sample"], np.ones((n_pings, n_channels, n_samples), dtype=np.float32)),
            "latitude": (["ping"], np.linspace(40.0, 40.01, n_pings)),
            "longitude": (["ping"], np.linspace(-70.0, -69.99, n_pings)),
            "heading_deg": (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
            "pitch_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "depth_m": (["ping"], np.full(n_pings, 25.0, dtype=np.float32)),
            "altitude_m": (["ping"], np.full(n_pings, 5.0, dtype=np.float32)),
            "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0, dtype=np.float32)),
            "fathometer_reading_m": (["ping"], np.full(n_pings, 5.0, dtype=np.float32)),
            "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        },
        coords={
            "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
            "time": (["ping"], time),
            "channel_id": (["channel"], np.array(["port", "starboard"])),
        },
        attrs={
            "freq_hz_start": freq_hz, "freq_hz_end": freq_hz,
            "pulse_length_us": 0.5, "sample_rate_hz": 1000,
            "processing_level": "raw", "processing_history": json.dumps([]),
        },
    )
    ds_geo = compute_geometric_ranges(base)
    ground_range = ds_geo["ground_range_m"].values  # (ping, channel, sample) -- same for both channels here

    alpha = absorption_for_frequency(freq_hz)
    full_physics = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)
    loss_db = tvg_gain_db(np.clip(ground_range, full_physics.min_range_m, None), full_physics)
    ref = np.nanmin(loss_db[np.isfinite(loss_db)])
    port_amplitude = 10.0 ** (-(loss_db - ref) / 20.0)
    port_amplitude = np.nan_to_num(port_amplitude, nan=0.0).astype(np.float32)

    starboard_amplitude = np.where(np.isfinite(ground_range), 1.0, 0.0).astype(np.float32)

    backscatter = base["backscatter_raw"].values.copy()
    backscatter[:, 0, :] = port_amplitude[:, 0, :]
    backscatter[:, 1, :] = starboard_amplitude[:, 1, :]
    base["backscatter_raw"] = (["ping", "channel", "sample"], backscatter)
    base["backscatter_intensity"] = (["ping", "channel", "sample"], backscatter.copy())

    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="two_channel")
    survey.add_frequency("hz_1495000", base)
    path = tmp_path / "two_channel.ss.h5"
    write_survey(survey, path)
    return path


class TestRunInvestigationChannel:
    def test_channel_none_matches_omitting_the_parameter(self, tmp_path):
        """channel=None (default) must be byte-identical to not passing
        the parameter at all -- every existing caller's exact prior
        behavior, unchanged."""
        path = _make_two_channel_survey_path(tmp_path)

        explicit_none = run_investigation(path, channel=None)
        omitted = run_investigation(path)

        assert explicit_none.stages["stage4"].metrics["best_candidate"] == omitted.stages["stage4"].metrics["best_candidate"]
        assert explicit_none.stages["stage4"].metrics["best_rmse_db"] == pytest.approx(omitted.stages["stage4"].metrics["best_rmse_db"])
        assert explicit_none.params["channel"] is None

    def test_channel_int_selects_one_physical_channel(self, tmp_path):
        """channel=0 (port, full uncompensated physics) and channel=1
        (starboard, flat/already-compensated) must classify DIFFERENTLY
        from each other -- proof they were fit independently, not pooled."""
        path = _make_two_channel_survey_path(tmp_path)

        port = run_investigation(path, channel=0)
        starboard = run_investigation(path, channel=1)

        assert port.stages["stage4"].metrics["best_candidate"] == "full_physics"
        assert port.stages["stage4"].classification == "none"  # nothing compensated yet
        assert starboard.stages["stage4"].classification == "full"  # already flat
        assert port.params["channel"] == 0
        assert starboard.params["channel"] == 1

    def test_channel_string_looks_up_channel_id(self, tmp_path):
        path = _make_two_channel_survey_path(tmp_path)

        by_string = run_investigation(path, channel="starboard")
        by_int = run_investigation(path, channel=1)

        assert by_string.stages["stage4"].metrics["best_candidate"] == by_int.stages["stage4"].metrics["best_candidate"]
        assert by_string.params["channel"] == "starboard"

    def test_channel_pooled_differs_from_either_single_channel(self, tmp_path):
        """The whole motivation: channel=None's pooled result must not
        just be a copy of either side's own result -- it's a real, if
        less-informative, blend of both."""
        path = _make_two_channel_survey_path(tmp_path)

        pooled = run_investigation(path)
        port = run_investigation(path, channel=0)
        starboard = run_investigation(path, channel=1)

        pooled_rmse = pooled.stages["stage4"].metrics["best_rmse_db"]
        port_rmse = port.stages["stage4"].metrics["best_rmse_db"]
        starboard_rmse = starboard.stages["stage4"].metrics["best_rmse_db"]
        assert pooled_rmse != pytest.approx(port_rmse)
        assert pooled_rmse != pytest.approx(starboard_rmse)

    def test_invalid_channel_string_raises(self, tmp_path):
        path = _make_two_channel_survey_path(tmp_path)

        with pytest.raises(ValueError, match="not found in channel_id"):
            run_investigation(path, channel="midship")
