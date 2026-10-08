"""
Synthetic converted surveys for export-window tests.

``geo_friendly_survey``: 20 pings x 2000 samples, slant range well past
altitude so geometric/georef coverage is real. ``swath_friendly_survey``:
many more pings than samples, so full-swath styles' square native window
(``n_samples`` / ``2 * n_samples`` pings) fits. Both use a tight
~100 m track so gridding stays small.

2026 rshomberg@uri.edu
"""

import numpy as np

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.processing.geometric import compute_geometric_ranges
from tests.conftest import make_minimal_sidescan_dataset


def geo_friendly_survey() -> SideScanSurvey:
    """Two frequencies, 20 pings x 2000 samples, altitude 0.5 m."""
    survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="survey_a")
    for freq_key, freq_hz in (("hz_1495000", 1_495_000), ("hz_547500", 547_500)):
        ds = make_minimal_sidescan_dataset(n_pings=20, n_samples=2000, freq_hz_start=freq_hz)
        ds["altitude_m"] = ("ping", np.full(20, 0.5, dtype=np.float32))
        ds["latitude"] = ("ping", np.linspace(40.0000, 40.0010, 20))
        ds["longitude"] = ("ping", np.linspace(-70.0000, -70.0010, 20))
        survey.add_frequency(freq_key, ds)
    return survey


def swath_friendly_survey(n_pings: int = 300, n_samples: int = 30) -> SideScanSurvey:
    """One frequency, ``n_pings`` x ``n_samples``, altitude 0.01 m.

    Max slant range at the fixture's 400 kHz sample rate is
    ``n_samples * 0.001875`` m, so altitude must stay well under it for any
    sample to leave the water column.
    """
    survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="survey_a")
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples, freq_hz_start=1_495_000)
    ds["altitude_m"] = ("ping", np.full(n_pings, 0.01, dtype=np.float32))
    ds["latitude"] = ("ping", np.linspace(40.0000, 40.0010, n_pings))
    ds["longitude"] = ("ping", np.linspace(-70.0000, -70.0010, n_pings))
    survey.add_frequency("hz_1495000", ds)
    return survey


def geo_channel_ds():
    """One frequency's dataset (20 pings x 2000 samples) with geometric ranges computed."""
    ds = make_minimal_sidescan_dataset(n_pings=20, n_samples=2000, freq_hz_start=1_495_000)
    ds["altitude_m"] = ("ping", np.full(20, 0.5, dtype=np.float32))
    ds["latitude"] = ("ping", np.linspace(40.0000, 40.0010, 20))
    ds["longitude"] = ("ping", np.linspace(-70.0000, -70.0010, 20))
    return compute_geometric_ranges(ds)
