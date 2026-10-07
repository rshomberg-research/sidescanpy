"""
Shared test fixtures for sidescanpy tests.

Builders (module-level functions) are pure — call them outside fixtures
when you need custom parameters. Fixtures (decorated) wrap builders with
sensible defaults.

2026 rshomberg@uri.edu
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# ============================================================
# BUILDERS — pure functions, no pytest decoration
# ============================================================

def _make_time_array(n: int, start: str = "2026-01-01", step_sec: float = 1.0) -> np.ndarray:
    """Build n-element datetime64[ns] array."""
    return pd.date_range(start, periods=n, freq=f"{step_sec}s").values.astype("datetime64[ns]")

def make_minimal_sidescan_dataset(
    n_pings: int = 10,
    n_samples: int = 100,
    n_channels: int = 2,
    backscatter_val: float | None = None,
    freq_hz_start: int = 1_495_000,
    freq_hz_end: int | None = None,
) -> xr.Dataset:
    """
    Build a minimal valid SideScan frequency channel dataset.

    Passes SideScanAccessor.validate(). Use as a base for tests that need
    schema-valid input.
    """
    if freq_hz_end is None:
        freq_hz_end = freq_hz_start

    if backscatter_val is None:
        backscatter_data = np.random.rand(n_pings, n_channels, n_samples).astype(np.float32)
    else:
        backscatter_data = np.full(
            (n_pings, n_channels, n_samples), backscatter_val, dtype=np.float32
        )

    backscatter_raw = xr.DataArray(
        backscatter_data,
        dims=["ping", "channel", "sample"],
    )

    return xr.Dataset(
        {
            "backscatter_raw": backscatter_raw,
            "backscatter_intensity": backscatter_raw.copy(),
            "latitude":     (["ping"], np.linspace(40.0, 40.5, n_pings)),
            "longitude":    (["ping"], np.linspace(-70.0, -69.9, n_pings)),
            "heading_deg":  (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
            "depth_m":      (["ping"], np.full(n_pings, 25.0, dtype=np.float32)),
            "roll_deg":     (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "pitch_deg":    (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "heave_m":      (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "altitude_m":   (["ping"], np.full(n_pings, 5.0, dtype=np.float32)),
            "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0, dtype=np.float32)),
            "fathometer_reading_m": (["ping"], np.full(n_pings, 5.0, dtype=np.float32)),
        },
        coords={
            "ping_number": (["ping"], np.arange(1, n_pings + 1, dtype=np.int32)),
            "time":        (["ping"], _make_time_array(n_pings)),
            "channel_id":  (["channel"], np.array(["port", "starboard"])),
        },
        attrs={
            "freq_hz_start":      freq_hz_start,
            "freq_hz_end":        freq_hz_end,
            "pulse_length_us":    0.5,
            "sample_rate_hz":     400_000,
            "processing_level":   "raw",
            "processing_history": json.dumps([]),
        },
    )


def make_minimal_navigation_record(n_obs: int = 10) -> xr.Dataset:
    """Minimal valid navigation record. Passes NavigationRecordAccessor.validate()."""
    return xr.Dataset(
        data_vars={
            "latitude":  (("obs",), np.linspace(41.0, 41.001, n_obs)),
            "longitude": (("obs",), np.linspace(-71.0, -71.001, n_obs)),
        },
        coords={"time": (("obs",), _make_time_array(n_obs))},
        attrs={"source": "nmea_rmc"},
    )


def make_minimal_attitude_record(n_obs: int = 10) -> xr.Dataset:
    """Minimal valid attitude record. Passes AttitudeRecordAccessor.validate()."""
    return xr.Dataset(
        data_vars={
            "roll_deg":  (("obs",), np.zeros(n_obs, dtype=np.float32)),
            "pitch_deg": (("obs",), np.zeros(n_obs, dtype=np.float32)),
            "heave_m":   (("obs",), np.zeros(n_obs, dtype=np.float32)),
        },
        coords={"time": (("obs",), _make_time_array(n_obs))},
        attrs={"source": "pitch_roll_data_message"},
    )


def make_minimal_dvl_record(n_obs: int = 10) -> xr.Dataset:
    """Minimal valid DVL record. Passes DVLRecordAccessor.validate()."""
    return xr.Dataset(
        data_vars={
            "altitude_m": (("obs",), np.full(n_obs, 5.0, dtype=np.float32)),
        },
        coords={"time": (("obs",), _make_time_array(n_obs))},
        attrs={"source": "doppler_velocity_log_data_message"},
    )


def make_minimal_sensor_record(
    n_obs: int = 10,
    sensor_type: str = "pressure",
    var_name: str = "depth_m",
) -> xr.Dataset:
    """Minimal valid generic sensor record. Passes SensorRecordAccessor.validate()."""
    return xr.Dataset(
        data_vars={
            var_name: (("obs",), np.linspace(10.0, 12.0, n_obs, dtype=np.float32)),
        },
        coords={"time": (("obs",), _make_time_array(n_obs))},
        attrs={"source": "pressure_sensor_reading_message", "sensor_type": sensor_type},
    )


# ============================================================
# FIXTURES — wrappers around builders
# ============================================================

@pytest.fixture
def sample_dataset(request):
    """
    Valid frequency channel dataset matching .ss.h5 schema.

    Default: 10 pings × 2 channels × 100 samples. Override via
    `@pytest.mark.parametrize("sample_dataset", [{"n_pings": 1000}], indirect=True)`.
    """
    params = getattr(request, "param", {}) or {}
    return make_minimal_sidescan_dataset(**params)


@pytest.fixture
def sample_navigation():
    return make_minimal_navigation_record()


@pytest.fixture
def sample_attitude():
    return make_minimal_attitude_record()


@pytest.fixture
def sample_dvl():
    return make_minimal_dvl_record()


@pytest.fixture
def sample_sensor():
    return make_minimal_sensor_record()


@pytest.fixture
def empty_navigation():
    return make_minimal_navigation_record(n_obs=0)


@pytest.fixture
def empty_attitude():
    return make_minimal_attitude_record(n_obs=0)


@pytest.fixture
def empty_dvl():
    return make_minimal_dvl_record(n_obs=0)


@pytest.fixture
def empty_sensor():
    return make_minimal_sensor_record(n_obs=0)


# Survey fixtures — defer import to avoid circular issues at collection time

@pytest.fixture
def sample_survey_channel(sample_dataset):
    from sidescanpy.core import SideScanFrequencyChannel
    return SideScanFrequencyChannel(sample_dataset)


@pytest.fixture
def sample_multi_frequency_survey(sample_dataset):
    from sidescanpy.core import SideScanSurvey

    survey = SideScanSurvey(
        source_files=["sample_survey.jsf"],
        source_format="JSF",
        survey_name="sample_survey",
    )

    # Two distinct frequency channels — set attrs, don't add data vars
    ds1 = sample_dataset.copy()
    ds1.attrs["freq_hz_start"] = 1_495_000
    ds1.attrs["freq_hz_end"]   = 1_495_000

    ds2 = sample_dataset.copy()
    ds2.attrs["freq_hz_start"] = 547_500
    ds2.attrs["freq_hz_end"]   = 547_500

    survey.add_frequency("hz_1495000", ds1)
    survey.add_frequency("hz_547500", ds2)
    return survey

@pytest.fixture
def sample_full_survey(sample_multi_frequency_survey, sample_navigation,
                      sample_attitude, sample_dvl, sample_sensor):
    s = sample_multi_frequency_survey
    s.navigation = sample_navigation
    s.attitude   = sample_attitude
    s.dvl        = sample_dvl
    s.add_sensor("pressure", sample_sensor)
    return s

@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for file I/O tests."""
    return tmp_path


@pytest.fixture
def sample_sensor_temperature():
    return make_minimal_sensor_record(sensor_type="temperature", var_name="temp_c")

# ============================================================
# PYTEST CONFIG
# ============================================================

def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests (integration, etc.)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="Slow test - use --slow to run")
    for item in items:
        if "slow" in item.keywords or "integration" in item.keywords:
            item.add_marker(skip_slow)
