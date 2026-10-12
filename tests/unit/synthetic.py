"""
Synthetic sidescan datasets shared by the analysis and processing unit tests.
"""

import numpy as np
import pandas as pd
import xarray as xr


def make_raw_dataset(
    *,
    n_pings: int = 20,
    n_samples: int = 200,
    sample_rate_hz: float = 1000.0,
    backscatter: np.ndarray | None = None,
    freq_hz: float = 200_000,
) -> xr.Dataset:
    """Minimal valid raw sidescan dataset (uncorrected, two channels)."""
    if backscatter is None:
        backscatter = np.ones((n_pings, 2, n_samples), dtype=np.float32)
    else:
        n_pings, _, n_samples = backscatter.shape

    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(n_samples),
        "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
        "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
        "channel_id": (["channel"], ["port", "starboard"]),
    }

    data_vars = {
        "backscatter_intensity": (["ping", "channel", "sample"], backscatter.astype(np.float32)),
        "latitude": (["ping"], np.full(n_pings, 41.5, dtype=np.float32)),
        "longitude": (["ping"], np.full(n_pings, -71.5, dtype=np.float32)),
        "heading_deg": (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
        "pitch_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "depth_m": (["ping"], np.full(n_pings, 50.0, dtype=np.float32)),
        "altitude_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        "fathometer_reading_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0, dtype=np.float32)),
    }
    attrs = {
        "freq_hz_start": freq_hz,
        "freq_hz_end": freq_hz,
        "sample_rate_hz": sample_rate_hz,
        "pulse_length_us": 100.0,
        "processing_level": "raw",
        "processing_history": "[]",
    }
    return xr.Dataset(data_vars, coords, attrs)


# ============================================================
# SAMPLE-AND-HOLD NAVIGATION / ATTITUDE
# ============================================================

_M_PER_DEG_LAT = 111_320.0


def _held(update_times: np.ndarray, update_values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Value of the latest update at or before each of ``at`` (sample-and-hold)."""
    idx = np.searchsorted(update_times, at, side="right") - 1
    return update_values[np.clip(idx, 0, None)]


def _jittered_schedule(rng, t_end: float, mean_s: float, jitter: float) -> np.ndarray:
    """Sensor update times from 0 to ``t_end`` with irregular intervals."""
    gaps = rng.uniform(mean_s * (1 - jitter), mean_s * (1 + jitter), int(t_end / mean_s * 1.5) + 10)
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    return times[times <= t_end]


def make_held_aux_scenario(
    *,
    duration_s: float = 120.0,
    ping_hz: float = 10.0,
    n_samples: int = 500,
    speed_ms: float = 2.0,
    stop_s: tuple[float, float] = (70.0, 85.0),
    gps_dropout_s: tuple[float, float] = (95.0, 100.0),
    seed: int = 0,
) -> dict:
    """A survey whose navigation and attitude are logged sample-and-hold.

    Truth: a platform at ``speed_ms`` on heading 340 deg, turning at 4 deg/s
    through north to 20 deg (40-50 s), stopping during ``stop_s``, then
    continuing. Pings at ``ping_hz``. A ~1 Hz GPS with irregular update
    intervals and no updates during ``gps_dropout_s``; heading (~2 Hz),
    roll (~5 Hz) and pitch (~3 Hz) on their own irregular schedules.

    The navigation record has one entry per ping holding the latest fix,
    plus a duplicated timestamp on every third ping (first the value just
    before the latest update, then the current one) and a few NaN entries.
    The attitude record has one entry per ping. The per-ping fields on the
    channel hold the same sample-and-hold values, as a converter writes
    them.

    Returns
    -------
    dict
        ``survey`` (SideScanSurvey), ``freq_key``, ``t`` (ping seconds),
        truth arrays ``lat``, ``lon``, ``heading``, ``roll``, ``pitch`` at the
        ping times, and boolean masks ``moving``, ``stopped``, ``dropout``.
    """
    from pathlib import Path

    from sidescanpy.core.survey import SideScanSurvey

    rng = np.random.default_rng(seed)
    t0 = np.datetime64("2026-01-01T00:00:00", "ns")
    lat0, lon0 = 41.5, -71.5
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.deg2rad(lat0))

    # Truth on a fine (1 ms) grid, integrated from speed and heading.
    fine = np.arange(0.0, duration_s + 1.0, 0.001)
    heading = np.where(fine < 40.0, 340.0, np.where(fine < 50.0, 340.0 + 4.0 * (fine - 40.0), 380.0)) % 360.0
    speed = np.where((fine >= stop_s[0]) & (fine < stop_s[1]), 0.0, speed_ms)
    hdg_rad = np.deg2rad(heading)
    north = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.cos(hdg_rad[:-1]) * 0.001)])
    east = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.sin(hdg_rad[:-1]) * 0.001)])

    def lat_at(tt):
        return lat0 + np.interp(tt, fine, north) / _M_PER_DEG_LAT

    def lon_at(tt):
        return lon0 + np.interp(tt, fine, east) / m_per_deg_lon

    def heading_at(tt):
        unwrapped = np.where(tt < 40.0, 340.0, np.where(tt < 50.0, 340.0 + 4.0 * (tt - 40.0), 380.0))
        return unwrapped % 360.0

    def roll_at(tt):
        return 3.0 * np.sin(2 * np.pi * tt / 8.0)

    def pitch_at(tt):
        return 2.0 * np.sin(2 * np.pi * tt / 11.0 + 0.3)

    n_pings = int(duration_s * ping_hz)
    t = np.arange(n_pings) / ping_hz

    # Sensor schedules.
    gps = _jittered_schedule(rng, duration_s, 1.0, 0.25)
    gps = gps[(gps < gps_dropout_s[0]) | (gps >= gps_dropout_s[1])]
    gps_lat, gps_lon = lat_at(gps), lon_at(gps)
    hdg_u = _jittered_schedule(rng, duration_s, 0.5, 0.3)
    roll_u = _jittered_schedule(rng, duration_s, 0.2, 0.3)
    pitch_u = _jittered_schedule(rng, duration_s, 1 / 3, 0.3)

    held_lat, held_lon = _held(gps, gps_lat, t), _held(gps, gps_lon, t)
    held_hdg = _held(hdg_u, heading_at(hdg_u), t)
    held_roll = _held(roll_u, roll_at(roll_u), t)
    held_pitch = _held(pitch_u, pitch_at(pitch_u), t)

    # Navigation record: one entry per ping, every 3rd ping logged twice at
    # the same time (stale value first), and a few invalid (NaN) entries.
    nav_t, nav_lat, nav_lon = [], [], []
    for k, tk in enumerate(t):
        if k % 3 == 0 and k > 0:
            stale = tk - 0.5 / ping_hz
            nav_t.append(tk)
            nav_lat.append(_held(gps, gps_lat, np.array([stale]))[0])
            nav_lon.append(_held(gps, gps_lon, np.array([stale]))[0])
        nav_t.append(tk)
        nav_lat.append(held_lat[k])
        nav_lon.append(held_lon[k])
    nav_lat, nav_lon = np.array(nav_lat), np.array(nav_lon)
    bad = rng.choice(np.arange(10, len(nav_lat) - 10), 15, replace=False)
    nav_lat[bad] = np.nan
    nav_lon[bad] = np.nan
    nav_times = t0 + (np.array(nav_t) * 1e9).round().astype("timedelta64[ns]")
    ping_times = t0 + (t * 1e9).round().astype("timedelta64[ns]")

    navigation = xr.Dataset(
        {"latitude": (("obs",), nav_lat), "longitude": (("obs",), nav_lon)},
        coords={"time": (("obs",), nav_times)},
        attrs={"source": "synthetic_held"},
    )
    attitude = xr.Dataset(
        {
            "heading_deg": (("obs",), held_hdg.astype(np.float32)),
            "roll_deg": (("obs",), held_roll.astype(np.float32)),
            "pitch_deg": (("obs",), held_pitch.astype(np.float32)),
            "heave_m": (("obs",), np.zeros(n_pings, dtype=np.float32)),
        },
        coords={"time": (("obs",), ping_times)},
        attrs={"source": "synthetic_held"},
    )

    ds = make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
    ds = ds.assign_coords(time=("ping", ping_times))
    ds = ds.assign(
        latitude=("ping", held_lat),
        longitude=("ping", held_lon),
        heading_deg=("ping", held_hdg.astype(np.float32)),
        roll_deg=("ping", held_roll.astype(np.float32)),
        pitch_deg=("ping", held_pitch.astype(np.float32)),
    )
    freq_key = "hz_200000"
    survey = SideScanSurvey(source_files=[Path("synthetic.sdf")], source_format="SDF", survey_name="held")
    survey.add_frequency(freq_key, ds)
    survey.navigation = navigation
    survey.attitude = attitude

    stopped = (t >= stop_s[0]) & (t < stop_s[1])
    dropout = (t >= gps_dropout_s[0] - 1.5) & (t < gps_dropout_s[1] + 1.5)
    edges = (t < 2.0) | (t > duration_s - 2.0)
    moving = ~stopped & ~dropout & ~edges & ~((t >= stop_s[0] - 1.5) & (t < stop_s[1] + 1.5))
    return {
        "survey": survey,
        "freq_key": freq_key,
        "t": t,
        "lat": lat_at(t),
        "lon": lon_at(t),
        "heading": heading_at(t),
        "roll": roll_at(t),
        "pitch": pitch_at(t),
        "moving": moving,
        "stopped": stopped,
        "dropout": dropout,
        "m_per_deg_lon": m_per_deg_lon,
    }


def make_split_update_scenario(
    *,
    duration_s: float = 60.0,
    log_hz: float = 14.5,
    update_hz: float = 3.5,
    speed_ms: float = 1.5,
    east_drift_ms: float = 0.01,
    n_samples: int = 200,
    seed: int = 0,
) -> dict:
    """A north-south line whose GPS updates each arrive over two log entries.

    Truth: a platform at ``speed_ms`` due north with a slow eastward drift
    (``east_drift_ms``), so longitude barely changes. Navigation is logged
    sample-and-hold once per ping at ``log_hz``. Real updates come at
    ``update_hz`` with a few percent of jitter (never fewer than four log
    entries apart). Each update arrives over two consecutive entries: the
    first entry after the update carries the new position off by 2-5 cm in
    latitude, the next entry carries the exact position. Both coordinates
    are quantized to 1e-7 deg (about 1 cm). Inter-change gaps are therefore
    bimodal: 1 entry (the follow-up), then 3-4 entries (to the next
    update).

    Returns
    -------
    dict
        ``survey`` (SideScanSurvey; navigation record and per-ping fields hold
        the logged values, attitude is a constant heading 0), ``freq_key``,
        ``t`` (ping seconds), ``lat``/``lon`` truth at the pings,
        ``update_t`` (real update times, s), ``nav`` (the navigation record),
        ``inner`` (pings away from the record ends) and ``m_per_deg_lon``.
    """
    from pathlib import Path

    from sidescanpy.core.survey import SideScanSurvey

    rng = np.random.default_rng(seed)
    t0 = np.datetime64("2026-01-01T00:00:00", "ns")
    lat0, lon0 = 41.5, -71.5
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.deg2rad(lat0))
    quantum = 1e-7

    def lat_at(tt):
        return lat0 + speed_ms * tt / _M_PER_DEG_LAT

    def lon_at(tt):
        return lon0 + east_drift_ms * tt / m_per_deg_lon

    n = int(duration_s * log_hz)
    t = np.arange(n) / log_hz
    upd = _jittered_schedule(rng, duration_s - 0.5, 1.0 / update_hz, 0.025)
    assert np.diff(upd).min() >= 4.0 / log_hz

    offset_m = rng.uniform(0.02, 0.05, len(upd)) * rng.choice([-1.0, 1.0], len(upd))
    offset_m[0] = 0.0                                       # the log starts on a settled fix
    k = np.searchsorted(upd, t, side="right") - 1          # latest update per entry
    first_entry = np.searchsorted(t, upd)                   # first entry seeing each update
    is_first = first_entry[k] == np.arange(n)
    lat = lat_at(upd[k]) + np.where(is_first, offset_m[k], 0.0) / _M_PER_DEG_LAT
    lon = lon_at(upd[k])
    lat = np.round(lat / quantum) * quantum
    lon = np.round(lon / quantum) * quantum

    times = t0 + (t * 1e9).round().astype("timedelta64[ns]")
    navigation = xr.Dataset(
        {"latitude": (("obs",), lat), "longitude": (("obs",), lon)},
        coords={"time": (("obs",), times)},
        attrs={"source": "synthetic_split_update"},
    )
    zeros = np.zeros(n, dtype=np.float32)
    attitude = xr.Dataset(
        {"heading_deg": (("obs",), zeros), "roll_deg": (("obs",), zeros),
         "pitch_deg": (("obs",), zeros), "heave_m": (("obs",), zeros)},
        coords={"time": (("obs",), times)},
        attrs={"source": "synthetic_split_update"},
    )
    ds = make_raw_dataset(n_pings=n, n_samples=n_samples)
    ds = ds.assign_coords(time=("ping", times))
    ds = ds.assign(
        latitude=("ping", lat), longitude=("ping", lon),
        heading_deg=("ping", zeros), roll_deg=("ping", zeros), pitch_deg=("ping", zeros),
    )
    freq_key = "hz_200000"
    survey = SideScanSurvey(source_files=[Path("synthetic.sdf")], source_format="SDF", survey_name="split")
    survey.add_frequency(freq_key, ds)
    survey.navigation = navigation
    survey.attitude = attitude
    return {
        "survey": survey,
        "freq_key": freq_key,
        "t": t,
        "lat": lat_at(t),
        "lon": lon_at(t),
        "update_t": upd,
        "nav": navigation,
        "inner": (t > 1.0) & (t < duration_s - 1.0),
        "m_per_deg_lon": m_per_deg_lon,
    }
