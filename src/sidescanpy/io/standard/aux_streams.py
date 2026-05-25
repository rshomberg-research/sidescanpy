"""
Read/write auxiliary sensor streams (navigation, attitude, environmental)

Raw data stored at native sensor rates; Gets interpolated to ping times for dataset

2026 rshomberg@uri.edu
"""

# FIXME: this might not work because these streams are not necessarily the same
# rate. May need further breaking up

import logging
from pathlib import Path 

import xarray as xr 
import numpy as np 
from scipy.interpolate import interp1d

from sidescanpy.core.schema import (
    HDF5_GROUP_NAVIGATION,
    HDF5_GROUP_ATTITUDE,
    HDF5_GROUP_ENVIRONMENTAL,
)

logger = logging.getLogger(__name__)


def write_navigation_stream(
        output_path : Path,
        time        : np.ndarray,
        latitude    : np.ndarray,
        longitude   : np.ndarray,
        heading_deg : np.ndarray, 
        speed_kn    : np.ndarray,
        depth_m     : np.ndarray,
) -> None:
    """
    Write raw GPS rate navigation stream to HDF5

    Parameters
    ----------
    output_path : Path
        HDF5 file path
    time : ndarray
        UTC epoch seconds, shape (n_fixes,).
    latitude, longitude, heading_deg, speed_kn, depth_m : ndarray
        Navigation data at GPS rate, shape (n_fixes,)
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "latitude"      : (["time"], latitude),
            "longitude"     : (["time"], longitude),
            "heading_deg"   : (["time"], heading_deg),
            "speed_kn"      : (["time"], speed_kn),
            "depth_m"       : (["time"], depth_m),
        },
        coords={"time"  : (["time"], time)},
    )

    mode = 'w' if not output_path.exists() else 'a' # ???: are these lines needed?
    ds.to_netcdf(
        output_path,
        mode=mode,
        engine='h5netcdf',
        group=HDF5_GROUP_NAVIGATION,
    )
    logger.debug(f"Wrote navigation stream to {output_path}")


def read_navigation_stream(output_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Read raw GPS-rate navigation stream from HDF5

    Returns
    -------
    time : ndarray
        UTC epoch seconds
    data : dict
        Dictionary with keys: latitude, longitude, heading_deg, speed_kn, depth_m
    """
    
    ds = xr.open_dataset(
        output_path,
        engine='h5netcdf',
        group=HDF5_GROUP_NAVIGATION,
    )

    time = ds.coords["time"].values 
    data = {
        "latitude"      : ds["latitude"].values,
        "longitude"     : ds["longitude"].values,
        "heading_deg"   : ds["heading_deg"].values,
        "speed_kn"      : ds["speed_kn"].values,
        "depth_m"       : ds["depth_m"].values,
    }

    logger.debug(f"Read navigation stream from {output_path}")
    return time, data


def write_attitude_stream(
    output_path: Path,
    time: np.ndarray,
    roll_deg: np.ndarray,
    pitch_deg: np.ndarray,
    heave_m: np.ndarray,
    altitude_m: np.ndarray,
) -> None:
    """
    Write raw motion sensor attitude stream to HDF5.

    Parameters
    ----------
    output_path : Path
        HDF5 file path.
    time : ndarray
        UTC epoch seconds, shape (n_samples,).
    roll_deg, pitch_deg, heave_m, altitude_m : ndarray
        Attitude data at sensor rate, shape (n_samples,).
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "roll_deg"      : (["time"], roll_deg),
            "pitch_deg"     : (["time"], pitch_deg),
            "heave_m"       : (["time"], heave_m),
            "altitude_m"    : (["time"], altitude_m),
        },
        coords={"time": (["time"], time)},
    )

    mode = "w" if not output_path.exists() else "a"
    ds.to_netcdf(
        output_path,
        mode=mode,
        engine="h5netcdf",
        group=HDF5_GROUP_ATTITUDE,
    )
    logger.debug(f"Wrote attitude stream to {output_path}")


def read_attitude_stream(output_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Read raw motion sensor attitude stream from HDF5.

    Returns
    -------
    time : ndarray
        UTC epoch seconds.
    data : dict
        Dictionary with keys: roll_deg, pitch_deg, heave_m, altitude_m.
    """
    
    ds = xr.open_dataset(
        output_path,
        engine="h5netcdf",
        group=HDF5_GROUP_ATTITUDE,
    )

    time = ds.coords["time"].values
    data = {
        "roll_deg"      : ds["roll_deg"].values,
        "pitch_deg"     : ds["pitch_deg"].values,
        "heave_m"       : ds["heave_m"].values,
        "altitude_m"    : ds["altitude_m"].values,
    }

    logger.debug(f"Read attitude stream from {output_path}")
    return time, data


def write_environmental_stream(
    output_path: Path,
    time: np.ndarray,
    sound_vel_ms: np.ndarray,
) -> None:
    """
    Write environmental sensor stream to HDF5.

    Parameters
    ----------
    output_path : Path
        HDF5 file path.
    time : ndarray
        UTC epoch seconds, shape (n_measurements,).
    sound_vel_ms : ndarray
        Sound velocity in m/s, shape (n_measurements,).
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "sound_vel_ms"  : (["time"], sound_vel_ms),
        },
        coords={"time"  : (["time"], time)},
    )

    mode = "w" if not output_path.exists() else "a"
    ds.to_netcdf(
        output_path,
        mode=mode,
        engine="h5netcdf",
        group=HDF5_GROUP_ENVIRONMENTAL,
    )
    logger.debug(f"Wrote environmental stream to {output_path}")


def read_environmental_stream(output_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Read environmental sensor stream from HDF5.

    Returns
    -------
    time : ndarray
        UTC epoch seconds.
    data : dict
        Dictionary with keys: sound_vel_ms.
    """

    ds = xr.open_dataset(
        output_path,
        engine="h5netcdf",
        group=HDF5_GROUP_ENVIRONMENTAL,
    )

    time = ds.coords["time"].values
    data = {
        "sound_vel_ms"  : ds["sound_vel_ms"].values,
    }

    logger.debug(f"Read environmental stream from {output_path}")
    return time, data


def interpolate_to_ping_time(
    source_time: np.ndarray,
    source_data: np.ndarray,
    ping_time: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate auxiliary stream data to ping times.

    Parameters
    ----------
    source_time : ndarray
        UTC epoch times of source measurements.
    source_data : ndarray
        Source data values (1D or 2D).
    ping_time : ndarray
        UTC epoch times of pings (interpolation targets).
    method : str
        Interpolation method: "linear", "nearest", "zero", "slinear", "quadratic", "cubic".

    Returns
    -------
    ndarray
        Interpolated data at ping times.
    """
    
    if method == "nearest":
        fill_value = "extrapolate" # handle out-of-bounds gracefully
    else:
        fill_value = np.nan 

    if source_data.ndim == 1:
        f = interp1d(
            source_time,
            source_data,
            kind=method,
            bounds_error=False,
            fill_value=fill_value,
        )
    else:
        raise NotImplementedError("Not sure this ever needs to come up.")
        result = np.empty((ping_time.size, source_data.shape[1]))
        for i in range(source_data.shape[1]):
            f = interp1d(
                source_time,
                source_data[:, i],
                kind=method,
                bounds_error=False,
                fill_value=fill_value,
            )
            result[:, i] = f(ping_time)
        return result

    return f(ping_time)