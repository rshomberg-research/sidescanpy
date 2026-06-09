"""
Core utilities for SideScan sonar surveys.

Two class levels:
- SideScanFrequencyChannel: single-frequency wrapper around an xr.Dataset
- SideScanSurvey: container for multiple frequency channels and aux streams

Aux streams (navigation, attitude, dvl, sensors) live at the survey level
because they come from the same platform regardless of which sonar frequency
was pinging. They are stored as separate xr.Datasets and can be re-interpolated
onto ping times by downstream processing steps.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ============================================================
# SINGLE-FREQUENCY CHANNEL
# ============================================================

class SideScanFrequencyChannel:
    """
    Wrapper for a single-frequency xr.Dataset.

    Validates the dataset against the SideScan schema on construction.
    """

    _dataset: xr.Dataset
    freq_key: str

    def __init__(self, dataset: xr.Dataset, freq_key: str | None = None) -> None:
        dataset.sidescan.validate()
        self._dataset = dataset
        if freq_key is None:
            freq_key = f"hz_{dataset.attrs['freq_hz_start']}"
        self.freq_key = freq_key

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset

    @property
    def frequency(self) -> float:
        """Base frequency in Hz (freq_hz_start)."""
        return float(self._dataset.attrs.get("freq_hz_start", 0))

    @property
    def n_pings(self) -> int:
        return int(self._dataset.sizes["ping"])

    @property
    def n_channels(self) -> int:
        return int(self._dataset.sizes["channel"])

    @property
    def n_samples(self) -> int:
        return int(self._dataset.sizes.get("sample", 0))

    def __repr__(self) -> str:
        return (
            f"SideScanFrequencyChannel(freq_key='{self.freq_key}', "
            f"pings={self.n_pings}, channels={self.n_channels}, "
            f"samples={self.n_samples})"
        )
    
    # def process_each_channel(self, fn, *args, **kwargs) -> SideScanSurvey:  # TODO: add to docs
    #     """Apply fn(channel.dataset, *args, **kwargs) to each frequency channel,
    #     returning a new survey with replaced channels."""
    #     new = _shallow_copy_survey(self)
    #     for fk, channel in list(self.freq_channels.items()):
    #         new._freq_channels[fk] = SideScanFrequencyChannel(
    #             fn(channel.dataset, *args, **kwargs),
    #             freq_key=fk,
    #         )
    #     return new


# ============================================================
# MULTI-FREQUENCY SURVEY (with aux streams)
# ============================================================

class SideScanSurvey:
    """
    Container for a sidescan sonar survey.

    Holds:
    - One or more SideScanFrequencyChannel datasets (different freqs cannot
      share a single xr.Dataset because n_samples may differ).
    - Optional navigation, attitude, and DVL auxiliary streams (one each).
    - Optional sensor auxiliary streams keyed by sensor_type.

    Aux streams are stored as separate xr.Datasets that pass their respective
    accessor's validate() method (navigation_record, attitude_record,
    dvl_record, sensor_record).
    """

    def __init__(
        self,
        source_files: list[Path],
        source_format: str,
        survey_name: str,
    ) -> None:
        self.source_files = source_files
        self.source_format = source_format
        self.survey_name = survey_name

        self._freq_channels: dict[str, SideScanFrequencyChannel] = {}

        self._navigation: xr.Dataset | None = None
        self._attitude:   xr.Dataset | None = None
        self._dvl:        xr.Dataset | None = None
        self._sensors:    dict[str, xr.Dataset] = {}

    # ----- frequency channels -----

    def add_frequency(self, freq_key: str, dataset: xr.Dataset) -> None:
        """Add (or replace) a frequency channel. Warns on overwrite."""
        if freq_key in self._freq_channels:
            logger.warning(f"Overwriting existing frequency channel: {freq_key}")
        self._freq_channels[freq_key] = SideScanFrequencyChannel(dataset, freq_key)

    def get_frequency(self, freq_key: str) -> SideScanFrequencyChannel:
        if freq_key not in self._freq_channels:
            raise KeyError(
                f"Frequency '{freq_key}' not found. Available: {self.frequencies}"
            )
        return self._freq_channels[freq_key]

    @property
    def frequencies(self) -> list[str]:
        return sorted(self._freq_channels.keys())

    @property
    def freq_channels(self) -> dict[str, SideScanFrequencyChannel]:
        return dict(self._freq_channels)

    @property
    def n_frequencies(self) -> int:
        return len(self._freq_channels)

    @property
    def frequency_hz_list(self) -> list[float]:
        return sorted(ch.frequency for ch in self._freq_channels.values())

    # ----- navigation -----

    @property
    def navigation(self) -> xr.Dataset | None:
        return self._navigation

    @navigation.setter
    def navigation(self, ds: xr.Dataset | None) -> None:
        if ds is None:
            self._navigation = None
            return
        ds.navigation_record.validate()
        if self._navigation is not None:
            logger.warning("Overwriting existing navigation record")
        self._navigation = ds

    @property
    def has_navigation(self) -> bool:
        """True if navigation is set AND has at least one observation."""
        if self._navigation is None:
            return False
        return not self._navigation.navigation_record.is_empty()

    # ----- attitude -----

    @property
    def attitude(self) -> xr.Dataset | None:
        return self._attitude

    @attitude.setter
    def attitude(self, ds: xr.Dataset | None) -> None:
        if ds is None:
            self._attitude = None
            return
        ds.attitude_record.validate()
        if self._attitude is not None:
            logger.warning("Overwriting existing attitude record")
        self._attitude = ds

    @property
    def has_attitude(self) -> bool:
        if self._attitude is None:
            return False
        return not self._attitude.attitude_record.is_empty()

    # ----- dvl -----

    @property
    def dvl(self) -> xr.Dataset | None:
        return self._dvl

    @dvl.setter
    def dvl(self, ds: xr.Dataset | None) -> None:
        if ds is None:
            self._dvl = None
            return
        ds.dvl_record.validate()
        if self._dvl is not None:
            logger.warning("Overwriting existing DVL record")
        self._dvl = ds

    @property
    def has_dvl(self) -> bool:
        if self._dvl is None:
            return False
        return not self._dvl.dvl_record.is_empty()

    # ----- sensors -----

    def add_sensor(self, sensor_type: str, dataset: xr.Dataset) -> None:
        """Add (or replace) a sensor record. Warns on overwrite."""
        dataset.sensor_record.validate()
        if sensor_type in self._sensors:
            logger.warning(f"Overwriting existing sensor record: {sensor_type}")
        self._sensors[sensor_type] = dataset

    def get_sensor(self, sensor_type: str) -> xr.Dataset:
        if sensor_type not in self._sensors:
            raise KeyError(
                f"Sensor '{sensor_type}' not found. Available: {self.sensor_types}"
            )
        return self._sensors[sensor_type]

    @property
    def sensors(self) -> dict[str, xr.Dataset]:
        """Read-only copy of the sensors dict."""
        return dict(self._sensors)

    @property
    def sensor_types(self) -> list[str]:
        return sorted(self._sensors.keys())

    @property
    def n_sensors(self) -> int:
        return len(self._sensors)

    # ----- derived survey-wide properties -----

    @property
    def time_range(self) -> tuple[np.datetime64, np.datetime64] | None:
        """
        (min, max) timestamp across all frequency channels and aux streams.
        Returns None if the survey is completely empty.
        """
        mins: list[np.datetime64] = []
        maxs: list[np.datetime64] = []

        for ch in self._freq_channels.values():
            t = ch.dataset["time"].values
            if t.size > 0:
                mins.append(t.min())
                maxs.append(t.max())

        for ds in (self._navigation, self._attitude, self._dvl):
            if ds is not None and ds.sizes.get("obs", 0) > 0:
                t = ds["time"].values
                mins.append(t.min())
                maxs.append(t.max())

        for ds in self._sensors.values():
            if ds.sizes.get("obs", 0) > 0:
                t = ds["time"].values
                mins.append(t.min())
                maxs.append(t.max())

        if not mins:
            return None
        return (min(mins), max(maxs))

    @property
    def bounding_box(self) -> tuple[float, float, float, float] | None:
        """
        (lat_min, lat_max, lon_min, lon_max).

        Prefers navigation aux stream when present and non-empty; otherwise
        falls back to per-ping latitude/longitude across all frequency channels.
        Returns None if no positional data is available.
        """
        lats: list[np.ndarray] = []
        lons: list[np.ndarray] = []

        if self.has_navigation:
            lats.append(self._navigation["latitude"].values)
            lons.append(self._navigation["longitude"].values)
        else:
            for ch in self._freq_channels.values():
                lat = ch.dataset["latitude"].values
                lon = ch.dataset["longitude"].values
                if lat.size > 0:
                    lats.append(lat)
                    lons.append(lon)

        if not lats:
            return None

        lat_all = np.concatenate(lats)
        lon_all = np.concatenate(lons)

        # Drop NaN (per-ping nav can have unflagged fixes)
        lat_valid = lat_all[~np.isnan(lat_all)]
        lon_valid = lon_all[~np.isnan(lon_all)]

        if lat_valid.size == 0:
            return None

        return (
            float(lat_valid.min()),
            float(lat_valid.max()),
            float(lon_valid.min()),
            float(lon_valid.max()),
        )

    # ----- repr -----

    def __repr__(self) -> str:
        def _aux_count(ds: xr.Dataset | None) -> str:
            if ds is None:
                return "-"
            return str(ds.sizes.get("obs", 0))

        return (
            f"SideScanSurvey(name='{self.survey_name}', "
            f"n_frequencies={self.n_frequencies}, "
            f"nav={_aux_count(self._navigation)}, "
            f"att={_aux_count(self._attitude)}, "
            f"dvl={_aux_count(self._dvl)}, "
            f"sensors={self.n_sensors})"
        )