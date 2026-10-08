"""
Survey track: per-ping nadir positions plus strided port/starboard
swath-edge positions, for a multi-file survey.

:func:`build_survey_track` returns a :class:`SurveyTrack` of WGS84
lat/lon arrays; ``sidescanpy.viz.track.plot_track_and_scanlines`` and
``sidescanpy.viz.targets.plot_track_with_targets`` draw it.

2026 rshomberg@uri.edu
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr

from sidescanpy.processing.correct import correct
from sidescanpy.processing.interpolate import interpolate_aux_onto_pings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurveyTrack:
    """Nadir track and scan-line endpoints for one frequency (WGS84 degrees).

    ``track_*`` arrays have one entry per ping across every file, in time
    order. ``scan_*`` arrays are index-aligned port/starboard swath-edge
    positions at every ``scanline_stride``-th ping of each file.
    """

    track_lat: np.ndarray
    track_lon: np.ndarray
    scan_port_lat: np.ndarray
    scan_port_lon: np.ndarray
    scan_stbd_lat: np.ndarray
    scan_stbd_lon: np.ndarray

    @property
    def n_pings(self) -> int:
        return len(self.track_lat)

    @property
    def n_scanlines(self) -> int:
        return len(self.scan_port_lat)


def build_survey_track(
    surveys: dict,
    freq_key: str,
    *,
    scanline_stride: int,
) -> Optional[SurveyTrack]:
    """Nadir track + strided port/starboard scan-line endpoints for one
    frequency, across every file that has it, in time order.

    Cheap by design, but *not* by slicing the sample axis down to one
    sample before georeferencing — that's tempting (we only want the
    farthest-range sample per ping) but wrong:
    ``compute_geometric_ranges`` computes each sample's position from the
    absolute ``sample`` coordinate when present (``processing.sample_axis.
    sample_index``), so slicing samples is safe for datasets that carry it.
    For legacy datasets *without* that coordinate it falls back to a
    *local* index (``arange(ds.sizes["sample"])``): pre-slicing to
    ``sample=[-1]`` then makes the function see "sample 0" (point-blank
    range) instead of the true far-range edge — silently producing NaN
    via the water-column mask instead of a real position.

    The safe optimization instead pre-selects the *strided pings*
    (slicing the ping axis doesn't touch ``ds.sizes["sample"]``, so it
    doesn't affect the sample-index math) while keeping each selected
    ping's full sample axis intact through georeferencing, and only
    slices to the last sample *afterward*. This still avoids
    georeferencing every ping — just the ones actually needed for a
    rung — while staying correct.


    Parameters
    ----------
    surveys : dict[Path, SideScanSurvey]
    freq_key : str
        e.g. ``"hz_1495000"``.
    scanline_stride : int
        Draw a scan-line rung every Nth ping.

    Returns
    -------
    SurveyTrack or None
        None if no file has ``freq_key``. Track arrays cover every ping;
        scan arrays cover every ``scanline_stride``-th ping.
    """
    entries = [(p, s) for p, s in surveys.items() if freq_key in s.frequencies]
    if not entries:
        return None

    entries = [(p, interpolate_aux_onto_pings(s)) for p, s in entries]
    entries.sort(
        key=lambda item: item[1].get_frequency(freq_key).dataset["time"].values[0]
    )

    track_lat_parts, track_lon_parts = [], []
    edge_datasets = []
    for _p, survey in entries:
        ds = survey.get_frequency(freq_key).dataset
        track_lat_parts.append(ds["latitude"].values)
        track_lon_parts.append(ds["longitude"].values)

        strided_idx = np.arange(0, ds.sizes["ping"], scanline_stride)
        edge_ds = ds.isel(ping=strided_idx).load()
        edge_ds = correct(edge_ds, ("geometric", "georef"))
        # Each file's sample axis may be a different length (e.g. one
        # file's channel was cropped further than another's by the
        # far-range noise crop step), so "sample"'s own index label
        # for the retained last sample (e.g. 557 vs. 663) differs
        # across files. Left in place, concatenating along "ping" below
        # aligns on that mismatched "sample" label and NaNs out
        # whichever file doesn't have that particular index value —
        # drop the label since only the value at "the last sample" is
        # wanted here, not which index it came from.
        last_sample = edge_ds.isel(sample=[-1])
        if "sample" in last_sample.coords:
            last_sample = last_sample.drop_vars("sample")
        edge_datasets.append(last_sample)

    track_lat = np.concatenate(track_lat_parts)
    track_lon = np.concatenate(track_lon_parts)

    # Each entry in edge_datasets is already the strided-ping subset for
    # its file (selected before georeferencing, above) — no further
    # striding needed here.
    edge_concat = xr.concat(edge_datasets, dim="ping")
    lat_edge = edge_concat["latitude_per_sample"].values[:, :, 0]   # (ping, channel)
    lon_edge = edge_concat["longitude_per_sample"].values[:, :, 0]

    track = SurveyTrack(
        track_lat=track_lat,
        track_lon=track_lon,
        scan_port_lat=lat_edge[:, 0],
        scan_port_lon=lon_edge[:, 0],
        scan_stbd_lat=lat_edge[:, 1],
        scan_stbd_lon=lon_edge[:, 1],
    )
    logger.debug(f"{freq_key}: survey track with {track.n_pings} pings, {track.n_scanlines} scan line(s)")
    return track
