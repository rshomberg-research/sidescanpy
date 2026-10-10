"""
Native slant-range windows that are square in metres.

Sidescan pings are often much further apart along track than samples are
across track (e.g. 24 cm between pings, 2.4 cm between samples). Resampling
such data onto a fine metric grid repeats or smears pings; cropping a square
of native pixels squashes shapes by the same ratio. The windows here cut a
region that is square (chip) or a fixed length (tile) *in metres* from the
native ping x sample grid and resize it with linear interpolation, so the
native detail is kept and shapes have their true proportions.

- :func:`native_chip` -- one side, centred on a target, ``size_m`` square.
- :func:`swath_tiles` -- the whole survey as consecutive dual-swath tiles
  (port mirrored on the left, starboard on the right), e.g. for annotating
  every object.
- :class:`NativeWindow` -- where an image sits in native (ping, sample,
  channel) space, with conversions both ways, so boxes drawn on an image
  can be turned back into sonar coordinates and re-cut at another scale.

Ping and sample arguments are positions along the dataset's ``ping`` and
``sample`` axes. Values are taken from ``value_var`` as stored (dB after the
``tvg`` step); convert to an image with :func:`sidescanpy.export.image.to_uint8`.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

from sidescanpy.export.windows.levels import nominal_sample_interval_m

logger = logging.getLogger(__name__)

_EARTH_M_PER_DEG = 111_320.0


def along_track_spacing_m(ds: xr.Dataset) -> float:
    """Mean distance between consecutive pings (m), from the per-ping positions.

    Repeated navigation fixes (several pings sharing one fix) do not bias the
    result: the summed track length is divided by the number of ping steps.

    Raises
    ------
    ValueError
        Fewer than two pings with a position, or a stationary track.
    """
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        raise ValueError("along_track_spacing_m needs at least two pings with a position")
    idx = np.nonzero(ok)[0]
    lat, lon = lat[ok], lon[ok]
    k = np.cos(np.radians(np.nanmean(lat)))
    dist = np.hypot(np.diff(lat) * _EARTH_M_PER_DEG, np.diff(lon) * _EARTH_M_PER_DEG * k)
    steps = idx[-1] - idx[0]
    total = float(dist.sum())
    if total <= 0:
        raise ValueError("along_track_spacing_m: the track does not move")
    return total / steps


@dataclass(frozen=True)
class NativeWindow:
    """Placement of an image in native (ping, sample, channel) space.

    Pixel coordinates are continuous, with (0, 0) at the image's top-left
    corner; native cell ``(ping, sample)`` covers ``[ping, ping + 1) x
    [sample, sample + 1)``. ``layout`` is ``"chip"`` (one ``channel``, samples
    increasing to the right) or ``"dual"`` (port mirrored left of
    ``side_width``, starboard right of it, samples increasing outward).

    Attributes
    ----------
    layout : {"chip", "dual"}
    channel : int or None
        The chip's side (0 port, 1 starboard); ``None`` for dual.
    ping_start, ping_stop, sample_start, sample_stop : int
        Native extent (half-open). A chip's extent can reach outside the
        dataset (padded with NaN) so the target stays centred.
    px_per_ping, px_per_sample : float
        Image pixels per native ping / sample.
    width, height, side_width : int
        Image size; ``side_width`` is the width of one side in ``"dual"``.
    m_per_ping, m_per_sample : float
        Native spacing used.
    """

    layout: str
    channel: int | None
    ping_start: int
    ping_stop: int
    sample_start: int
    sample_stop: int
    px_per_ping: float
    px_per_sample: float
    width: int
    height: int
    side_width: int
    m_per_ping: float
    m_per_sample: float

    def to_pixel(self, ping: float, sample: float, channel: int) -> tuple[float, float]:
        """Image ``(x, y)`` of native position ``(ping, sample)`` on ``channel``."""
        y = (ping - self.ping_start) * self.px_per_ping
        ds = (sample - self.sample_start) * self.px_per_sample
        if self.layout == "chip":
            return ds, y
        return (self.side_width + ds if channel == 1 else self.side_width - ds), y

    def to_native(self, x: float, y: float) -> tuple[float, float, int]:
        """Native ``(ping, sample, channel)`` of image position ``(x, y)``."""
        ping = self.ping_start + y / self.px_per_ping
        if self.layout == "chip":
            return ping, self.sample_start + x / self.px_per_sample, int(self.channel)
        channel = 1 if x >= self.side_width else 0
        return ping, self.sample_start + abs(x - self.side_width) / self.px_per_sample, channel

    def box_to_native(self, x: float, y: float, w: float, h: float) -> dict:
        """Native extent of an image box ``(x, y, w, h)`` (top-left corner, size).

        The side is the one under the box centre; a dual-swath box crossing
        the centre line is clipped to that side.
        """
        cx = x + w / 2
        if self.layout == "dual":
            channel = 1 if cx >= self.side_width else 0
            x0, x1 = (max(x, self.side_width), x + w) if channel == 1 else (x, min(x + w, self.side_width))
        else:
            channel, x0, x1 = int(self.channel), x, x + w
        p0 = self.to_native(x0, y)[0]
        p1 = self.to_native(x0, y + h)[0]
        s_a = self.to_native(x0, y)[1]
        s_b = self.to_native(x1, y)[1]
        return {"channel": channel, "ping_start": min(p0, p1), "ping_stop": max(p0, p1),
                "sample_start": min(s_a, s_b), "sample_stop": max(s_a, s_b)}

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NativeWindow:
        return cls(**data)


def _resize(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Linear resize that keeps NaN cells NaN (and does not bleed them)."""
    finite = np.isfinite(grid)
    if not finite.any():
        return np.full(shape, np.nan, dtype=np.float32)
    factors = (shape[0] / grid.shape[0], shape[1] / grid.shape[1])
    filled = np.where(finite, grid, np.nanmean(grid))
    out = zoom(filled, factors, order=1, mode="nearest", grid_mode=True)
    mask = zoom(finite.astype(np.float32), factors, order=0, mode="nearest", grid_mode=True) < 0.5
    out = out[: shape[0], : shape[1]].astype(np.float32)
    out[mask[: shape[0], : shape[1]]] = np.nan
    return out


def native_chip(
    ds: xr.Dataset,
    *,
    ping: int,
    channel: int,
    sample: int,
    size_m: float,
    size_px: int,
    value_var: str = "backscatter_intensity",
    m_per_ping: float | None = None,
) -> tuple[np.ndarray, NativeWindow]:
    """A ``size_m`` x ``size_m`` window around a target, as a ``size_px`` square image.

    Parameters
    ----------
    ds : xr.Dataset
        One frequency channel (loaded or lazy).
    ping, channel, sample : int
        Target position (positions along ``ping``/``sample``; channel 0 port, 1 starboard).
    size_m : float
        Window side in metres, both along and across track (slant range).
    size_px : int
        Output side in pixels.
    value_var : str, default "backscatter_intensity"
    m_per_ping : float, optional
        Along-track spacing; default :func:`along_track_spacing_m` over the
        pings around the target.

    Returns
    -------
    image : np.ndarray
        float32 ``(size_px, size_px)``, NaN where there is no data (outside
        the dataset or already NaN); rows are pings, columns samples.
    window : NativeWindow
    """
    m_s = nominal_sample_interval_m(ds)
    if m_per_ping is None:
        lo, hi = max(0, ping - 500), min(ds.sizes["ping"], ping + 500)
        m_per_ping = along_track_spacing_m(ds.isel(ping=slice(lo, hi)))
    n_s = max(1, int(round(size_m / m_s)))
    n_p = max(1, int(round(size_m / m_per_ping)))
    s0, p0 = sample - n_s // 2, ping - n_p // 2
    a0, a1 = max(p0, 0), min(p0 + n_p, ds.sizes["ping"])
    b0, b1 = max(s0, 0), min(s0 + n_s, ds.sizes["sample"])
    grid = np.full((n_p, n_s), np.nan, dtype=np.float32)
    if a1 > a0 and b1 > b0:
        sub = ds[value_var].isel(channel=channel, ping=slice(a0, a1), sample=slice(b0, b1)).values
        grid[a0 - p0:a1 - p0, b0 - s0:b1 - s0] = sub
    img = _resize(grid, (size_px, size_px))
    win = NativeWindow("chip", int(channel), p0, p0 + n_p, s0, s0 + n_s, size_px / n_p, size_px / n_s,
                       size_px, size_px, 0, float(m_per_ping), float(m_s))
    return img, win


def swath_tiles(
    ds: xr.Dataset,
    *,
    tile_m: float,
    px_m: float,
    overlap_m: float = 0.0,
    value_var: str = "backscatter_intensity",
    m_per_ping: float | None = None,
) -> list[tuple[np.ndarray, NativeWindow]]:
    """Cut the whole channel into consecutive dual-swath tiles.

    Each tile spans ``tile_m`` along track (the last one may be shorter) and
    the full recorded range on both sides, resized so one pixel is
    ``px_m`` metres both ways. Port is mirrored on the left, starboard on
    the right, nadir in the middle.

    Parameters
    ----------
    ds : xr.Dataset
        One frequency channel.
    tile_m : float
        Tile length along track (m).
    px_m : float
        Pixel size (m) in both directions.
    overlap_m : float, default 0
        Along-track overlap between consecutive tiles, so an object on a
        tile edge appears whole in one of them.
    value_var : str, default "backscatter_intensity"
    m_per_ping : float, optional
        Default: :func:`along_track_spacing_m` over the whole channel.

    Returns
    -------
    list of (image, NativeWindow)
        float32 images (NaN where there is no data).
    """
    m_s = nominal_sample_interval_m(ds)
    if m_per_ping is None:
        m_per_ping = along_track_spacing_m(ds)
    n_pings, n_s = ds.sizes["ping"], ds.sizes["sample"]
    tile_pings = max(1, int(round(tile_m / m_per_ping)))
    step = max(1, tile_pings - int(round(overlap_m / m_per_ping)))
    side_w = max(1, int(round(n_s * m_s / px_m)))
    values = ds[value_var]
    tiles = []
    start = 0
    while True:
        stop = min(start + tile_pings, n_pings)
        h = max(1, int(round((stop - start) * m_per_ping / px_m)))
        block = values.isel(ping=slice(start, stop)).values
        port = _resize(block[:, 0, :].astype(np.float32), (h, side_w))
        stbd = _resize(block[:, 1, :].astype(np.float32), (h, side_w))
        img = np.hstack([port[:, ::-1], stbd])
        win = NativeWindow("dual", None, start, stop, 0, n_s, h / (stop - start), side_w / n_s,
                           2 * side_w, h, side_w, float(m_per_ping), float(m_s))
        tiles.append((img, win))
        if stop >= n_pings:
            break
        start += step
    logger.debug(f"swath_tiles: {len(tiles)} tile(s) of {tile_pings} pings, {2 * side_w} px wide")
    return tiles
