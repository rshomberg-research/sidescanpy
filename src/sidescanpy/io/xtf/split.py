"""
Split XTF pings into runs of constant sample interval.

Some recorders keep a fixed number of samples per ping and change the
range setting mid-file, so the sample interval (``TimeDuration /
NumSamples``) changes from one ping to the next. The survey model has
one ``sample_rate_hz`` per frequency, so a file like that cannot become
one survey without giving some pings the wrong range.
:func:`split_by_sample_rate` cuts the pings of one file into consecutive
runs (in file order) within which every channel keeps its sample rate,
each run an :class:`~sidescanpy.io.xtf.reader.XtfPingData` that
:func:`sidescanpy.io.xtf.convert` turns into its own survey.

Cut rule
--------
For each port/starboard channel the per-ping sample rate is
``NumSamples / TimeDuration`` (``NumSamples / SlantRange``, proportional
at constant sound speed, when a channel never records ``TimeDuration``).
A run's reference rate for a channel is the first rate it records in the
run; a ping whose rate differs from that reference by more than ``rtol``
(relative) starts a new run. Comparing with the run's reference rather
than the previous ping stops a slow drift from accumulating inside a
run. Pings where a channel is absent (``present`` False) or carries no
usable rate do not take part for that channel, so they never cause a
cut. A cut in any channel cuts every channel.

Runs shorter than ``min_pings`` are dropped (a range change usually
leaves a few pings at an intermediate setting). A file without a cut is
returned whole whatever its length.

Each run is a view of the file's arrays (no copy, no re-read). A channel
absent from every ping of a run is left out of that run, so a run may
hold fewer frequencies than the file. ``stats`` and ``block_stats``
are the whole file's. ``segment`` (:class:`XtfSegmentInfo`) records
where the run sits in the file; the converter turns it into ``xtf_``
provenance attributes.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from sidescanpy.io.xtf.reader import (
    XtfChannel,
    XtfFrequencyGroup,
    XtfPingData,
    XtfSegmentInfo,
)

logger = logging.getLogger(__name__)

#: Default relative change in sample rate that starts a new run.
SPLIT_RTOL = 0.01
#: Default minimum run length (pings) kept by :func:`split_by_sample_rate`.
SPLIT_MIN_PINGS = 50


def sample_rate_runs(data: XtfPingData, *, rtol: float = SPLIT_RTOL) -> list[tuple[int, int]]:
    """Consecutive runs of constant sample rate, as ``(start, stop)`` ping ranges.

    The runs cover every ping of ``data`` in order. See the module
    docstring for the cut rule.

    Parameters
    ----------
    data : XtfPingData
        Pings of one file (:func:`~sidescanpy.io.xtf.read_xtf`).
    rtol : float, default 0.01
        Relative change in a channel's sample rate that starts a new run.

    Returns
    -------
    list[tuple[int, int]]
        Half-open ping ranges; empty when ``data`` has no pings.
    """
    n = data.n_pings
    if n == 0:
        return []
    cuts: set[int] = set()
    for ch in _sonar_channels(data):
        cuts.update(_channel_cuts(ch, rtol))
    bounds = [0, *sorted(c for c in cuts if 0 < c < n), n]
    return list(zip(bounds[:-1], bounds[1:]))


def split_by_sample_rate(
    data: XtfPingData,
    *,
    min_pings: int = SPLIT_MIN_PINGS,
    rtol: float = SPLIT_RTOL,
) -> list[XtfPingData]:
    """Split one file's pings at sample-rate (range-setting) changes.

    Parameters
    ----------
    data : XtfPingData
        Pings of one file (:func:`~sidescanpy.io.xtf.read_xtf`).
    min_pings : int, default 50
        Runs with fewer pings are dropped. Ignored when the file has a
        single run.
    rtol : float, default 0.01
        Relative change in a channel's sample rate that starts a new run.

    Returns
    -------
    list[XtfPingData]
        The kept runs in file order, each a view of ``data`` with
        ``segment`` set. One element (covering the whole file) when the
        sample rate never changes; empty when every run is shorter than
        ``min_pings`` (logged at WARNING).
    """
    runs = sample_rate_runs(data, rtol=rtol)
    if not runs:
        return []
    if len(runs) == 1:
        kept, dropped = runs, []
    else:
        kept = [r for r in runs if r[1] - r[0] >= min_pings]
        dropped = [r for r in runs if r[1] - r[0] < min_pings]
    n_dropped = sum(b - a for a, b in dropped)
    if len(runs) > 1:
        logger.debug(
            "%s: sample rate changes %d time(s); %d run(s) kept %s, %d run(s) shorter "
            "than %d pings dropped (%d pings)",
            data.path, len(runs) - 1, len(kept), [b - a for a, b in kept],
            len(dropped), min_pings, n_dropped,
        )
    if not kept:
        logger.warning(
            "%s: every one of %d constant-sample-rate runs is shorter than %d pings; "
            "nothing kept",
            data.path, len(runs), min_pings,
        )
        return []
    return [
        _slice(
            data, a, b,
            XtfSegmentInfo(
                index=i, count=len(kept), ping_start=a, ping_stop=b,
                n_pings_file=data.n_pings, n_runs_dropped=len(dropped),
                n_pings_dropped=n_dropped, rtol=rtol,
            ),
        )
        for i, (a, b) in enumerate(kept, 1)
    ]


# ============================================================
# INTERNALS
# ============================================================

def _sonar_channels(data: XtfPingData) -> list[XtfChannel]:
    return [
        ch for g in data.frequency_groups for ch in (g.port, g.stbd) if ch is not None
    ]


def _channel_rates(ch: XtfChannel) -> np.ndarray:
    """Per-ping sample rate of one channel (NaN where absent or unknown)."""
    h = ch.headers
    n = ch.num_samples.astype(np.float64)
    ok = ch.present & (n > 0)
    duration = h["TimeDuration"].astype(np.float64)
    basis = duration if np.any(ok & (duration > 0)) else h["SlantRange"].astype(np.float64)
    ok &= basis > 0
    rate = np.full(len(n), np.nan)
    rate[ok] = n[ok] / basis[ok]
    return rate


def _channel_cuts(ch: XtfChannel, rtol: float) -> list[int]:
    rate = _channel_rates(ch)
    idx = np.flatnonzero(np.isfinite(rate))
    vals = rate[idx]
    cuts = []
    k = 0
    while k < len(vals):
        ref = vals[k]
        off = np.flatnonzero(np.abs(vals[k:] - ref) > rtol * ref)
        if off.size == 0:
            break
        k += int(off[0])
        cuts.append(int(idx[k]))
    return cuts


def _slice_channel(ch: XtfChannel, a: int, b: int) -> XtfChannel:
    num = ch.num_samples[a:b]
    width = int(num.max()) if num.size else 0
    return replace(
        ch,
        samples=ch.samples[a:b, :width],
        num_samples=num,
        present=ch.present[a:b],
        headers=ch.headers[a:b],
    )


def _slice(data: XtfPingData, a: int, b: int, info: XtfSegmentInfo) -> XtfPingData:
    channels = {
        num: _slice_channel(ch, a, b)
        for num, ch in data.channels.items()
        if ch.present[a:b].any()
    }
    groups = []
    for g in data.frequency_groups:
        port = channels.get(g.port.channel_number) if g.port is not None else None
        stbd = channels.get(g.stbd.channel_number) if g.stbd is not None else None
        if port is not None or stbd is not None:
            groups.append(XtfFrequencyGroup(frequency=g.frequency, port=port, stbd=stbd))
    return replace(
        data,
        ping_headers=data.ping_headers[a:b],
        packet_offsets=data.packet_offsets[a:b],
        channels=channels,
        frequency_groups=groups,
        segment=info,
    )
