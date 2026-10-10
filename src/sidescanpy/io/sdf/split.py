"""
Split one Klein SDF file's pages into runs of constant sample interval.

The range setting can change mid-file. Klein keeps the metres per sample
of a frequency only when the sample count scales with the range; on some
recorders it does not (for example 2500 samples at 60 m, then 3334 at 40
m), so the sample interval changes. The survey model has one
``sample_rate_hz`` per frequency (computed by the converter from the
first page), so such a file cannot become one survey without mapping
some pings to the wrong range. :func:`split_by_sample_rate` cuts the
pages into consecutive runs (in file order) within which every
frequency keeps its sample rate, each run converted on its own by
:func:`sidescanpy.io.sdf.convert`.

Cut rule (same as the XTF split): a frequency's rate is
``num_samples / range_m`` (proportional to the sample rate at constant
sound speed). A run's reference rate for a frequency is the first one it
records; a page whose rate differs from that reference by more than
``rtol`` (relative) starts a new run, for every frequency. Pages without
a positive range do not take part. Runs with fewer than ``min_pings``
pages of their most-recorded frequency are dropped (a range change often
leaves a few pings at an intermediate setting); a file without a cut is
kept whole.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

from sidescanpy.io.sdf.pages import SDFSonarPage

logger = logging.getLogger(__name__)

#: Default relative change in sample rate that starts a new run.
SPLIT_RTOL = 0.01
#: Default minimum run length (pings of the most-recorded frequency).
SPLIT_MIN_PINGS = 50


def sample_rate_runs(pages: Sequence, *, rtol: float = SPLIT_RTOL) -> list[tuple[int, int]]:
    """Consecutive runs of constant sample rate, as ``(start, stop)`` page ranges.

    Parameters
    ----------
    pages : sequence
        Pages of one file in file order (non-sonar pages are allowed and
        never cause a cut).
    rtol : float, default 0.01
        Relative change in a frequency's rate that starts a new run.

    Returns
    -------
    list[tuple[int, int]]
        Half-open ranges over ``pages`` covering all of it; empty when
        ``pages`` is empty.
    """
    n = len(pages)
    if n == 0:
        return []
    bounds = [0]
    reference: dict[int, float] = {}
    for i, page in enumerate(pages):
        if not isinstance(page, SDFSonarPage) or page.range_m <= 0:
            continue
        rate = page.num_samples / page.range_m
        ref = reference.get(page.center_freq_hz)
        if ref is not None and abs(rate - ref) > rtol * ref:
            bounds.append(i)
            reference = {}
        reference.setdefault(page.center_freq_hz, rate)
    bounds.append(n)
    return list(zip(bounds[:-1], bounds[1:]))


def _n_pings(pages: Sequence) -> int:
    counts = Counter(p.center_freq_hz for p in pages if isinstance(p, SDFSonarPage))
    return max(counts.values(), default=0)


def split_by_sample_rate(
    pages: Sequence,
    *,
    min_pings: int = SPLIT_MIN_PINGS,
    rtol: float = SPLIT_RTOL,
    source: str = "",
) -> list[list]:
    """Split one file's pages at sample-rate (range-setting) changes.

    Parameters
    ----------
    pages : sequence
        Pages of one file in file order (:func:`~sidescanpy.io.sdf.read_sdf`).
    min_pings : int, default 50
        Runs with fewer pings (of their most-recorded frequency) are
        dropped. Ignored when the file has a single run.
    rtol : float, default 0.01
        Relative change in a frequency's rate that starts a new run.
    source : str, optional
        File name for log messages.

    Returns
    -------
    list[list]
        The kept runs in file order. One run (the whole file) when the
        rate never changes; empty when every run is shorter than
        ``min_pings`` (logged at WARNING).
    """
    runs = sample_rate_runs(pages, rtol=rtol)
    if len(runs) <= 1:
        return [list(pages)] if runs else []
    groups = [list(pages[a:b]) for a, b in runs]
    kept = [g for g in groups if _n_pings(g) >= min_pings]
    n_dropped = sum(_n_pings(g) for g in groups if _n_pings(g) < min_pings)
    logger.debug(
        f"{source}: sample rate changes {len(runs) - 1} time(s); kept runs of "
        f"{[_n_pings(g) for g in kept]} pings, dropped {n_dropped} ping(s) in runs shorter than {min_pings}"
    )
    if not kept:
        logger.warning(f"{source}: every one of {len(runs)} constant-sample-rate runs is shorter than "
                       f"{min_pings} pings; nothing kept")
    return kept
