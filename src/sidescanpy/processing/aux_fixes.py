"""
Reduce sample-and-hold auxiliary records to the sensor's real fixes.

Many recorders log navigation and attitude as sample-and-hold: every ping
or packet carries the latest value the sensor delivered, so the same value
repeats until the sensor updates (a ~1 Hz GPS under ~10 Hz pings repeats
each position about ten times). Interpolating linearly between those held
entries keeps the stair-steps: consecutive pings stack on one position and
then jump. :func:`held_value_fixes` keeps only the entries where a value
actually changed, timed where the change most likely happened, so that
interpolating the result recovers a smooth track. Some recorders also
spread one sensor update over consecutive entries (the bulk change, then a
small follow-up one entry later); ``merge_window_s`` collapses such
changes into one update event, and :func:`aux_update_events` reports the
change and event statistics needed to choose it.

2026 rshomberg@uri.edu
"""

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: Latitude and longitude as one group, for ``groups=[POSITION_GROUP]``.
#: Not the default: some recorders log the two coordinates of one GPS fix in
#: separate entries, and joining them turns each half-update into a fix
#: with a stale other coordinate.
POSITION_GROUP: tuple[str, str] = ("latitude", "longitude")

_OBS = "obs"

#: Percentiles reported by :func:`aux_update_events`.
_PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)

#: An update event combining more changes than this suggests the stream
#: updates faster than ``merge_window_s`` (a WARNING is logged).
_MANY_CHANGES_PER_EVENT = 5

#: A merge window: one for every group, or per variable name.
MergeWindow = float | Mapping[str, float] | None


def held_value_fixes(
    record: xr.Dataset,
    *,
    groups: Sequence[Sequence[str]] | None = None,
    max_hold_s: float = 2.0,
    merge_window_s: MergeWindow = None,
) -> xr.Dataset:
    """Keep only the real fixes of a sample-and-hold auxiliary record.

    Parameters
    ----------
    record : xr.Dataset
        Auxiliary record with dimension ``obs`` and a ``time`` coordinate
        along it (datetime64). Entries need not be sorted.
    groups : sequence of sequence of str, optional
        Variables that update together. A group's entry is a fix when any of
        its variables changed. By default every numeric ``obs`` variable is
        its own group, latitude and longitude included: some recorders log
        the two coordinates of one fix in separate entries a fraction of a
        second apart, and a joint group would turn each half-update into a
        fix carrying a stale other coordinate. Pass
        ``groups=[POSITION_GROUP]`` to join them. Variables not named in
        ``groups`` are single-variable groups.
    max_hold_s : float, default 2.0
        A value unchanged for longer than this (from the first to the last
        entry that carries it) is treated as a real hold, such as a stop or
        a sensor dropout: an extra anchor entry keeps the value at the time
        of the last entry before the change, so interpolation stays flat
        across the hold instead of ramping through it. ``np.inf`` disables
        anchors. With ``merge_window_s``, the hold is measured from the
        last change of the update event.
    merge_window_s : float or mapping of str to float, optional
        Within each group, consecutive changes at most this many seconds
        apart (chained: each measured from the previous change) collapse
        into one update event, for recorders that log one sensor update
        over consecutive entries (the bulk change, then a small follow-up
        one entry later). The event carries the group's value after its
        last change and is timed by the fix-time rule applied to its
        *first* change (see Notes). Choose a window longer than the
        follow-up delay and shorter than the real update interval;
        :func:`aux_update_events` shows both. A float applies to every
        group; a mapping ``{variable: seconds}`` applies only to the groups
        holding those variables (a group takes the smallest window of its
        variables) and leaves the others unmerged, so a window sized for a
        ~3 Hz GPS does not chain the changes of a ~10 Hz roll sensor into
        one event. ``None`` (default) keeps every change as its own fix.
        Groups are merged independently.

    Returns
    -------
    xr.Dataset
        A record on a new ``obs`` axis: the sorted union of every group's
        fix and anchor times. A variable is NaN at times that are not one
        of its own fixes (drop NaN rows per variable before interpolating;
        the ``interpolate_aux`` step does). Variables are float64. The
        input attrs are kept, plus ``held_value_fixes_entries_in``,
        ``held_value_fixes_entries_out``, ``held_value_fixes_dropped``
        (entries in minus out) and ``held_value_fixes_max_hold_s``, plus
        ``held_value_fixes_merge_window_s`` when ``merge_window_s`` is set
        (a float, or a JSON string of the mapping).
        Non-numeric variables and variables without the ``obs`` dimension
        are left out.

    Raises
    ------
    KeyError
        If ``record`` has no ``time`` coordinate, or ``groups`` names a
        variable the record does not have.
    ValueError
        If ``max_hold_s`` is negative or NaN, or a ``merge_window_s`` value
        is negative or not finite.

    Notes
    -----
    Per group, after sorting by time (stable, so logging order is kept
    within equal times) and dropping entries where any of the group's
    variables is NaN or the time is NaT:

    - **Duplicate timestamps** collapse to the *last* entry logged at that
      time, the most recent value the logger had. Earlier entries at the
      same time are superseded, not averaged (averaging a stale and a new
      value would invent a position, and does not apply to compass angles).
      When the entries at one time disagree, the update was seen at that
      time, so a fix there keeps that time.
    - **A fix** is an entry whose value differs from the previous entry
      (exact inequality; this is right for compass angles too, which the
      interpolator unwraps). The first entry is always a fix. NaN entries
      are never fixes.
    - **Fix time.** When the previous entry repeated the value before it
      (the value was being held), the new value arrived somewhere between
      the two entries, so the fix is placed at their midpoint. When the
      previous entry was itself a fix, the stream is updating at the
      logging rate and the entry's own time is kept. A record in which
      every entry differs from the previous one is therefore returned with
      its values and times unchanged. The first fix keeps its own time, as
      does a fix at a duplicated timestamp whose entries disagree.
    - **Holds** longer than ``max_hold_s`` add an anchor (see above),
      including the hold that ends the record.
    - **Merged updates** (``merge_window_s``). The first entry of the
      record is never merged with what follows: it is the start of the
      record, not an observed change. An event takes the time its first
      change would have had as a fix (the midpoint after a hold) and the
      value after its last change. The first changed entry is when the
      update started reaching the log; the follow-up entries complete the
      same update rather than report new motion, so the sensor's fix
      epoch is best estimated from the first. Timing at the last change
      instead would add a delay that varies with how many entries an
      update happened to spread over, which shows up as speed jitter;
      timing at the first gives a near-constant latency, which leaves
      speed unbiased. A stream that changes at every entry with entries
      closer than ``merge_window_s`` chains into long events; a WARNING
      is logged when an event combines more than five changes.
    """
    if "time" not in record.coords and "time" not in record.data_vars:
        raise KeyError("held_value_fixes needs a 'time' coordinate on the record")
    _check_hold_and_merge(max_hold_s, merge_window_s)

    numeric = [
        name for name, var in record.data_vars.items()
        if var.dims == (_OBS,) and (np.issubdtype(var.dtype, np.number) or var.dtype == bool)
    ]
    skipped = [name for name in record.data_vars if name not in numeric]
    if skipped:
        logger.debug("held_value_fixes: leaving out non-numeric or non-obs variables %s", skipped)
    group_list = _resolve_groups(numeric, groups, record)

    n_in = int(record.sizes.get(_OBS, 0))
    t_sorted, order = _sorted_times(record)
    max_hold_ns = np.inf if np.isposinf(max_hold_s) else max_hold_s * 1e9

    per_group: list[tuple[tuple[str, ...], np.ndarray, np.ndarray]] = []
    for group in group_list:
        values = _group_values(record, group, order)
        merge_ns = _group_merge_ns(group, merge_window_s)
        g_times, g_values, n_fix, n_anchor, n_changes, n_many = _group_fixes(
            t_sorted, values, max_hold_ns, merge_ns,
        )
        if n_many:
            logger.warning(
                "held_value_fixes %s: %d update events combine more than %d changes; the stream may "
                "update faster than merge_window_s=%.3g s",
                list(group), n_many, _MANY_CHANGES_PER_EVENT, merge_ns / 1e9,
            )
        logger.debug(
            "held_value_fixes %s: %d finite entries -> %d fixes (%d changes) + %d hold anchors",
            list(group), int(np.isfinite(values).all(axis=1).sum()), n_fix, n_changes, n_anchor,
        )
        per_group.append((group, g_times, g_values))

    all_times = (
        np.unique(np.concatenate([g[1] for g in per_group]))
        if per_group else np.empty(0, dtype=np.int64)
    )
    data_vars = {}
    for group, g_times, g_values in per_group:
        pos = np.searchsorted(all_times, g_times)
        for j, name in enumerate(group):
            col = np.full(len(all_times), np.nan)
            col[pos] = g_values[:, j]
            data_vars[name] = ((_OBS,), col)
    # Keep the input's variable order.
    data_vars = {name: data_vars[name] for name in numeric if name in data_vars}

    n_out = len(all_times)
    attrs = dict(record.attrs)
    attrs.update({
        "held_value_fixes_entries_in": n_in,
        "held_value_fixes_entries_out": n_out,
        "held_value_fixes_dropped": n_in - n_out,
        "held_value_fixes_max_hold_s": float(max_hold_s),
    })
    if merge_window_s is not None:
        attrs["held_value_fixes_merge_window_s"] = (
            json.dumps({k: float(v) for k, v in merge_window_s.items()})
            if isinstance(merge_window_s, Mapping) else float(merge_window_s)
        )
    return xr.Dataset(
        data_vars,
        coords={"time": ((_OBS,), all_times.astype("datetime64[ns]"))},
        attrs=attrs,
    )


@dataclass(frozen=True)
class AuxUpdateStats:
    """How one variable group of a sample-and-hold record updates.

    Attributes
    ----------
    group : tuple of str
        The group's variables.
    entries : int
        Entries with a valid time and every group variable finite.
    changes : int
        Entries whose value differs from the previous one (after duplicate
        timestamps collapse to their last entry). The record's first entry
        is not counted.
    events : int
        Update events after merging changes within ``merge_window_s``
        (equal to ``changes`` without merging). The first entry is not
        counted.
    merge_window_s : float or None
        The merge window used for this group.
    change_gap_entries, change_gap_s : dict of str to float
        Gap between consecutive changes, in entries (of the de-duplicated
        finite sequence) and seconds, at percentiles ``p5``, ``p25``,
        ``p50``, ``p75``, ``p95``. NaN with fewer than two changes.
    event_gap_entries, event_gap_s : dict of str to float
        Same, between the first changes of consecutive events.
    merged_fraction : float
        Fraction of changes that come at most ``merge_window_s`` after the
        previous change (the first change has no previous one and counts as
        not merged). NaN without a merge window or without changes.
    """

    group: tuple[str, ...]
    entries: int
    changes: int
    events: int
    merge_window_s: float | None
    change_gap_entries: dict[str, float]
    change_gap_s: dict[str, float]
    event_gap_entries: dict[str, float]
    event_gap_s: dict[str, float]
    merged_fraction: float

    def as_dict(self) -> dict:
        """JSON-friendly copy (``group`` as a list)."""
        out = asdict(self)
        out["group"] = list(self.group)
        return out


def aux_update_events(
    record: xr.Dataset,
    *,
    groups: Sequence[Sequence[str]] | None = None,
    merge_window_s: MergeWindow = None,
) -> dict[str, AuxUpdateStats]:
    """Change and update-event statistics of a sample-and-hold record.

    A diagnostic for choosing ``groups`` and ``merge_window_s`` of
    :func:`held_value_fixes`, which uses the same change detection and
    merging. A sensor that delivers each update over two consecutive log
    entries shows bimodal change gaps (one peak at 1 entry, one at the
    real update interval) that become unimodal events once merged.

    Parameters
    ----------
    record : xr.Dataset
        Auxiliary record with dimension ``obs`` and a ``time`` coordinate.
    groups : sequence of sequence of str, optional
        As in :func:`held_value_fixes`; every other numeric ``obs``
        variable is its own group.
    merge_window_s : float or mapping of str to float, optional
        As in :func:`held_value_fixes`. ``None`` reports changes as events.

    Returns
    -------
    dict of str to AuxUpdateStats
        One entry per group, keyed by its variable names joined with
        ``"+"`` (``"latitude+longitude"`` for ``POSITION_GROUP``).

    Raises
    ------
    KeyError
        If ``record`` has no ``time`` coordinate, or ``groups`` names a
        variable the record does not have.
    ValueError
        If a ``merge_window_s`` value is negative or not finite.
    """
    if "time" not in record.coords and "time" not in record.data_vars:
        raise KeyError("aux_update_events needs a 'time' coordinate on the record")
    _check_hold_and_merge(np.inf, merge_window_s)
    numeric = [
        name for name, var in record.data_vars.items()
        if var.dims == (_OBS,) and (np.issubdtype(var.dtype, np.number) or var.dtype == bool)
    ]
    t_sorted, order = _sorted_times(record)

    out: dict[str, AuxUpdateStats] = {}
    for group in _resolve_groups(numeric, groups, record):
        values = _group_values(record, group, order)
        merge_ns = _group_merge_ns(group, merge_window_s)
        n_entries = int(np.isfinite(values).all(axis=1).sum()) if values.size else 0
        t, _v, change, _changed_at_time = _group_changes(t_sorted, values)
        idx = np.flatnonzero(change)[1:] if len(t) else np.empty(0, np.int64)
        first, _last = _merge_clusters(t, idx, merge_ns)
        if merge_ns is None or len(idx) == 0:
            merged = np.nan
        else:
            merged = float(np.sum(np.diff(t[idx]) <= merge_ns) / len(idx))
        out["+".join(group)] = AuxUpdateStats(
            group=tuple(group),
            entries=n_entries,
            changes=len(idx),
            events=len(first),
            merge_window_s=None if merge_ns is None else merge_ns / 1e9,
            change_gap_entries=_percentiles(np.diff(idx)),
            change_gap_s=_percentiles(np.diff(t[idx]) / 1e9),
            event_gap_entries=_percentiles(np.diff(first)),
            event_gap_s=_percentiles(np.diff(t[first]) / 1e9),
            merged_fraction=merged,
        )
        logger.debug(
            "aux_update_events %s: %d entries, %d changes, %d events",
            list(group), n_entries, len(idx), len(first),
        )
    return out


# ============================================================
# INTERNALS
# ============================================================

def _check_hold_and_merge(max_hold_s: float, merge_window_s: MergeWindow) -> None:
    if np.isnan(max_hold_s) or max_hold_s < 0:
        raise ValueError(f"max_hold_s must be >= 0 (or inf), got {max_hold_s!r}")
    if merge_window_s is None:
        return
    values = merge_window_s.values() if isinstance(merge_window_s, Mapping) else [merge_window_s]
    for value in values:
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(f"merge_window_s values must be finite and >= 0, got {merge_window_s!r}")


def _group_merge_ns(group: tuple[str, ...], merge_window_s: MergeWindow) -> float | None:
    """A group's merge window in ns: the float, or its variables' smallest mapped one."""
    if merge_window_s is None:
        return None
    if isinstance(merge_window_s, Mapping):
        mine = [merge_window_s[name] for name in group if name in merge_window_s]
        return min(mine) * 1e9 if mine else None
    return merge_window_s * 1e9


def _sorted_times(record: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Valid times (int64 ns) in stable sorted order, and that order."""
    times = np.asarray(record["time"].values).astype("datetime64[ns]")
    t_ns = times.astype(np.int64)
    order = np.argsort(t_ns, kind="stable")
    order = order[~np.isnat(times)[order]]
    return t_ns[order], order


def _group_values(record: xr.Dataset, group: tuple[str, ...], order: np.ndarray) -> np.ndarray:
    """``(n, k)`` float64 values of a group's variables in ``order``."""
    if not len(order):
        return np.empty((0, len(group)))
    return np.column_stack([np.asarray(record[name].values, dtype=np.float64)[order] for name in group])


def _percentiles(x: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        return {f"p{p}": float("nan") for p in _PERCENTILES}
    vals = np.percentile(np.asarray(x, dtype=np.float64), _PERCENTILES)
    return {f"p{p}": float(v) for p, v in zip(_PERCENTILES, vals, strict=True)}

def _resolve_groups(
    numeric: list[str], groups: Iterable[Iterable[str]] | None, record: xr.Dataset,
) -> list[tuple[str, ...]]:
    """Variable groups: the caller's, then every other variable on its own."""
    if groups is None:
        groups = []
    resolved: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for group in groups:
        group = tuple(group)
        for name in group:
            if name not in record.data_vars:
                raise KeyError(f"held_value_fixes: group variable {name!r} is not in the record")
            if name not in numeric:
                raise KeyError(f"held_value_fixes: group variable {name!r} is not a numeric 'obs' variable")
            if name in seen:
                raise ValueError(f"held_value_fixes: variable {name!r} is in more than one group")
            seen.add(name)
        if group:
            resolved.append(group)
    resolved.extend((name,) for name in numeric if name not in seen)
    return resolved


def _group_changes(
    t_ns: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finite, de-duplicated times and values of one group, and its changes.

    ``t_ns`` is sorted; ``values`` is ``(n, k)`` aligned with it. Returns
    ``(t, v, change, changed_at_time)`` on the de-duplicated sequence:
    ``change[i]`` is True when entry ``i`` differs from entry ``i - 1``
    (always True for the first entry), ``changed_at_time[i]`` when the
    entries collapsed into ``i`` disagreed.
    """
    finite = np.isfinite(values).all(axis=1) if values.size else np.zeros(len(t_ns), bool)
    t = t_ns[finite]
    v = values[finite]
    if len(t) == 0:
        empty = np.zeros(0, bool)
        return np.empty(0, np.int64), np.empty((0, values.shape[1])), empty, empty

    # Duplicate timestamps: the last entry logged at a time wins. If the
    # first entry at that time differs from the last, the change was seen
    # at that very time.
    last = np.ones(len(t), dtype=bool)
    last[:-1] = t[1:] != t[:-1]
    first = np.ones(len(t), dtype=bool)
    first[1:] = t[1:] != t[:-1]
    changed_at_time = np.any(v[first] != v[last], axis=1)
    t, v = t[last], v[last]

    change = np.ones(len(t), dtype=bool)
    change[1:] = np.any(v[1:] != v[:-1], axis=1)
    return t, v, change, changed_at_time


def _merge_clusters(
    t: np.ndarray, idx: np.ndarray, merge_ns: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """First and last change index of each cluster of changes ``idx``.

    Consecutive changes at most ``merge_ns`` apart chain into one cluster.
    Without a window every change is its own cluster.
    """
    if merge_ns is None or len(idx) == 0:
        return idx, idx
    new = np.ones(len(idx), dtype=bool)
    new[1:] = np.diff(t[idx]) > merge_ns
    heads = np.flatnonzero(new)
    tails = np.append(heads[1:] - 1, len(idx) - 1)
    return idx[heads], idx[tails]


def _group_fixes(
    t_ns: np.ndarray, values: np.ndarray, max_hold_ns: float, merge_ns: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Fix and anchor times (int64 ns, sorted, unique) and values for one group.

    ``t_ns`` is sorted; ``values`` is ``(n, k)`` aligned with it. Also
    returns the number of fixes, anchors, raw changes and events of more
    than ``_MANY_CHANGES_PER_EVENT`` changes.
    """
    t, v, change, changed_at_time = _group_changes(t_ns, values)
    if len(t) == 0:
        return np.empty(0, np.int64), np.empty((0, values.shape[1])), 0, 0, 0, 0

    starts = np.flatnonzero(change)
    # The first entry is the record's start, never merged with what follows.
    first, last = _merge_clusters(t, starts[1:], merge_ns)
    first = np.concatenate([[0], first]).astype(np.int64)
    last = np.concatenate([[0], last]).astype(np.int64)
    n_merged = np.searchsorted(starts, last) - np.searchsorted(starts, first) + 1
    many = int(np.sum(n_merged > _MANY_CHANGES_PER_EVENT))
    # The value after the event's last change holds until the next event.
    ends = np.append(first[1:] - 1, len(t) - 1)

    fix_t = t[first].copy()
    # A fix that ends a held run (previous entry was a repeat) arrived
    # between the previous entry and this one: use the midpoint. For a
    # merged event this applies to its first change.
    held_before = np.zeros(len(first), dtype=bool)
    held_before[1:] = ~change[first[1:] - 1]
    held_before &= ~changed_at_time[first]
    prev = t[first[held_before] - 1]
    fix_t[held_before] = prev + (t[first[held_before]] - prev) // 2
    fix_v = v[last]

    long_hold = (t[ends] - t[last]) > max_hold_ns
    anchor_t = t[ends[long_hold]]
    anchor_v = v[ends[long_hold]]

    out_t = np.concatenate([fix_t, anchor_t])
    out_v = np.concatenate([fix_v, anchor_v])
    # Sorted; on an exact time collision keep the later-ordered (a fix over an anchor).
    order = np.argsort(out_t, kind="stable")
    out_t, out_v = out_t[order], out_v[order]
    keep = np.ones(len(out_t), dtype=bool)
    keep[:-1] = out_t[1:] != out_t[:-1]
    return out_t[keep], out_v[keep], len(first), int(long_hold.sum()), len(starts) - 1, many
