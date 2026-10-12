"""
Reduce sample-and-hold auxiliary records to the sensor's real fixes.

Many recorders log navigation and attitude as sample-and-hold: every ping
or packet carries the latest value the sensor delivered, so the same value
repeats until the sensor updates (a ~1 Hz GPS under ~10 Hz pings repeats
each position about ten times). Interpolating linearly between those held
entries keeps the stair-steps: consecutive pings stack on one position and
then jump. :func:`held_value_fixes` keeps only the entries where a value
actually changed, timed where the change most likely happened, so that
interpolating the result recovers a smooth track.

2026 rshomberg@uri.edu
"""

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: Variables that change together and form one group by default.
DEFAULT_POSITION_GROUP: tuple[str, str] = ("latitude", "longitude")

_OBS = "obs"


def held_value_fixes(
    record: xr.Dataset,
    *,
    groups: Sequence[Sequence[str]] | None = None,
    max_hold_s: float = 2.0,
) -> xr.Dataset:
    """Keep only the real fixes of a sample-and-hold auxiliary record.

    Parameters
    ----------
    record : xr.Dataset
        Auxiliary record with dimension ``obs`` and a ``time`` coordinate
        along it (datetime64). Entries need not be sorted.
    groups : sequence of sequence of str, optional
        Variables that update together. A group's entry is a fix when any of
        its variables changed. By default ``latitude`` + ``longitude`` form
        one group (when both are present) and every other numeric ``obs``
        variable is its own group. Variables not named in ``groups`` are
        single-variable groups.
    max_hold_s : float, default 2.0
        A value unchanged for longer than this (from the first to the last
        entry that carries it) is treated as a real hold, such as a stop or
        a sensor dropout: an extra anchor entry keeps the value at the time
        of the last entry before the change, so interpolation stays flat
        across the hold instead of ramping through it. ``np.inf`` disables
        anchors.

    Returns
    -------
    xr.Dataset
        A record on a new ``obs`` axis: the sorted union of every group's
        fix and anchor times. A variable is NaN at times that are not one
        of its own fixes (drop NaN rows per variable before interpolating;
        the ``interpolate_aux`` step does). Variables are float64. The
        input attrs are kept, plus ``held_value_fixes_entries_in``,
        ``held_value_fixes_entries_out``, ``held_value_fixes_dropped``
        (entries in minus out) and ``held_value_fixes_max_hold_s``.
        Non-numeric variables and variables without the ``obs`` dimension
        are left out.

    Raises
    ------
    KeyError
        If ``record`` has no ``time`` coordinate, or ``groups`` names a
        variable the record does not have.
    ValueError
        If ``max_hold_s`` is negative or NaN.

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
    """
    if "time" not in record.coords and "time" not in record.data_vars:
        raise KeyError("held_value_fixes needs a 'time' coordinate on the record")
    if np.isnan(max_hold_s) or max_hold_s < 0:
        raise ValueError(f"max_hold_s must be >= 0 (or inf), got {max_hold_s!r}")

    numeric = [
        name for name, var in record.data_vars.items()
        if var.dims == (_OBS,) and (np.issubdtype(var.dtype, np.number) or var.dtype == bool)
    ]
    skipped = [name for name in record.data_vars if name not in numeric]
    if skipped:
        logger.debug("held_value_fixes: leaving out non-numeric or non-obs variables %s", skipped)
    group_list = _resolve_groups(numeric, groups, record)

    n_in = int(record.sizes.get(_OBS, 0))
    times = np.asarray(record["time"].values).astype("datetime64[ns]")
    t_ns = times.astype(np.int64)
    valid_time = ~np.isnat(times)
    order = np.argsort(t_ns, kind="stable")
    order = order[valid_time[order]]
    max_hold_ns = np.inf if np.isposinf(max_hold_s) else max_hold_s * 1e9

    per_group: list[tuple[tuple[str, ...], np.ndarray, np.ndarray]] = []
    for group in group_list:
        values = np.column_stack(
            [np.asarray(record[name].values, dtype=np.float64)[order] for name in group]
        ) if len(order) else np.empty((0, len(group)))
        g_times, g_values, n_fix, n_anchor = _group_fixes(t_ns[order], values, max_hold_ns)
        logger.debug(
            "held_value_fixes %s: %d finite entries -> %d fixes + %d hold anchors",
            list(group), int(np.isfinite(values).all(axis=1).sum()), n_fix, n_anchor,
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
    return xr.Dataset(
        data_vars,
        coords={"time": ((_OBS,), all_times.astype("datetime64[ns]"))},
        attrs=attrs,
    )


# ============================================================
# INTERNALS
# ============================================================

def _resolve_groups(
    numeric: list[str], groups: Iterable[Iterable[str]] | None, record: xr.Dataset,
) -> list[tuple[str, ...]]:
    """Variable groups: the caller's (or the default position group), then singletons."""
    if groups is None:
        groups = [DEFAULT_POSITION_GROUP] if all(v in numeric for v in DEFAULT_POSITION_GROUP) else []
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


def _group_fixes(
    t_ns: np.ndarray, values: np.ndarray, max_hold_ns: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Fix and anchor times (int64 ns, sorted, unique) and values for one group.

    ``t_ns`` is sorted; ``values`` is ``(n, k)`` aligned with it.
    """
    finite = np.isfinite(values).all(axis=1) if values.size else np.zeros(len(t_ns), bool)
    t = t_ns[finite]
    v = values[finite]
    if len(t) == 0:
        return np.empty(0, np.int64), np.empty((0, values.shape[1])), 0, 0

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
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:] - 1, len(t) - 1)

    fix_t = t[starts].copy()
    # A fix that ends a held run (previous entry was a repeat) arrived
    # between the previous entry and this one: use the midpoint.
    held_before = np.zeros(len(starts), dtype=bool)
    held_before[1:] = ~change[starts[1:] - 1]
    held_before &= ~changed_at_time[starts]
    prev = t[starts[held_before] - 1]
    fix_t[held_before] = prev + (t[starts[held_before]] - prev) // 2
    fix_v = v[starts]

    long_hold = (t[ends] - t[starts]) > max_hold_ns
    anchor_t = t[ends[long_hold]]
    anchor_v = v[ends[long_hold]]

    out_t = np.concatenate([fix_t, anchor_t])
    out_v = np.concatenate([fix_v, anchor_v])
    # Sorted; on an exact time collision keep the later-ordered (a fix over an anchor).
    order = np.argsort(out_t, kind="stable")
    out_t, out_v = out_t[order], out_v[order]
    keep = np.ones(len(out_t), dtype=bool)
    keep[:-1] = out_t[1:] != out_t[:-1]
    return out_t[keep], out_v[keep], len(starts), int(long_hold.sum())
