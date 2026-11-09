"""
Score candidate targets (e.g. detector output) against a reference catalog.

:func:`match_targets` pairs each candidate with at most one catalog target
within a distance, closest pairs first, and reports hits, false alarms and
misses with precision and recall. Distances are geodesic on the WGS84
ellipsoid, so both lists need latitude and longitude.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from pyproj import Geod

from sidescanpy.core.target import Target

logger = logging.getLogger(__name__)

_GEOD = Geod(ellps="WGS84")


@dataclass
class TargetMatch:
    """Candidates matched to a catalog.

    Attributes
    ----------
    max_distance_m : float
        Largest distance accepted as a match.
    min_score : float or None
        Candidates scoring below this were left out before matching.
    matches : list of (int, int, float)
        ``(candidate index, catalog index, distance in metres)``, one per
        hit, indices into the lists given to :func:`match_targets`.
    false_alarms : list of int
        Candidates with no catalog target in range (indices).
    misses : list of int
        Catalog targets no candidate matched (indices).
    skipped_candidates, skipped_catalog : list of int
        Entries without a position, or (candidates) below ``min_score``;
        not counted in precision or recall.
    """

    max_distance_m: float
    min_score: float | None
    matches: list[tuple[int, int, float]] = field(default_factory=list)
    false_alarms: list[int] = field(default_factory=list)
    misses: list[int] = field(default_factory=list)
    skipped_candidates: list[int] = field(default_factory=list)
    skipped_catalog: list[int] = field(default_factory=list)

    @property
    def n_hits(self) -> int:
        return len(self.matches)

    @property
    def precision(self) -> float:
        """Hits over scored candidates (NaN when there are none)."""
        n = self.n_hits + len(self.false_alarms)
        return self.n_hits / n if n else float("nan")

    @property
    def recall(self) -> float:
        """Hits over positioned catalog targets (NaN when there are none)."""
        n = self.n_hits + len(self.misses)
        return self.n_hits / n if n else float("nan")

    def summary(self) -> str:
        """One line: hits, false alarms, misses, precision and recall."""
        return (f"{self.n_hits} hit(s), {len(self.false_alarms)} false alarm(s), "
                f"{len(self.misses)} miss(es) within {self.max_distance_m:g} m: "
                f"precision {self.precision:.2f}, recall {self.recall:.2f}")


def _positioned(t: Target) -> bool:
    return (t.latitude is not None and t.longitude is not None
            and np.isfinite(t.latitude) and np.isfinite(t.longitude))


def _score(t: Target) -> float | None:
    s = t.extra_attributes.get("score")
    return None if s is None else float(s)


def match_targets(
    candidates: Sequence[Target],
    catalog: Sequence[Target],
    *,
    max_distance_m: float,
    min_score: float | None = None,
) -> TargetMatch:
    """Match candidate targets to a reference catalog by position.

    Every candidate-catalog pair within ``max_distance_m`` is a possible
    match; pairs are accepted closest first, and each candidate and each
    catalog target is used at most once. Category is not compared.

    Parameters
    ----------
    candidates : sequence of Target
        E.g. detector output from
        :func:`~sidescanpy.export.ml.tiles.boxes_to_targets`; their
        ``extra_attributes["score"]`` is used with ``min_score``.
    catalog : sequence of Target
        The reference (ground truth) targets.
    max_distance_m : float
        Largest distance (m) between a candidate and its catalog target.
    min_score : float, optional
        Leave out candidates scoring below this (candidates without a
        score are kept).

    Returns
    -------
    TargetMatch
    """
    result = TargetMatch(max_distance_m=float(max_distance_m), min_score=min_score)
    cand_idx = []
    for i, t in enumerate(candidates):
        score = _score(t)
        if not _positioned(t) or (min_score is not None and score is not None and score < min_score):
            result.skipped_candidates.append(i)
        else:
            cand_idx.append(i)
    cat_idx = []
    for j, t in enumerate(catalog):
        (cat_idx if _positioned(t) else result.skipped_catalog).append(j)
    if result.skipped_catalog:
        logger.debug("match_targets: %d catalog target(s) without a position left out",
                     len(result.skipped_catalog))

    pairs = []
    if cand_idx and cat_idx:
        c_lat = np.array([candidates[i].latitude for i in cand_idx], dtype=float)
        c_lon = np.array([candidates[i].longitude for i in cand_idx], dtype=float)
        k_lat = np.array([catalog[j].latitude for j in cat_idx], dtype=float)
        k_lon = np.array([catalog[j].longitude for j in cat_idx], dtype=float)
        lon1, lon2 = np.meshgrid(c_lon, k_lon, indexing="ij")
        lat1, lat2 = np.meshgrid(c_lat, k_lat, indexing="ij")
        _, _, dist = _GEOD.inv(lon1.ravel(), lat1.ravel(), lon2.ravel(), lat2.ravel())
        dist = np.asarray(dist, dtype=float).reshape(len(cand_idx), len(cat_idx))
        a, b = np.nonzero(dist <= max_distance_m)
        pairs = sorted(zip(dist[a, b], a, b))

    used_c, used_k = set(), set()
    for d, a, b in pairs:
        if a in used_c or b in used_k:
            continue
        used_c.add(a)
        used_k.add(b)
        result.matches.append((cand_idx[a], cat_idx[b], float(d)))
    result.false_alarms = [cand_idx[a] for a in range(len(cand_idx)) if a not in used_c]
    result.misses = [cat_idx[b] for b in range(len(cat_idx)) if b not in used_k]
    return result
