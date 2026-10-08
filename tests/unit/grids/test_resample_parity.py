"""
Parity: the vectorized per-cell aggregation in ``grids.resample`` against
the original one-``query_ball_point``-per-cell loop, kept here as the
reference implementation.

MAX, MEDIAN, NEAREST and the per-cell counts must match exactly. MEAN and
IDW sum the same values in a different order, so they are compared at
float64 rounding (rtol=1e-12, far below the float32 resolution of the
backscatter they aggregate).

One deliberate difference: the original IDW normalized its weights over
*every* in-radius sample, NaN values included, then multiplied the first
``len(finite)`` weights positionally against the finite values -- a
misalignment whenever a cell had a NaN contributor. The reference below
weights only the finite contributors (what the original intended); with
no NaN contributors both formulations are identical.
"""

import numpy as np
import pytest
from pyproj import CRS
from scipy.spatial import cKDTree

from sidescanpy.grids import resample as resample_mod
from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.resample import ResamplingMethod, _aggregate_cells


def _reference_loop(tree, ys, xs, values, grid_ys, grid_xs, radius, method):
    """The pre-vectorization implementation (IDW weighting corrected, see module docstring)."""
    n_ns, n_ew = grid_ys.shape
    out = np.full((n_ns, n_ew), np.nan)
    n = np.zeros((n_ns, n_ew), dtype=int)
    for i in range(n_ns):
        for j in range(n_ew):
            cy, cx = grid_ys[i, j], grid_xs[i, j]
            idx = tree.query_ball_point([cy, cx], r=radius)
            if not idx:
                continue
            contrib = values[idx]
            finite = np.isfinite(contrib)
            valid = contrib[finite]
            if len(valid) == 0:
                continue
            n[i, j] = len(valid)
            if method == ResamplingMethod.MEAN:
                out[i, j] = np.mean(valid)
            elif method == ResamplingMethod.MEDIAN:
                out[i, j] = np.median(valid)
            elif method == ResamplingMethod.MAX:
                out[i, j] = np.max(valid)
            elif method == ResamplingMethod.NEAREST:
                _, k = tree.query([cy, cx], k=1)
                if np.isfinite(values[k]):
                    out[i, j] = values[k]
            elif method == ResamplingMethod.IDW:
                idx_valid = np.asarray(idx)[finite]
                d = np.sqrt((ys[idx_valid] - cy) ** 2 + (xs[idx_valid] - cx) ** 2)
                d[d == 0] = 1e-9
                w = 1.0 / d
                w /= np.sum(w)
                out[i, j] = np.sum(valid * w)
    return out, n


def _swath(seed, n_pings=60, n_samples=80, heading_deg=20.0, spacing=0.3, interval=0.25, nan_frac=0.0):
    """Synthetic port+starboard swath in projected metres around (1000, 2000)."""
    rng = np.random.default_rng(seed)
    h = np.radians(heading_deg)
    along = np.arange(n_pings) * spacing + rng.normal(0, 0.02, n_pings)
    track_y = 1000 + along * np.cos(h)
    track_x = 2000 + along * np.sin(h)
    across = np.arange(1, n_samples + 1) * interval
    ys, xs = [], []
    for side in (-1, 1):
        ys.append(track_y[:, None] - side * across[None, :] * np.sin(h))
        xs.append(track_x[:, None] + side * across[None, :] * np.cos(h))
    ys, xs = np.concatenate(ys).ravel(), np.concatenate(xs).ravel()
    vals = rng.gamma(2.0, 5.0, ys.size)
    if nan_frac:
        vals[rng.random(ys.size) < nan_frac] = np.nan
    return ys, xs, vals


def _grid(ys, xs, res):
    pad = 3 * res  # leaves empty cells around the swath
    bbox = (ys.min() - pad, xs.min() - pad, ys.max() + pad, xs.max() + pad)
    return GridDefinition(bbox=bbox, resolution_m=res, crs=CRS.from_epsg(32619)).get_grid_centers()


def _lattice():
    """Samples on a 0.25 m lattice, cells on a 1 m lattice: many sample-cell
    distances are exactly the 1.5 m search radius (inclusive boundary)."""
    g = np.arange(0.0, 8.0, 0.25)
    sy, sx = np.meshgrid(g, g, indexing="ij")
    ys, xs = sy.ravel(), sx.ravel()
    vals = np.random.default_rng(7).random(ys.size)
    vals[::11] = np.nan
    c = np.arange(0.5, 10.0, 1.0)
    gy, gx = np.meshgrid(c, c, indexing="ij")
    return ys, xs, vals, gy, gx, 1.5


def _cases():
    for name, kw, res in [
        ("clean", {}, 0.5),
        ("nan5pct", {"nan_frac": 0.05}, 0.5),
        ("sparse_fine", {"n_pings": 30, "spacing": 1.2}, 0.2),
        ("dense_coarse", {"heading_deg": 135.0, "nan_frac": 0.3}, 2.0),
    ]:
        ys, xs, vals = _swath(seed=len(name), **kw)
        gy, gx = _grid(ys, xs, res)
        yield name, (ys, xs, vals, gy, gx, 1.5 * res)
    yield "lattice_boundary", _lattice()


CASES = dict(_cases())


@pytest.mark.parametrize("method", list(ResamplingMethod))
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("max_pairs", [None, 37])
def test_vectorized_matches_reference_loop(case, method, max_pairs, monkeypatch):
    ys, xs, vals, gy, gx, radius = CASES[case]
    if max_pairs is not None:  # force many small blocks
        monkeypatch.setattr(resample_mod, "_MAX_PAIRS_PER_BLOCK", max_pairs)
    tree = cKDTree(np.column_stack([ys, xs]))

    got, got_n = _aggregate_cells(tree, ys, xs, vals, gy, gx, radius, method)
    ref, ref_n = _reference_loop(tree, ys, xs, vals, gy, gx, radius, method)

    np.testing.assert_array_equal(got_n, ref_n)
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))
    assert np.any(ref_n == 0), "case should include empty cells"
    if method in (ResamplingMethod.MEAN, ResamplingMethod.IDW):
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0, equal_nan=True)
    else:
        np.testing.assert_array_equal(got, ref)


def test_lattice_case_exercises_exact_radius_boundary():
    ys, xs, _, gy, gx, radius = CASES["lattice_boundary"]
    d = np.hypot(ys[:, None] - gy.ravel()[None, :], xs[:, None] - gx.ravel()[None, :])
    assert np.sum(d == radius) > 50


def test_idw_without_nan_matches_original_formulation():
    """With no NaN contributors the corrected IDW equals the original
    weights-over-all-indices formulation."""
    ys, xs, vals, gy, gx, radius = CASES["clean"]
    tree = cKDTree(np.column_stack([ys, xs]))
    got, _ = _aggregate_cells(tree, ys, xs, vals, gy, gx, radius, ResamplingMethod.IDW)
    for i, j in [(5, 5), (10, 20), (gy.shape[0] // 2, gy.shape[1] // 2)]:
        idx = tree.query_ball_point([gy[i, j], gx[i, j]], r=radius)
        if not idx:
            assert np.isnan(got[i, j])
            continue
        d = np.sqrt((ys[idx] - gy[i, j]) ** 2 + (xs[idx] - gx[i, j]) ** 2)
        d[d == 0] = 1e-9
        w = 1.0 / d
        w /= np.sum(w)
        np.testing.assert_allclose(got[i, j], np.sum(vals[idx] * w), rtol=1e-12)


def test_cell_blocks_cover_every_nonempty_cell():
    counts = np.array([0, 5, 0, 0, 3, 10, 0, 1, 0])
    blocks = list(resample_mod._cell_blocks(counts, max_pairs=6))
    covered = np.zeros(counts.size, dtype=bool)
    for a, b in blocks:
        covered[a:b] = True
        assert counts[a:b].sum() <= 6 or b - a == 1
    assert covered[counts > 0].all()
