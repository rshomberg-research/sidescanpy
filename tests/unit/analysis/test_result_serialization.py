"""
Every analysis result's ``to_dict()`` must be ``json.dumps``-able.
"""

import json

import numpy as np
import pytest

from sidescanpy.analysis.beam import estimate_beam_geometry
from sidescanpy.analysis.characterize import characterize_values, infer_units
from sidescanpy.analysis.geometry import measure_ping_geometry
from sidescanpy.analysis.nadir import analyze_nadir
from sidescanpy.analysis.navigation import summarize_navigation
from sidescanpy.analysis.outliers import compare_outlier_rejection
from sidescanpy.analysis.tvg import fit_range_decay
from sidescanpy.analysis.usable_range import estimate_usable_range
from sidescanpy.processing.geometric import compute_geometric_ranges
from tests.unit.synthetic import make_raw_dataset


@pytest.fixture(scope="module")
def results():
    rng = np.random.default_rng(0)
    n_pings, n_samples = 30, 200
    sample = np.arange(n_samples)
    bs = 50.0 * np.exp(-sample / 80.0)[None, None, :] + rng.uniform(0.0, 1.0, (n_pings, 2, n_samples))
    bs[..., :20] = 0.0
    ds = compute_geometric_ranges(make_raw_dataset(backscatter=bs.astype(np.float32)))

    values = characterize_values(ds)
    nadir = analyze_nadir(ds, nadir_n_samples=10)
    usable = estimate_usable_range(ds)
    decay = fit_range_decay(ds, bin_width_m=2.0, usable_range_m=usable.usable_range_m)
    return {
        "values": values,
        "units": infer_units(values, decay),
        "outliers": compare_outlier_rejection(ds),
        "nadir": nadir,
        "usable_range": usable,
        "beam": estimate_beam_geometry(ds, nadir.first_bottom_return_sample, usable.usable_range_m),
        "decay": decay,
        "navigation": summarize_navigation(ds),
        "geometry": measure_ping_geometry(ds),
    }


@pytest.mark.parametrize(
    "name", ["values", "units", "outliers", "nadir", "usable_range", "beam", "decay", "navigation", "geometry"],
)
def test_to_dict_is_json_serializable(results, name):
    d = results[name].to_dict()
    assert isinstance(d, dict)
    json.loads(json.dumps(d))


def test_large_arrays_left_out_of_to_dict(results):
    assert "first_bottom_return_sample" not in results["nadir"].to_dict()
    assert "altitude_sample" not in results["nadir"].to_dict()
    assert "filtered_values" not in results["outliers"].to_dict()
    assert "fit" not in results["decay"].to_dict()
