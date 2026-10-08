"""
Smoke-run the scripts in ``examples/`` on a small synthetic survey.

Run with ``.venv/bin/python -m pytest tests/examples -q -o addopts=""``.

2026 rshomberg@uri.edu
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from sidescanpy.core import SideScanSurvey
from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.io.hdf5 import write_survey
from tests.unit.synthetic import make_raw_dataset

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run_example(script: str, *args) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *map(str, args)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    return proc


def _load_example(script: str):
    spec = importlib.util.spec_from_file_location(Path(script).stem, EXAMPLES / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _moving_survey(path: Path, n_pings: int = 60, n_samples: int = 300) -> Path:
    """Synthetic survey heading east at ~1 m/ping with a range decay, so
    it can be georeferenced and gridded."""
    rng = np.random.default_rng(1)
    slant = np.arange(n_samples) * 0.75
    decay = 1000.0 * np.maximum(slant, 1.0) ** -1.0 * 10 ** (-2 * 0.05 * slant / 20)
    data = decay[None, None, :] * (1.0 + 0.1 * rng.standard_normal((n_pings, 2, n_samples)))
    ds = make_raw_dataset(backscatter=np.abs(data).astype(np.float32))
    ds["longitude"] = ("ping", (-71.5 + np.arange(n_pings) * 1.2e-5).astype(np.float64))
    ds["latitude"] = ("ping", np.full(n_pings, 41.5, dtype=np.float64))
    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
    survey.add_frequency("hz_200000", ds)
    write_survey(survey, path)
    return path


def _synthetic_survey(path: Path) -> Path:
    n_pings, n_samples = 30, 400
    rng = np.random.default_rng(0)
    decay = 100.0 * np.exp(-np.arange(n_samples) / 120.0) + 1.0
    data = decay[None, None, :] * (1.0 + 0.05 * rng.standard_normal((n_pings, 2, n_samples)))
    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
    survey.add_frequency("hz_200000", make_raw_dataset(backscatter=data.astype(np.float32)))
    write_survey(survey, path)
    return path


def test_investigate_dataset_writes_metrics_and_plots(tmp_path):
    source = _synthetic_survey(tmp_path / "synthetic.ss.h5")
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "investigate_dataset.py"), str(source), "--output-dir", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["freq_key"] == "hz_200000"
    for key in ("values", "outlier_comparison", "nadir", "usable_range", "beam_geometry", "range_decay", "units"):
        assert isinstance(metrics[key], dict), key
    pngs = {p.name for p in out.glob("*.png")}
    assert {"value_histograms.png", "nadir_zoom.png", "usable_range.png", "range_decay_fit.png"} <= pngs
    assert len(pngs) == 11


def test_quickstart_writes_waterfall_and_grid(tmp_path):
    source = _moving_survey(tmp_path / "moving.ss.h5")
    out = tmp_path / "out"
    _run_example("quickstart.py", source, "--output-dir", out, "--resolution-m", "2")
    for name in ("waterfall.png", "grid.png"):
        assert Image.open(out / name).size[0] > 100, name


def test_tvg_calibration_fits_and_applies(tmp_path):
    source = _moving_survey(tmp_path / "moving.ss.h5")
    out = tmp_path / "out"
    _run_example("tvg_calibration.py", source, "--output-dir", out, "--bin-width-m", "5", "--usable-range-m", "200")
    params = json.loads((out / "tvg_params.json").read_text())
    assert params["freq_key"] == "hz_200000"
    assert set(params["tvg"]) == {"spreading_loss_db", "absorption_db_per_m"}
    assert params["fit"]["best_candidate"] is not None
    assert (out / "tvg_calibration.png").stat().st_size > 0


def test_gridding_and_geotiff(tmp_path):
    source = _moving_survey(tmp_path / "moving.ss.h5")
    out = tmp_path / "out"
    proc = _run_example("gridding_and_geotiff.py", source, "--output-dir", out, "--resolution-m", "2")
    assert (out / "moving_hz_200000.png").stat().st_size > 0
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        assert "skipping moving_hz_200000.tif" in proc.stderr
        assert not (out / "moving_hz_200000.tif").exists()
    else:
        assert (out / "moving_hz_200000.tif").stat().st_size > 0


def test_raster_to_dataarray_is_north_up():
    module = _load_example("gridding_and_geotiff.py")
    grid = GridDefinition(bbox=(4320000.0, 400000.0, 4330000.0, 410000.0), resolution_m=1000.0, crs=_utm())
    raster = SideScanGriddedRaster(grid_def=grid, backscatter=np.arange(100.0).reshape(10, 10))
    da = module.raster_to_dataarray(raster)
    assert da.dims == ("y", "x")
    assert np.all(np.diff(da["y"].values) < 0)
    assert np.all(np.diff(da["x"].values) > 0)
    assert float(da.isel(y=-1, x=0)) == float(raster.backscatter[np.argmin(raster.y_values[:, 0]), 0])


def _utm():
    from pyproj import CRS

    return CRS.from_epsg(26919)


@pytest.mark.parametrize("script, png", [
    ("figures/compare_grid_methods.py", "grid_methods.png"),
    ("figures/grid_coverage.py", "grid_coverage.png"),
])
def test_figure_recipes_run(tmp_path, script, png):
    source = _moving_survey(tmp_path / "moving.ss.h5")
    out = tmp_path / "out"
    _run_example(script, source, "--output-dir", out, "--resolution-m", "2")
    assert (out / png).stat().st_size > 0


@pytest.fixture
def sample_raster():
    grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=5000.0)
    rng = np.random.default_rng(0)
    backscatter = rng.random((grid.num_cells_ns, grid.num_cells_ew))
    raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter, channel_name="455 kHz")
    raster.dataset["n_contributing_samples"] = (["ns", "ew"], rng.integers(10, 100, size=backscatter.shape))
    return raster


class TestPlotMethodComparison:
    plot = staticmethod(_load_example("figures/compare_grid_methods.py").plot_method_comparison)

    def test_one_row_for_three_methods(self, sample_raster):
        fig, axes = self.plot({"mean": sample_raster, "max": sample_raster, "median": sample_raster})
        assert axes.shape == (1, 3)
        plt.close(fig)

    def test_wraps_to_two_rows_and_hides_unused(self, sample_raster):
        fig, axes = self.plot({f"m{i}": sample_raster for i in range(5)})
        assert axes.shape == (2, 3)
        assert not axes[1, 2].get_visible()
        plt.close(fig)

    def test_nan_renders_black(self, sample_raster):
        fig, axes = self.plot({"mean": sample_raster, "max": sample_raster})
        mesh = axes[0, 0].collections[-1]
        np.testing.assert_array_equal(mesh.get_cmap().get_bad(), (0.0, 0.0, 0.0, 1.0))
        plt.close(fig)

    def test_shared_explicit_range(self, sample_raster):
        fig, axes = self.plot({"mean": sample_raster, "max": sample_raster}, vmin=0.2, vmax=0.8)
        for ax in axes.flat:
            assert ax.collections[-1].get_clim() == (0.2, 0.8)
        plt.close(fig)


class TestPlotGridCoverage:
    plot = staticmethod(_load_example("figures/grid_coverage.py").plot_grid_coverage)

    def test_title_reports_coverage(self, sample_raster):
        fig, ax = self.plot(sample_raster)
        assert "coverage" in ax.get_title()
        plt.close(fig)

    def test_alpha_follows_sample_count(self, sample_raster):
        fig, ax = self.plot(sample_raster)
        rgba = ax.images[0].get_array()
        counts = sample_raster.dataset["n_contributing_samples"].values
        np.testing.assert_allclose(rgba[..., 3], counts / counts.max())
        plt.close(fig)

    def test_sparse_raster_without_counts(self):
        grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=5000.0)
        backscatter = np.full((grid.num_cells_ns, grid.num_cells_ew), np.nan)
        backscatter.ravel()[::10] = 1.0
        fig, ax = self.plot(SideScanGriddedRaster(grid_def=grid, backscatter=backscatter))
        alpha = ax.images[0].get_array()[..., 3]
        assert set(np.unique(alpha)) == {0.0, 1.0}
        plt.close(fig)
