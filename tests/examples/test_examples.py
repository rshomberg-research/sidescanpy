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
        import rioxarray
    except ImportError:
        assert "skipping moving_hz_200000.tif" in proc.stderr
        assert not (out / "moving_hz_200000.tif").exists()
    else:
        tif = out / "moving_hz_200000.tif"
        back = rioxarray.open_rasterio(tif).squeeze("band", drop=True)
        assert back.rio.crs is not None and back.rio.crs.to_epsg() is not None
        assert np.all(np.diff(back["y"].values) < 0)  # north-up
        assert np.isfinite(back.values).any()


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


def test_build_yolo_dataset_writes_ultralytics_layout(tmp_path):
    proc = _run_example("build_yolo_dataset.py", "--output-dir", tmp_path)
    dataset = tmp_path / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    assert set(manifest["outputs"]) == {"chip", "dual-swath"}
    for style, counts in manifest["outputs"].items():
        assert counts["target_images"] > 0 and counts["background_images"] > 0
        data = (dataset / style / "data.yaml").read_text()
        assert "train: images/train" in data
        labels = list((dataset / style / "labels").rglob("bg_*.txt"))
        assert labels and all(p.read_text() == "" for p in labels)
    assert "dataset/chip/images/train/" in proc.stdout


def test_correction_progression_figure(tmp_path):
    from sidescanpy.core.target import Target
    from sidescanpy.processing import correct_survey
    from sidescanpy.targets.io import write_targets_json
    from tests.unit.export.surveys import swath_friendly_survey

    ss_dir = tmp_path / "ss_h5"
    ss_dir.mkdir()
    write_survey(correct_survey(swath_friendly_survey()), ss_dir / "a.ss.h5")
    lat, lon = np.linspace(40.0, 40.001, 300), np.linspace(-70.0, -70.001, 300)
    targets = [
        Target(target_id=str(i), source_format="x", catalog_source="x", latitude=float(lat[p]), longitude=float(lon[p]),
               channel=i % 2, sample=15, position_confidence="verified")
        for i, p in enumerate((100, 200))
    ]
    write_targets_json(targets, tmp_path / "targets.json")
    out = tmp_path / "out"
    _run_example("figures/correction_progression.py", ss_dir, tmp_path / "targets.json", "--output-dir", out,
                 "--chip-size-px", "16", "--geo-size-px", "16")
    assert Image.open(out / "correction_progression.png").size[1] > 200


def test_plot_progression_blank_cells():
    module = _load_example("figures/correction_progression.py")
    image = np.arange(64.0).reshape(8, 8)
    fig, axes = module.plot_progression([("a", {"uncorrected": (image, (2.0, 3.0)), "geometric": (None, None)})])
    assert axes.shape == (1, 5)
    assert len(axes[0, 0].images) == 1 and len(axes[0, 3].images) == 0
    plt.close(fig)


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


# ============================================================
# examples/project_template: the five step scripts end to end
# ============================================================

def _template_inputs(root: Path) -> None:
    """An uncorrected survey (as 01_convert would write it) with three bright
    targets, plus their targets.json, under ``root/output``."""
    import pandas as pd
    import xarray as xr

    from sidescanpy.core.target import Target
    from sidescanpy.targets.io import write_targets_json

    n_pings, n_samples = 400, 100
    rng = np.random.default_rng(0)
    lat = np.linspace(41.5, 41.5004, n_pings)
    lon = np.linspace(-71.5, -71.5004, n_pings)
    backscatter = rng.uniform(5.0, 10.0, (n_pings, 2, n_samples)).astype(np.float32)
    picks = [(80, 1, 60), (200, 0, 40), (320, 1, 30)]
    for p, ch, s in picks:
        backscatter[p - 1:p + 2, ch, s - 2:s + 3] = 60.0
    per_ping = {name: (["ping"], np.full(n_pings, value, dtype=np.float32)) for name, value in (
        ("heading_deg", 45.0), ("depth_m", 5.0), ("roll_deg", 0.0), ("pitch_deg", 0.0), ("heave_m", 0.0),
        ("altitude_m", 0.1), ("sound_vel_ms", 1500.0), ("fathometer_reading_m", 0.1),
    )}
    ds = xr.Dataset(
        {"backscatter_intensity": (["ping", "channel", "sample"], backscatter),
         "latitude": (["ping"], lat), "longitude": (["ping"], lon), **per_ping},
        coords={"ping_number": (["ping"], np.arange(1, n_pings + 1, dtype=np.int32)),
                "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="100ms").values),
                "channel_id": (["channel"], np.array(["port", "starboard"]))},
        attrs={"freq_hz_start": 900_000, "freq_hz_end": 900_000, "pulse_length_us": 0.5,
               "sample_rate_hz": 40_000.0, "processing_history": "[]"},
    )
    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
    survey.add_frequency("hz_900000", ds)
    out = root / "output"
    (out / "ss_h5").mkdir(parents=True)
    write_survey(survey, out / "ss_h5" / "synthetic.ss.h5")
    (out / "targets").mkdir()
    write_targets_json([
        Target(target_id=f"{i:04d}", source_format="synthetic", catalog_source="test", latitude=float(lat[p]),
               longitude=float(lon[p]), channel=ch, sample=s, position_confidence="verified")
        for i, (p, ch, s) in enumerate(picks)
    ], out / "targets" / "targets.json")


def test_project_template_steps_run_end_to_end(tmp_path):
    import yaml

    template = EXAMPLES / "project_template"
    _template_inputs(tmp_path)
    cfg = yaml.safe_load((template / "config.yaml").read_text())
    cfg["paths"] = {"output_root": "output"}  # no raw files or catalog: start from converted files
    cfg["frequencies"] = ["hz_900000"]
    cfg["analyze"]["pings"] = [0, 400]
    cfg["analyze"]["bin_width_m"] = 0.1
    cfg["correct"]["steps"] = ["tvg", "geometric", "georef"]
    cfg["correct"]["chunk_pings"] = 150
    cfg["export"]["waterfall"].update(window_pings=200, overlap_pings=50, overview_rows=20, dpi=40)
    cfg["export"]["grid"].update(segment_pings=200, resolution_m=0.2, dpi=40)
    cfg["export"]["track"]["dpi"] = 40
    cfg["dataset"].update(size_px=32, swath_size_px=32, jitter_px=4, box_size=8.0, with_context=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))

    for step in ("01_convert", "02_analyze", "03_correct", "04_export", "05_dataset"):
        proc = subprocess.run([sys.executable, str(template / f"{step}.py"), str(config)],
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, f"{step}:\n{proc.stderr[-3000:]}"
        assert "Traceback" not in proc.stderr
        report = json.loads((tmp_path / "output" / "reports" / f"{step}.json").read_text())
        assert report["step"] == step

    out = tmp_path / "output"
    corrected = json.loads((out / "reports" / "03_correct.json").read_text())["results"]
    assert set(corrected["applied"]["hz_900000"]) >= {"tvg", "geometric", "georef"}
    assert list((out / "exports" / "waterfall" / "hz_900000").glob("segment_*.png"))
    assert (out / "exports" / "track" / "hz_900000" / "track.png").exists()
    dataset = json.loads((out / "reports" / "05_dataset.json").read_text())["results"]
    assert dataset["n_matched"] == 3 and dataset["n_images"] > 0
