"""grids.to_geotiff and SideScanGriddedRaster.to_dataset."""

import builtins

import numpy as np
import pytest
from pyproj import CRS

from sidescanpy.grids import GridDefinition, SideScanGriddedRaster, to_geotiff


def _raster(crs=CRS.from_epsg(32619), bbox=(4_540_000.0, 300_000.0, 4_540_037.0, 300_051.0), res=2.0, seed=0):
    grid = GridDefinition(bbox=bbox, resolution_m=res, crs=crs)
    rng = np.random.default_rng(seed)
    bs = rng.normal(-20.0, 5.0, (grid.num_cells_ns, grid.num_cells_ew))
    bs[0, :3] = np.nan
    bs[-1, -1] = np.nan
    r = SideScanGriddedRaster(grid_def=grid, backscatter=bs, channel_name="455 kHz", units="dB")
    r.dataset["n_contributing_samples"] = (["ns", "ew"], rng.integers(0, 9, bs.shape))
    return r


class TestToDataset:
    def test_plain_dataset_with_cf_crs(self):
        r = _raster()
        ds = r.to_dataset()
        assert ds["backscatter"].dims == ("y", "x")
        np.testing.assert_array_equal(ds["backscatter"].values, r.backscatter)
        np.testing.assert_array_equal(ds["y"].values, r.y_values[:, 0])
        np.testing.assert_array_equal(ds["x"].values, r.x_values[0, :])
        assert CRS.from_wkt(ds["spatial_ref"].attrs["crs_wkt"]) == r.crs
        assert ds["backscatter"].attrs["grid_mapping"] == "spatial_ref"
        assert ds["backscatter"].attrs["units"] == "dB"
        assert "n_contributing_samples" in ds
        assert ds.attrs["channel_name"] == "455 kHz"

    def test_geographic_axes(self):
        r = _raster(crs=CRS.from_epsg(4326), bbox=(41.0, -71.0, 41.001, -70.9985), res=10.0)
        ds = r.to_dataset()
        assert ds["y"].attrs["standard_name"] == "latitude"
        np.testing.assert_array_equal(ds["x"].values, r.x_values[0, :])


class TestToGeotiff:
    @pytest.fixture(autouse=True)
    def _need_rioxarray(self):
        pytest.importorskip("rioxarray")

    @pytest.mark.parametrize("crs, bbox, res", [
        (CRS.from_epsg(32619), (4_540_000.0, 300_000.0, 4_540_037.0, 300_051.0), 2.0),
        (CRS.from_epsg(4326), (41.0, -71.0, 41.001, -70.9985), 10.0),
    ])
    def test_round_trip(self, tmp_path, crs, bbox, res):
        import rioxarray

        r = _raster(crs=crs, bbox=bbox, res=res)
        path = to_geotiff(r, tmp_path / "out.tif")
        back = rioxarray.open_rasterio(path, masked=False).squeeze("band", drop=True)

        assert back.rio.crs == crs
        assert np.isnan(back.rio.nodata)
        assert back.dtype == np.float32
        # North-up: first row is the northernmost cell row.
        expected = r.backscatter[::-1].astype(np.float32)
        np.testing.assert_array_equal(back.values, expected)
        # Pixel centers read back equal the raster's cell centers.
        np.testing.assert_allclose(back["y"].values, r.y_values[::-1, 0], rtol=0, atol=1e-9 * max(1.0, abs(bbox[2])))
        np.testing.assert_allclose(back["x"].values, r.x_values[0, :], rtol=0, atol=1e-9 * max(1.0, abs(bbox[3])))
        t = back.rio.transform()
        ys, xs = r.y_values[:, 0], r.x_values[0, :]
        assert t.a == pytest.approx((xs[-1] - xs[0]) / (xs.size - 1))
        assert t.e == pytest.approx(-(ys[-1] - ys[0]) / (ys.size - 1))
        left, bottom, right, top = back.rio.bounds()
        assert left == pytest.approx(xs[0] - t.a / 2)
        assert top == pytest.approx(ys[-1] - t.e / 2)
        assert back.attrs["units"] == "dB"
        assert back.attrs["channel_name"] == "455 kHz"

    def test_from_dataset_and_other_variable_with_int_nodata(self, tmp_path):
        import rioxarray

        r = _raster()
        path = to_geotiff(r.to_dataset(), tmp_path / "n.tif", variable="n_contributing_samples", dtype="int32", nodata=-1)
        back = rioxarray.open_rasterio(path).squeeze("band", drop=True)
        assert back.rio.nodata == -1
        np.testing.assert_array_equal(back.values, r.dataset["n_contributing_samples"].values[::-1])

    def test_finite_nodata_replaces_nan(self, tmp_path):
        import rioxarray

        r = _raster()
        back = rioxarray.open_rasterio(to_geotiff(r, tmp_path / "f.tif", nodata=-9999.0)).squeeze("band", drop=True)
        assert back.rio.nodata == -9999.0
        assert np.all(back.values[::-1][np.isnan(r.backscatter)] == -9999.0)

    def test_unknown_variable(self, tmp_path):
        with pytest.raises(KeyError, match="nope"):
            to_geotiff(_raster(), tmp_path / "x.tif", variable="nope")


def test_missing_rioxarray_points_at_geo_extra(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rioxarray" or name.startswith("rioxarray."):
            raise ImportError("No module named 'rioxarray'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"sidescanpy\[geo\]"):
        to_geotiff(_raster(), tmp_path / "x.tif")
