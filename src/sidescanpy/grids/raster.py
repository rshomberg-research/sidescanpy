"""
SideScanGriddedRaster: xarray-validated wrapper for gridded backscatter.

A raster is a GridDefinition + 2D backscatter array with xarray schema.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from pyproj import CRS

from sidescanpy.grids.definition import GridDefinition

logger = logging.getLogger(__name__)


# TODO(lib): this wrapper could become a plain xr.Dataset with a rioxarray
# .rio CRS/transform (see to_dataset()); needs viz.gridded / cli / export
# callers ported and a parity test on their outputs (PNG hashes, chips).
class SideScanGriddedRaster:
    """
    Georeferenced gridded backscatter data.

    Wraps a GridDefinition + 2D backscatter array in an xarray Dataset
    with validated schema (coordinates, variables, attributes).

    Attributes
    ----------
    grid_def : GridDefinition
        Grid definition (bbox, resolution, CRS).
    backscatter : np.ndarray
        2D array of shape (num_cells_ns, num_cells_ew).
    channel_name : str
        Frequency/channel identifier (e.g., "455 kHz").
    processing_level : str
        Processing level: "raw", "radiometric", etc.
    dataset : xr.Dataset
        Validated xarray Dataset with backscatter, lat/lon, metadata.
    """

    def __init__(
        self,
        grid_def: GridDefinition,
        backscatter: np.ndarray,
        channel_name: Optional[str] = None,
        processing_level: str = "raw",
        units: str = "linear",
    ):
        """
        Create a SideScanGriddedRaster.

        Parameters
        ----------
        grid_def : GridDefinition
            Grid definition (bbox, resolution, CRS).
        backscatter : np.ndarray
            2D backscatter array of shape (num_cells_ns, num_cells_ew).
        channel_name : str, optional
            Channel/frequency name (e.g., "455 kHz").
        processing_level : str, default "raw"
            Processing level ("raw", "radiometric", etc.).
        units : str, default "linear"
            Units of the backscatter values ("linear" or "dB"). Callers
            resampling TVG-corrected data (dB) must pass units="dB" —
            this is not inferred from processing_level.

        Raises
        ------
        ValueError
            If backscatter shape doesn't match grid, is not 2D, or is all NaN.
        """
        self.grid_def = grid_def
        self.backscatter = np.asarray(backscatter, dtype=float)
        self.channel_name = channel_name or "unknown"
        self.processing_level = processing_level
        self.units = units

        self._validate()
        self._dataset = None  # Lazy build


    def _validate(self) -> None:
        """Validate backscatter against grid definition."""
        # Check dimensionality
        if self.backscatter.ndim != 2:
            raise ValueError(
                f"backscatter must be 2D, got shape {self.backscatter.shape} "
                f"(ndim={self.backscatter.ndim})"
            )

        # Check shape is approximately correct (within ±1 cell due to rounding)
        expected_ns = self.grid_def.num_cells_ns
        expected_ew = self.grid_def.num_cells_ew
        actual_ns, actual_ew = self.backscatter.shape

        # Allow ±1 cell difference due to floating-point rounding in grid computation
        ns_ok = abs(actual_ns - expected_ns) <= 1
        ew_ok = abs(actual_ew - expected_ew) <= 1

        if not (ns_ok and ew_ok):
            raise ValueError(
                f"backscatter shape {self.backscatter.shape} does not match "
                f"grid cell count {(expected_ns, expected_ew)} "
                f"(tolerance: ±1 cell)"
            )

        # Check not all NaN
        if np.all(np.isnan(self.backscatter)):
            raise ValueError("backscatter cannot be all NaN (no data)")

    @property
    def crs(self) -> CRS:
        """Coordinate reference system."""
        return self.grid_def.crs

    @property
    def is_geographic(self) -> bool:
        """True for a geographic CRS (e.g. WGS84), False for projected (e.g. UTM)."""
        return self.crs.is_geographic

    @property
    def y_coord_name(self) -> str:
        """Name of the north-south position coordinate in ``dataset``.

        ``"latitude"`` for a geographic CRS, ``"northing"`` for a
        projected one — matches whichever :meth:`GridDefinition.get_grid_centers`
        actually returns for this raster's CRS.
        """
        return "latitude" if self.is_geographic else "northing"

    @property
    def x_coord_name(self) -> str:
        """Name of the east-west position coordinate in ``dataset``.

        ``"longitude"`` for a geographic CRS, ``"easting"`` for a
        projected one.
        """
        return "longitude" if self.is_geographic else "easting"

    @property
    def y_values(self) -> np.ndarray:
        """North-south position array (degrees or meters — see ``y_coord_name``)."""
        return self.dataset[self.y_coord_name].values

    @property
    def x_values(self) -> np.ndarray:
        """East-west position array (degrees or meters — see ``x_coord_name``)."""
        return self.dataset[self.x_coord_name].values

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box (south, west, north, east)."""
        return self.grid_def.bbox

    @property
    def bounds_dict(self) -> dict:
        """Bounding box as dict."""
        south, west, north, east = self.bbox
        return {
            "south": south,
            "west": west,
            "north": north,
            "east": east,
        }

    @property
    def resolution_m(self) -> float:
        """Grid resolution in meters."""
        return self.grid_def.resolution_m

    @property
    def num_cells_ns(self) -> int:
        """Number of cells in north-south direction."""
        return self.grid_def.num_cells_ns

    @property
    def num_cells_ew(self) -> int:
        """Number of cells in east-west direction."""
        return self.grid_def.num_cells_ew

    @property
    def num_cells_total(self) -> int:
        """Total number of cells."""
        return self.grid_def.num_cells_total

    @property
    def extent_m_ns(self) -> float:
        """North-south extent in meters."""
        return self.grid_def.extent_m_ns

    @property
    def extent_m_ew(self) -> float:
        """East-west extent in meters."""
        return self.grid_def.extent_m_ew

    @property
    def data_coverage(self) -> float:
        """Fraction of non-NaN cells."""
        n_valid = np.sum(~np.isnan(self.backscatter))
        return float(n_valid / self.backscatter.size)

    @property
    def dataset(self) -> xr.Dataset:
        """
        Build and cache xarray Dataset.

        Returns
        -------
        xr.Dataset
            Dataset with backscatter data variable, lat/lon coordinates,
            and validated schema attributes.
        """
        if self._dataset is None:
            self._dataset = self._build_dataset()
        return self._dataset

    def _build_dataset(self) -> xr.Dataset:
        """Construct xarray Dataset with schema.

        Position coordinates are named/labeled per the grid's CRS —
        ``latitude``/``longitude`` (degrees) for a geographic CRS,
        ``northing``/``easting`` (meters) for a projected one (e.g.
        UTM). See ``y_coord_name``/``x_coord_name``.
        """
        # Generate grid centers matching the actual backscatter shape
        # Get nominal grid centers from grid def
        ys, xs = self.grid_def.get_grid_centers()

        # Trim/pad to match backscatter shape exactly
        ns_actual, ew_actual = self.backscatter.shape
        ns_grid, ew_grid = ys.shape

        if (ns_grid, ew_grid) != (ns_actual, ew_actual):
            # Trim to match backscatter shape
            ys = ys[:ns_actual, :ew_actual]
            xs = xs[:ns_actual, :ew_actual]

        # Create data variables and coordinates
        data_vars = {
            "backscatter": (["ns", "ew"], self.backscatter),
        }

        coords = {
            self.y_coord_name: (["ns", "ew"], ys),
            self.x_coord_name: (["ns", "ew"], xs),
        }

        # Build dataset
        ds = xr.Dataset(data_vars, coords=coords)

        # Add attributes
        ds.attrs["crs"] = self.crs.to_string()
        ds.attrs["crs_epsg"] = self.crs.to_epsg() or "undefined"
        ds.attrs["resolution_m"] = self.resolution_m
        ds.attrs["channel_name"] = self.channel_name
        ds.attrs["processing_level"] = self.processing_level
        ds.attrs["bbox_south"] = self.bbox[0]
        ds.attrs["bbox_west"] = self.bbox[1]
        ds.attrs["bbox_north"] = self.bbox[2]
        ds.attrs["bbox_east"] = self.bbox[3]

        # Add variable attributes
        ds["backscatter"].attrs["units"] = self.units
        ds["backscatter"].attrs["long_name"] = "Backscatter intensity"

        if self.is_geographic:
            ds[self.y_coord_name].attrs["units"] = "degrees_north"
            ds[self.y_coord_name].attrs["long_name"] = "Latitude"
            ds[self.x_coord_name].attrs["units"] = "degrees_east"
            ds[self.x_coord_name].attrs["long_name"] = "Longitude"
        else:
            ds[self.y_coord_name].attrs["units"] = "m"
            ds[self.y_coord_name].attrs["long_name"] = "Northing"
            ds[self.x_coord_name].attrs["units"] = "m"
            ds[self.x_coord_name].attrs["long_name"] = "Easting"

        return ds

    def to_dataset(self) -> xr.Dataset:
        """This raster as a plain ``xr.Dataset`` on 1-D cell-center axes.

        Dims ``("y", "x")`` with 1-D ``y``/``x`` coordinates (northing/
        easting in metres for a projected CRS, latitude/longitude in
        degrees for a geographic one), in this raster's own row order
        (south to north). Every data variable of :attr:`dataset`
        (``backscatter``, plus ``n_contributing_samples`` when present) is
        carried over with its attrs, and the CRS is stored CF-style in a
        scalar ``spatial_ref`` coordinate (``crs_wkt`` / ``spatial_ref``
        attrs, referenced from each variable's ``grid_mapping``), which
        rioxarray, GDAL and most CF readers recognize. Needs no optional
        dependency.

        The grid's cell centers are evenly spaced but, because
        ``GridDefinition`` stretches ``ceil(extent / resolution_m)`` cells
        across the bbox, the spacing is only approximately
        ``resolution_m``; ``x``/``y`` hold the real centers.
        """
        src = self.dataset
        ys = src[self.y_coord_name].values[:, 0]
        xs = src[self.x_coord_name].values[0, :]
        data_vars = {}
        for name, var in src.data_vars.items():
            da = xr.DataArray(var.values, dims=("y", "x"), attrs=dict(var.attrs))
            da.attrs["grid_mapping"] = "spatial_ref"
            data_vars[name] = da
        y_attrs = dict(src[self.y_coord_name].attrs)
        x_attrs = dict(src[self.x_coord_name].attrs)
        y_attrs["standard_name"] = "latitude" if self.is_geographic else "projection_y_coordinate"
        x_attrs["standard_name"] = "longitude" if self.is_geographic else "projection_x_coordinate"
        cf = self.crs.to_cf()
        cf["spatial_ref"] = cf.get("crs_wkt", self.crs.to_wkt())
        ds = xr.Dataset(
            data_vars,
            coords={
                "y": ("y", ys, y_attrs),
                "x": ("x", xs, x_attrs),
                "spatial_ref": ((), np.int64(0), cf),
            },
            attrs=dict(src.attrs),
        )
        return ds

    def reproject_to(self, target_crs: CRS) -> "SideScanGriddedRaster":
        """
        Reproject raster to a different CRS.

        Parameters
        ----------
        target_crs : pyproj.CRS
            Target coordinate reference system.

        Returns
        -------
        SideScanGriddedRaster
            New raster in target CRS. Backscatter data preserved
            (grid bounds transformed).

        Notes
        -----
        This reprojects the grid definition; backscatter values are
        not resampled. Use resample_channel_to_grid() for resampling.
        Grid cell count may change slightly due to extent changes;
        backscatter shape is preserved.
        """
        # TODO(lib): rioxarray .rio.reproject would actually resample the
        # values into the target CRS (this only relabels the bbox); a
        # behavior change, not a swap -- needs a decision first.
        if self.crs == target_crs:
            return SideScanGriddedRaster(
                grid_def=self.grid_def,
                backscatter=self.backscatter.copy(),
                channel_name=self.channel_name,
                processing_level=self.processing_level,
                units=self.units,
            )

        reprojected_grid = self.grid_def.reproject_to(target_crs)

        return SideScanGriddedRaster(
            grid_def=reprojected_grid,
            backscatter=self.backscatter.copy(),
            channel_name=self.channel_name,
            processing_level=self.processing_level,
            units=self.units,
        )

    def slice_by_bbox(
        self, bbox: Tuple[float, float, float, float]
    ) -> "SideScanGriddedRaster":
        """
        Slice raster by geographic bounding box.

        Parameters
        ----------
        bbox : tuple of (south, west, north, east)
            Bounding box to extract. Must overlap with raster bbox.

        Returns
        -------
        SideScanGriddedRaster
            Subset raster. Cell alignment may cause subset bounds to
            exceed requested bbox slightly.

        Raises
        ------
        ValueError
            If bbox does not overlap with raster bounds.
        """
        # Get grid coordinates
        lats, lons = self.grid_def.get_grid_centers()

        target_south, target_west, target_north, target_east = bbox
        raster_south, raster_west, raster_north, raster_east = self.bbox

        # Check overlap
        if (
            target_north < raster_south
            or target_south > raster_north
            or target_east < raster_west
            or target_west > raster_east
        ):
            raise ValueError(
                f"Requested bbox {bbox} does not overlap with raster bbox {self.bbox}"
            )

        # Find indices within target bounds
        # Use small tolerance for floating-point comparison -- in CRS units,
        # so ~111 m for a geographic raster but 1 mm for a projected one.
        # TODO(lib): xarray .sel(y=slice(...), x=slice(...)) on to_dataset()
        # (or rioxarray .rio.clip_box) replaces this mask-and-rebuild; needs
        # a parity test on the returned subset and its bbox.
        tol = 0.001
        ns_mask = (lats >= target_south - tol) & (lats <= target_north + tol)
        ew_mask = (lons >= target_west - tol) & (lons <= target_east + tol)

        # Find contiguous slices
        ns_indices = np.where(np.any(ns_mask, axis=1))[0]
        ew_indices = np.where(np.any(ew_mask, axis=0))[0]

        if len(ns_indices) == 0 or len(ew_indices) == 0:
            raise ValueError(
                f"Requested bbox {bbox} does not intersect raster"
            )

        ns_start, ns_end = ns_indices[0], ns_indices[-1] + 1
        ew_start, ew_end = ew_indices[0], ew_indices[-1] + 1

        # Subset backscatter
        subset_backscatter = self.backscatter[ns_start:ns_end, ew_start:ew_end]

        # Get actual bounds from the subset coordinates
        subset_lats = lats[ns_start:ns_end, ew_start:ew_end]
        subset_lons = lons[ns_start:ns_end, ew_start:ew_end]

        new_south = float(np.nanmin(subset_lats))
        new_north = float(np.nanmax(subset_lats))
        new_west = float(np.nanmin(subset_lons))
        new_east = float(np.nanmax(subset_lons))

        # Create grid from actual subset bounds
        # Allow the grid cell count to differ from backscatter shape by up to 1 cell
        subset_grid = GridDefinition(
            bbox=(new_south, new_west, new_north, new_east),
            resolution_m=self.resolution_m,
            crs=self.crs,
        )

        return SideScanGriddedRaster(
            grid_def=subset_grid,
            backscatter=subset_backscatter,
            channel_name=self.channel_name,
            processing_level=self.processing_level,
            units=self.units,
        )

    def slice_by_indices(
        self,
        ns_slice: Optional[slice] = None,
        ew_slice: Optional[slice] = None,
    ) -> "SideScanGriddedRaster":
        """
        Slice raster by array indices.

        Parameters
        ----------
        ns_slice : slice, optional
            North-south slice (default: all).
        ew_slice : slice, optional
            East-west slice (default: all).

        Returns
        -------
        SideScanGriddedRaster
            Subset raster.
        """
        if ns_slice is None:
            ns_slice = slice(None)
        if ew_slice is None:
            ew_slice = slice(None)

        subset_backscatter = self.backscatter[ns_slice, ew_slice]

        # Get grid coordinates
        lats, lons = self.grid_def.get_grid_centers()
        subset_lats = lats[ns_slice, ew_slice]
        subset_lons = lons[ns_slice, ew_slice]

        new_south = float(np.nanmin(subset_lats))
        new_north = float(np.nanmax(subset_lats))
        new_west = float(np.nanmin(subset_lons))
        new_east = float(np.nanmax(subset_lons))

        # Create grid from actual subset bounds
        subset_grid = GridDefinition(
            bbox=(new_south, new_west, new_north, new_east),
            resolution_m=self.resolution_m,
            crs=self.crs,
        )

        return SideScanGriddedRaster(
            grid_def=subset_grid,
            backscatter=subset_backscatter,
            channel_name=self.channel_name,
            processing_level=self.processing_level,
            units=self.units,
        )


    def to_dict(self, include_data: bool = False) -> dict:
        """
        Serialize to dict.

        Parameters
        ----------
        include_data : bool, default False
            If True, include backscatter array (as nested list).
            If False, include only metadata.

        Returns
        -------
        dict
            Serializable dict.
        """
        d = {
            "grid_def": self.grid_def.to_dict(),
            "channel_name": self.channel_name,
            "processing_level": self.processing_level,
            "units": self.units,
        }

        if include_data:
            d["backscatter"] = self.backscatter.tolist()

        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SideScanGriddedRaster":
        """Reconstruct from dict."""
        grid_def = GridDefinition.from_dict(d["grid_def"])
        backscatter = np.array(d.get("backscatter"))

        return cls(
            grid_def=grid_def,
            backscatter=backscatter,
            channel_name=d.get("channel_name"),
            processing_level=d.get("processing_level", "raw"),
            units=d.get("units", "linear"),
        )

    def __repr__(self) -> str:
        """Informative string representation."""
        return (
            f"SideScanGriddedRaster("
            f"channel_name={self.channel_name!r}, "
            f"shape={self.backscatter.shape}, "
            f"coverage={self.data_coverage:.1%}, "
            f"crs={self.crs.to_epsg()})"
        )
