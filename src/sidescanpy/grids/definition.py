# sidescanpy/grids/definition.py
"""
GridDefinition: metadata contract for a georeferenced grid.

Defines bounding box, resolution, and projection. Handles CRS conversion.
No data — just the schema/contract.
"""

import logging
from typing import NamedTuple, Tuple

import numpy as np
from pyproj import CRS, Transformer

logger = logging.getLogger(__name__)


class GridDefinition:
    """
    Georeferenced grid schema.

    A grid is defined by:
    - bbox: (south, west, north, east) in the given CRS
    - resolution_m: pixel size in meters
    - crs: pyproj.CRS (default WGS84)

    Provides derived properties (cell counts, coordinates) and
    projection helpers (reproject_to).
    """

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        resolution_m: float,
        crs: CRS | None = None,
    ):
        """
        Create a GridDefinition.

        Parameters
        ----------
        bbox : tuple of (south, west, north, east)
            Bounding box in the CRS. All in degrees if CRS is geographic,
            in meters otherwise.
        resolution_m : float
            Grid cell size in meters. Must be positive.
        crs : pyproj.CRS, optional
            Coordinate reference system. Defaults to WGS84 (EPSG:4326).

        Raises
        ------
        ValueError
            If bounds are invalid, resolution is non-positive, or
            bbox crosses antimeridian.
        """
        self.crs = crs or CRS.from_epsg(4326)
        self.resolution_m = float(resolution_m)
        self.bbox = tuple(float(x) for x in bbox)

        self._validate()

    def _validate(self) -> None:
        """Validate bbox and resolution."""
        south, west, north, east = self.bbox

        # Check resolution
        if self.resolution_m <= 0:
            raise ValueError(
                f"resolution_m must be positive, got {self.resolution_m}"
            )

        # Check for empty/zero extent
        if north == south or east == west:
            raise ValueError(
                f"Bounding box has zero extent: "
                f"north={north}, south={south}, east={east}, west={west}"
            )

        # Check north > south
        if north < south:
            raise ValueError(
                f"north ({north}) must be > south ({south})"
            )

        # Check east > west
        if east < west:
            # Might be antimeridian crossing
            logger.warning(
                "Bounding box has east < west (possibly antimeridian crossing). "
                "This is not yet supported. Use two grids, one on each side of ±180°."
            )
            raise ValueError(
                "Bounding box has east < west (antimeridian crossing). "
                "Not supported; use two grids on each side of ±180°."
            )

    def _geodetic_distance_m(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Compute distance between two WGS84 points in meters.

        Uses pyproj Transformer to convert to Web Mercator (meters).
        """
        if self.crs.to_epsg() != 4326:
            raise RuntimeError(
                "_geodetic_distance_m requires WGS84 (EPSG:4326)"
            )

        transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )  # WGS84 → Web Mercator (meters)
        x1, y1 = transformer.transform(lon1, lat1)
        x2, y2 = transformer.transform(lon2, lat2)

        return float(np.hypot(x2 - x1, y2 - y1))

    def _project_distance_m(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Compute distance between two projected points in meters."""
        return float(np.hypot(x2 - x1, y2 - y1))

    @property
    def extent_m_ns(self) -> float:
        """North-south extent in meters (geodetic if WGS84)."""
        south, west, north, east = self.bbox
        mid_lon = (west + east) / 2.0

        if self.crs.to_epsg() == 4326:
            # Use geodetic distance
            return self._geodetic_distance_m(south, mid_lon, north, mid_lon)
        else:
            # Projected; direct distance
            return self._project_distance_m(south, mid_lon, north, mid_lon)

    @property
    def extent_m_ew(self) -> float:
        """East-west extent in meters (geodetic if WGS84, at mid-latitude)."""
        south, west, north, east = self.bbox
        mid_lat = (south + north) / 2.0

        if self.crs.to_epsg() == 4326:
            # Use geodetic distance at mid-latitude
            return self._geodetic_distance_m(mid_lat, west, mid_lat, east)
        else:
            # Projected; direct distance
            return self._project_distance_m(south, west, south, east)

    @property
    def num_cells_ns(self) -> int:
        """Number of grid cells in north-south direction."""
        return int(np.ceil(self.extent_m_ns / self.resolution_m))

    @property
    def num_cells_ew(self) -> int:
        """Number of grid cells in east-west direction."""
        return int(np.ceil(self.extent_m_ew / self.resolution_m))

    @property
    def num_cells_total(self) -> int:
        """Total number of grid cells."""
        return self.num_cells_ns * self.num_cells_ew

    @property
    def corner_coords(self) -> Tuple[float, float, float, float]:
        """
        Southwest and northeast corner coordinates.

        Returns
        -------
        (sw_lat, sw_lon, ne_lat, ne_lon)
        """
        south, west, north, east = self.bbox
        return (south, west, north, east)

    def get_grid_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D arrays of grid cell center coordinates.

        Returns
        -------
        lats : ndarray of shape (num_cells_ns, num_cells_ew)
        lons : ndarray of shape (num_cells_ns, num_cells_ew)
            Grid center coordinates in the CRS.

        Notes
        -----
        For WGS84, returns lat/lon. For projected CRS, returns
        northing/easting (but named as lats/lons for consistency).
        """
        south, west, north, east = self.bbox

        if self.crs.to_epsg() == 4326:
            # Geographic; generate lat/lon grid
            # For latitude, step linearly (acceptable small error)
            lats = np.linspace(
                south + self.resolution_m / 2 / 111000,
                north - self.resolution_m / 2 / 111000,
                self.num_cells_ns,
            )

            # For longitude at mid-latitude
            mid_lat = (south + north) / 2.0
            lon_offset_m = self.resolution_m / 2
            lon_scale = 111000 * np.cos(np.radians(mid_lat))

            lons = np.linspace(
                west + lon_offset_m / lon_scale,
                east - lon_offset_m / lon_scale,
                self.num_cells_ew,
            )

            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            return lat_grid, lon_grid
        else:
            # Projected; generate easting/northing grid
            ys = np.linspace(
                south + self.resolution_m / 2,
                north - self.resolution_m / 2,
                self.num_cells_ns,
            )
            xs = np.linspace(
                west + self.resolution_m / 2,
                east - self.resolution_m / 2,
                self.num_cells_ew,
            )
            y_grid, x_grid = np.meshgrid(ys, xs, indexing="ij")
            return y_grid, x_grid

    def reproject_to(self, target_crs: CRS) -> "GridDefinition":
        """
        Reproject grid to a different CRS.

        Parameters
        ----------
        target_crs : pyproj.CRS
            Target coordinate reference system.

        Returns
        -------
        GridDefinition
            New grid in target CRS. Resolution is preserved;
            bbox is transformed to target CRS.

        Notes
        -----
        This transforms the four corners + intermediate points to
        estimate the new bounding box in the target CRS.
        """
        if self.crs == target_crs:
            return GridDefinition(
                bbox=self.bbox,
                resolution_m=self.resolution_m,
                crs=target_crs,
            )

        transformer = Transformer.from_crs(self.crs, target_crs, always_xy=True)

        south, west, north, east = self.bbox

        # Collect points to transform: corners + dense edge sampling
        if self.crs.to_epsg() == 4326:
            # (lon, lat) order for geographic
            corners = [
                (west, south),
                (west, north),
                (east, north),
                (east, south),
            ]
            # Sample edges densely to capture distortion
            n_edge_samples = 10
            south_edge = [(west + i * (east - west) / n_edge_samples, south) 
                          for i in range(n_edge_samples + 1)]
            north_edge = [(west + i * (east - west) / n_edge_samples, north) 
                          for i in range(n_edge_samples + 1)]
            west_edge = [(west, south + i * (north - south) / n_edge_samples) 
                         for i in range(n_edge_samples + 1)]
            east_edge = [(east, south + i * (north - south) / n_edge_samples) 
                         for i in range(n_edge_samples + 1)]
            
            points_to_transform = corners + south_edge + north_edge + west_edge + east_edge
        else:
            # (x, y) order for projected
            corners = [
                (west, south),
                (west, north),
                (east, north),
                (east, south),
            ]
            n_edge_samples = 10
            south_edge = [(west + i * (east - west) / n_edge_samples, south) 
                          for i in range(n_edge_samples + 1)]
            north_edge = [(west + i * (east - west) / n_edge_samples, north) 
                          for i in range(n_edge_samples + 1)]
            west_edge = [(west, south + i * (north - south) / n_edge_samples) 
                         for i in range(n_edge_samples + 1)]
            east_edge = [(east, south + i * (north - south) / n_edge_samples) 
                         for i in range(n_edge_samples + 1)]
            
            points_to_transform = corners + south_edge + north_edge + west_edge + east_edge

        transformed_points = [transformer.transform(x, y) for x, y in points_to_transform]

        # Find new bounding box
        xs = [pt[0] for pt in transformed_points]
        ys = [pt[1] for pt in transformed_points]

        new_south = min(ys)
        new_north = max(ys)
        new_west = min(xs)
        new_east = max(xs)

        return GridDefinition(
            bbox=(new_south, new_west, new_north, new_east),
            resolution_m=self.resolution_m,
            crs=target_crs,
        )

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-compatible)."""
        return {
            "bbox": self.bbox,
            "resolution_m": self.resolution_m,
            "crs": self.crs.to_string(),  # WKT or PROJ string
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridDefinition":
        """Reconstruct from dict."""
        return cls(
            bbox=tuple(d["bbox"]),
            resolution_m=d["resolution_m"],
            crs=CRS.from_string(d["crs"]),
        )

    def __repr__(self) -> str:
        """Informative string representation."""
        south, west, north, east = self.bbox
        crs_name = self.crs.to_string()[:50]  # truncate if long
        return (
            f"GridDefinition("
            f"bbox=({south:.4f}, {west:.4f}, {north:.4f}, {east:.4f}), "
            f"resolution_m={self.resolution_m}, "
            f"crs={crs_name})"
        )