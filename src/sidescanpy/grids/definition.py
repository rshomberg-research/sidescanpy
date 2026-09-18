# sidescanpy/grids/definition.py
"""
GridDefinition: metadata contract for a georeferenced grid.

Defines bounding box, resolution, and projection. Handles CRS conversion.
No data — just the schema/contract.
"""

import logging
from typing import NamedTuple, Tuple

import numpy as np
from pyproj import CRS, Geod, Transformer

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
        Compute true geodesic distance between two WGS84 points in meters.

        Uses pyproj.Geod (WGS84 ellipsoid) — not a Web Mercator (EPSG:3857)
        Euclidean distance. Web Mercator's vertical scale is
        1/cos(latitude), so a Euclidean distance in its projected meters
        overstates true ground distance away from the equator (e.g.
        ~39% too large at 44°N) — this silently made extent_m_ns/
        extent_m_ew, and therefore num_cells_ns/num_cells_ew, wrong for
        any WGS84 grid away from the equator. Fixed 2026-07-02; found
        while investigating why gridded output looked noisier than
        expected relative to the requested resolution_m.
        """
        if self.crs.to_epsg() != 4326:
            raise RuntimeError(
                "_geodetic_distance_m requires WGS84 (EPSG:4326)"
            )

        geod = Geod(ellps="WGS84")
        _, _, distance_m = geod.inv(lon1, lat1, lon2, lat2)
        return float(distance_m)

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


# ============================================================
# UTM AUTO-SELECTION
# ============================================================

def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    """Pick the WGS84 UTM zone CRS containing (lon, lat).

    Standard "auto-UTM" heuristic: zone = floor((lon + 180) / 6) + 1,
    hemisphere from the sign of ``lat``. EPSG 326xx for northern-
    hemisphere zones, 327xx for southern. Good enough to pick a sane
    projected CRS automatically for a survey's gridded output; near a
    zone boundary (every 6° of longitude) the "correct" zone is a
    judgment call anyway — pass an explicit CRS via ``GridDefinition``
    if a specific zone/datum is required (e.g. matching a state-plane
    system already in use for a project).

    Parameters
    ----------
    lon, lat : float
        WGS84 degrees. Typically the center of the survey's bbox.

    Returns
    -------
    pyproj.CRS
        WGS84 / UTM zone CRS (EPSG 32601-32660 or 32701-32760).
    """
    zone = int((lon + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def utm_grid_from_wgs84_bbox(
    bbox: Tuple[float, float, float, float],
    resolution_m: float,
) -> "GridDefinition":
    """Build a GridDefinition in an auto-selected UTM zone from a WGS84 bbox.

    Picks the UTM zone containing the bbox's center point (via
    ``utm_crs_for_lonlat``), then reprojects the bbox into that zone's
    meters via ``GridDefinition.reproject_to`` (dense edge sampling,
    not just 4 corners, for accuracy).

    Parameters
    ----------
    bbox : (south, west, north, east)
        WGS84 degrees.
    resolution_m : float
        Grid cell size in meters.

    Returns
    -------
    GridDefinition
        Projected (UTM) grid covering the same ground area as ``bbox``.
    """
    south, west, north, east = bbox
    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    utm_crs = utm_crs_for_lonlat(center_lon, center_lat)

    wgs84_grid = GridDefinition(bbox=bbox, resolution_m=resolution_m, crs=CRS.from_epsg(4326))
    return wgs84_grid.reproject_to(utm_crs)


def bbox_from_channel_dataset(
    ds, margin_frac: float = 0.1, min_margin_deg: float = 0.001
) -> Tuple[float, float, float, float]:
    """(south, west, north, east) bbox fit to a channel dataset's
    per-sample lat/lon extent, with a small margin so edge cells aren't
    clipped.

    Promoted from a project-local helper (``auto_grid_bbox`` in
    ``project-a/scripts/visualization_demo.py``) — same logic, renamed
    to fit this module's existing ``utm_crs_for_lonlat``/
    ``utm_grid_from_wgs84_bbox`` naming. Typically the first step before
    ``utm_grid_from_wgs84_bbox`` when auto-fitting a grid to a segment.

    Parameters
    ----------
    ds : xr.Dataset
        Must have ``latitude_per_sample``/``longitude_per_sample``
        (i.e. already georeferenced via ``compute_georeferencing``).
    margin_frac : float, default 0.1
        Fraction of the lat/lon extent added as margin on each side.
    min_margin_deg : float, default 0.001
        Minimum margin (degrees), for a near-zero-extent segment.

    Returns
    -------
    (south, west, north, east)

    Raises
    ------
    ValueError
        If no finite latitude/longitude_per_sample values are present.
    """
    lats = ds["latitude_per_sample"].values
    lons = ds["longitude_per_sample"].values
    # Finite alone isn't enough -- a physically-invalid-but-finite value
    # (e.g. lat=103 deg, found via a synthetic-data exercise tracing to a
    # corrupted compute_georeferencing() result) can blow up this bbox's
    # extent, which then blows up an auto-selected UTM grid's cell count
    # downstream (OverflowError converting an infinite cell count to
    # int). One bad sample in an otherwise-good survey shouldn't be able
    # to do that.
    valid_lats = lats[np.isfinite(lats) & (np.abs(lats) <= 90)]
    valid_lons = lons[np.isfinite(lons) & (np.abs(lons) <= 180)]
    if valid_lats.size == 0 or valid_lons.size == 0:
        raise ValueError("No valid latitude/longitude_per_sample in dataset")

    south, north = float(np.min(valid_lats)), float(np.max(valid_lats))
    west, east = float(np.min(valid_lons)), float(np.max(valid_lons))
    margin_lat = max((north - south) * margin_frac, min_margin_deg)
    margin_lon = max((east - west) * margin_frac, min_margin_deg)
    return (south - margin_lat, west - margin_lon, north + margin_lat, east + margin_lon)