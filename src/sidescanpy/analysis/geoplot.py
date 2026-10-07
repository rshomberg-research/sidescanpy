"""Geographic plotting of sonar survey tracks"""

# FIXME: does this get used?

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def extract_tracks(hdf5_file: str) -> List[dict]:
    """
    Extract survey tracks from HDF5 file.
    
    Returns list of dicts with keys:
        - frequency: sonar frequency
        - channel: sonar channel
        - latitude: array of latitudes
        - longitude: array of longitudes
        - ping_numbers: array of ping numbers
    """
    from sidescanpy.io.hdf5_manager import HDF5Loader
    
    tracks = []
    
    with HDF5Loader(hdf5_file) as loader:
        freq_channels = loader.get_sonar_frequencies()
        
        for freq, channel in freq_channels:
            data = loader.get_sonar_pings(freq, channel)
            
            # Remove duplicate positions (keep first occurrence of each position)
            lats = data['latitude']
            lons = data['longitude']
            
            # Only keep unique positions
            unique_positions = []
            seen = set()
            for lat, lon in zip(lats, lons):
                pos = (round(lat, 6), round(lon, 6))  # Round to avoid floating point diffs
                if pos not in seen:
                    seen.add(pos)
                    unique_positions.append((lat, lon))
            
            if unique_positions:
                lats_unique, lons_unique = zip(*unique_positions)
                tracks.append({
                    'frequency': freq,
                    'channel': channel,
                    'latitude': np.array(lats_unique),
                    'longitude': np.array(lons_unique),
                    'ping_numbers': data['ping_numbers'],
                    'file': Path(hdf5_file).stem,
                })
    
    return tracks


def _get_utm_zone(longitude: float, latitude: float) -> Tuple[int, str]:
    """Determine UTM zone from lat/lon"""
    zone_number = int((longitude + 180) / 6) + 1
    zone_letter = 'CDEFGHJKLMNPQRSTUVWX'[int((latitude + 80) / 8)]
    return zone_number, zone_letter


def _lat_lon_to_utm(latitude: np.ndarray, longitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Convert lat/lon to UTM coordinates (northing, easting)
    
    Returns: easting, northing, zone_number, zone_letter
    """
    try:
        import pyproj
    except ImportError:
        logger.error("pyproj required for UTM conversion. Install with: pip install pyproj")
        return None, None, None, None
    
    # Get UTM zone from center of data
    center_lat = np.mean(latitude)
    center_lon = np.mean(longitude)
    zone_number, zone_letter = _get_utm_zone(center_lon, center_lat)
    
    # Create coordinate transformers
    wgs84 = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    utm_zone = f'EPSG:326{zone_number}' if center_lat >= 0 else f'EPSG:327{zone_number}'
    utm = pyproj.Proj(utm_zone)
    
    # Transform
    easting, northing = pyproj.transform(wgs84, utm, longitude, latitude)
    
    return easting, northing, zone_number, zone_letter


def save_tracks_shapefile(tracks: List[dict], output_file: str) -> None:
    """Save tracks as ESRI Shapefile"""
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Point
    except ImportError:
        logger.error("geopandas required for shapefile output. Install with: pip install geopandas")
        return
    
    features = []
    
    for track in tracks:
        lons = track['longitude']
        lats = track['latitude']
        
        # Create line geometry
        coords = list(zip(lons, lats))
        line = LineString(coords)
        
        features.append({
            'geometry': line,
            'frequency': int(track['frequency']),
            'channel': track['channel'],
            'num_points': len(lats),
            'file': track['file'],
        })
    
    if features:
        gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
        gdf.to_file(output_file, driver='ESRI Shapefile')
        logger.info(f"Saved shapefile to {output_file}")


def save_tracks_geojson(tracks: List[dict], output_file: str) -> None:
    """Save tracks as GeoJSON"""
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError:
        logger.error("geopandas required for GeoJSON output. Install with: pip install geopandas")
        return
    
    features = []
    
    for track in tracks:
        lons = track['longitude']
        lats = track['latitude']
        
        # Create line geometry
        coords = list(zip(lons, lats))
        line = LineString(coords)
        
        features.append({
            'geometry': line,
            'frequency': int(track['frequency']),
            'channel': track['channel'],
            'num_points': len(lats),
            'file': track['file'],
        })
    
    if features:
        gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
        gdf.to_file(output_file, driver='GeoJSON')
        logger.info(f"Saved GeoJSON to {output_file}")


def save_tracks_kml(tracks: List[dict], output_file: str) -> None:
    """Save tracks as KML"""
    try:
        import simplekml
    except ImportError:
        logger.error("simplekml required for KML output. Install with: pip install simplekml")
        return
    
    kml = simplekml.Kml()
    
    for track in tracks:
        lons = track['longitude']
        lats = track['latitude']
        
        coords = list(zip(lons, lats))
        
        name = f"{track['file']}_freq{int(track['frequency'])}_ch{track['channel']}"
        linestring = kml.newlinestring(name=name, coords=coords)
        linestring.style.linestyle.color = simplekml.Color.rgb(255, 0, 0, 200)
        linestring.style.linestyle.width = 2
    
    kml.save(output_file)
    logger.info(f"Saved KML to {output_file}")


def plot_tracks(tracks: List[dict], output_file: Optional[str] = None) -> None:
    """Plot tracks in UTM projection (northing/easting)"""
    import matplotlib.pyplot as plt
    
    # Check for pyproj
    try:
        import pyproj
    except ImportError:
        logger.warning("pyproj not available, plotting in lat/lon instead. Install with: pip install pyproj")
        _plot_tracks_latlon(tracks, output_file)
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    
    zone_info = None
    
    for i, track in enumerate(tracks):
        color = colors[i % len(colors)]
        label = f"{track['file']}_freq{int(track['frequency'])}_ch{track['channel']}"
        
        # Convert to UTM
        easting, northing, zone_num, zone_letter = _lat_lon_to_utm(
            track['latitude'], 
            track['longitude']
        )
        
        if easting is None:
            logger.warning("Failed to convert to UTM, plotting in lat/lon")
            _plot_tracks_latlon(tracks, output_file)
            return
        
        # Store zone info from first track
        if zone_info is None:
            zone_info = (zone_num, zone_letter)
        
        ax.plot(easting, northing, 
               color=color, marker='.', markersize=3, label=label, alpha=0.7)
    
    zone_num, zone_letter = zone_info
    
    ax.set_xlabel(f'Easting (m)')
    ax.set_ylabel(f'Northing (m)')
    ax.set_title(f'Survey Track - UTM Zone {zone_num}{zone_letter} (WGS84)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add projection info to figure
    fig.text(0.12, 0.02, f'Projection: UTM Zone {zone_num}{zone_letter} WGS84', 
            fontsize=9, style='italic')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()


def _plot_tracks_latlon(tracks: List[dict], output_file: Optional[str] = None) -> None:
    """Fallback: plot tracks in lat/lon"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    
    for i, track in enumerate(tracks):
        color = colors[i % len(colors)]
        label = f"{track['file']}_freq{int(track['frequency'])}_ch{track['channel']}"
        
        ax.plot(track['longitude'], track['latitude'], 
               color=color, marker='.', markersize=3, label=label, alpha=0.7)
    
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title('Survey Track - WGS84 Geographic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()