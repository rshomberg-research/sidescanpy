# NOTE: this is used directly by several of the scripts currently.

# TODO: refactor and rename. This may not be used at all anymore. If it is still
# used, it try to replace the functions with more established version.

import re

def dms_to_decimal(coord_str):
    # Example: "32° 36.58772' N"
    match = re.match(r"(\d+)°\s*([\d\.]+)'\s*([NSEW])", coord_str.strip())
    if not match:
        raise ValueError(f"Invalid coordinate: {coord_str}")
    deg, minute, hemi = match.groups()
    deg, minute = float(deg), float(minute)
    dec = deg + minute / 60
    if hemi in ['S', 'W']:
        dec = -dec
    return dec

def split_location(loc_str):
    # Split the string by spaces and create lat_str and lon_str
    parts = loc_str.split(' ')
    
    # Join the first four parts for latitude and the last four parts for longitude
    lat_str = ' '.join(parts[:3])  # First four elements for latitude
    lon_str = ' '.join(parts[3:])  # Last four elements for longitude
    
    return lat_str, lon_str


if __name__=='__main__':

    loc_str = "32° 36.58772' N 117° 11.46230' W"
    
    # lat_str = "32° 36.58772' N"
    # lon_str = "117° 11.46230' W"
    lat_str, lon_str = split_location(loc_str)
    print((lat_str, lon_str))

    lat = dms_to_decimal(lat_str)
    lon = dms_to_decimal(lon_str)
    print(lat, lon)
    # → 32.60979533333333, -117.19103833333333

    import pyproj
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)

    easting, northing = transformer.transform(lon, lat)
    print(easting, northing)

