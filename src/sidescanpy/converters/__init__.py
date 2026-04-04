"""Format converters for sonar data."""

from sidescanpy.converters.base import SonarConverter
from sidescanpy.converters.jsf import JSFConverter

__all__ = [
    'SonarConverter',
    'JSFConverter',
    'convert_to_netcdf',
    'get_converter',
    'CONVERTERS',
]

# Converter registry
CONVERTERS = {
    '.jsf': JSFConverter(),
}

def get_converter(file_path):
    """Get appropriate converter for file type."""
    from pathlib import Path
    suffix = Path(file_path).suffix.lower()
    if suffix not in CONVERTERS:
        raise ValueError(
            f"No converter for {suffix}. "
            f"Available: {list(CONVERTERS.keys())}"
        )
    return CONVERTERS[suffix]

def convert_to_netcdf(input_path, output_path=None):
    """Convert sonar file to netCDF format."""
    from pathlib import Path
    converter = get_converter(input_path)
    if output_path is None:
        output_path = Path(input_path).with_suffix('.nc')
    return converter.convert(input_path, output_path)