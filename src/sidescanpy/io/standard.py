"""Standard helpers for sonar datasets."""

import xarray as xr
import numpy as np
import logging


def create_sonar_dataset(
    data_dict: dict[str, np.ndarray],
    coordinates: dict[str, np.ndarray],
    attributes: dict,
) -> xr.Dataset:
    """Create standardized xarray.Dataset for sonar data."""
    ds = xr.Dataset(data_vars=data_dict, coords=coordinates)
    ds.attrs.update(attributes)
    return ds


def validate_sonar_dataset(ds: xr.Dataset) -> None:
    """Validate dataset schema - same as SideScanSurvey._validate_schema."""
    required_dims = {'ping', 'channel'}
    if not required_dims.issubset(set(ds.dims)):
        raise ValueError(f"Missing dimensions: {required_dims}")
    
    required_vars = {'samples', 'ping_number', 'timestamp', 'latitude', 'longitude', 'heading', 'depth'}
    if not required_vars.issubset(set(ds.data_vars)):
        raise ValueError(f"Missing variables: {required_vars}")


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Testing io/standard.py\n")
    
    # Test 1: Create valid dataset
    logging.info("Test 1: Creating valid dataset...")
    
    data_dict = {
        'samples': (['ping', 'channel', 'sample'], np.random.randn(10, 2, 100)),
        'ping_number': (['ping', 'channel'], np.tile(np.arange(10)[:, None], (1, 2))),
        'timestamp': (['ping', 'channel'], np.tile(np.arange(10)[:, None], (1, 2)) * 0.1),
        'latitude': (['ping', 'channel'], np.full((10, 2), 41.55)),
        'longitude': (['ping', 'channel'], np.full((10, 2), -71.42)),
        'heading': (['ping', 'channel'], np.full((10, 2), 180.0)),
        'depth': (['ping', 'channel'], np.full((10, 2), 25.0)),
    }
    
    coordinates = {
        'ping': np.arange(10),
        'channel': np.arange(2),
        'sample': np.arange(100),
    }
    
    attributes = {
        'transmit_freq_start': 1495000,
        'transmit_freq_end': 1505000,
    }
    
    ds = create_sonar_dataset(data_dict, coordinates, attributes)
    logging.info(f"✓ Created dataset with shape: {dict(ds.dims)}")
    logging.info(f"✓ Variables: {list(ds.data_vars)}")
    
    # Test 2: Validate valid dataset
    logging.info("\nTest 2: Validating valid dataset...")
    try:
        validate_sonar_dataset(ds)
        logging.info("✓ Validation passed")
    except ValueError as e:
        logging.error(f"✗ Validation failed: {e}")
    
    # Test 3: Missing dimension
    logging.info("\nTest 3: Dataset with missing 'channel' dimension...")
    bad_data = {
        'samples': (['ping', 'sample'], np.random.randn(10, 100)),
        'ping_number': (['ping'], np.arange(10)),
        'timestamp': (['ping'], np.arange(10) * 0.1),
        'latitude': (['ping'], np.full(10, 41.55)),
        'longitude': (['ping'], np.full(10, -71.42)),
        'heading': (['ping'], np.full(10, 180.0)),
        'depth': (['ping'], np.full(10, 25.0)),
    }
    bad_coords = {'ping': np.arange(10), 'sample': np.arange(100)}
    bad_ds = xr.Dataset(data_vars=bad_data, coords=bad_coords)
    
    try:
        validate_sonar_dataset(bad_ds)
        logging.error("✗ Should have failed")
    except ValueError as e:
        logging.info(f"✓ Caught expected error: {e}")
    
    # Test 4: Missing variable
    logging.info("\nTest 4: Dataset with missing 'samples' variable...")
    bad_data2 = {
        'ping_number': (['ping', 'channel'], np.tile(np.arange(10)[:, None], (1, 2))),
        'timestamp': (['ping', 'channel'], np.tile(np.arange(10)[:, None], (1, 2)) * 0.1),
        'latitude': (['ping', 'channel'], np.full((10, 2), 41.55)),
        'longitude': (['ping', 'channel'], np.full((10, 2), -71.42)),
        'heading': (['ping', 'channel'], np.full((10, 2), 180.0)),
        'depth': (['ping', 'channel'], np.full((10, 2), 25.0)),
    }
    bad_coords2 = {'ping': np.arange(10), 'channel': np.arange(2)}
    bad_ds2 = xr.Dataset(data_vars=bad_data2, coords=bad_coords2)
    
    try:
        validate_sonar_dataset(bad_ds2)
        logging.error("✗ Should have failed")
    except ValueError as e:
        logging.info(f"✓ Caught expected error: {e}")
    
    logging.info("\nAll tests completed!")