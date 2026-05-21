'''
Read HDF5 file into MultiFrequencySideScanSurvey

2026 rshomberg@uri.edu
'''

# TODO: I need to ensure that the hdf5 format is standardized.  
# I am expecting to get a new hdf5 file format shortly. It may make sense to
# store those separately or create a different io module

import logging 
import xarray as xr 
from pathlib import Path 
import h5py 

from sidescanpy.core.survey import MultiFrequencySideScanSurvey, SideScanFrequencyChannel 

def from_hdf5(input_file: Path) -> MultiFrequencySideScanSurvey:
    '''Read survey from HDF5 file with multiple frequencies'''

    logging.debug(f"Reading survey from HDF5: {input_file}")

    # Construct empty survey w/ metadata
    survey = MultiFrequencySideScanSurvey(
        source_file=str(input_file),
        source_format='hdf5',
        survey_name=''
    )

    # Read all frequency groups
    with h5py.File(input_file, 'r') as f:
        freq_keys = [key for key in f.keys() if key.startswith('freq_')]

        if not freq_keys:
            raise FileNotFoundError(
                f"No frequency groups (freq_*) found in {input_file}" +
                f"Available groups: {list(f.keys())}"
            )

    logging.debug(f"Found {len(freq_keys)} frequencies")

    for freq_key in sorted(freq_keys):
        ds = xr.open_dataset(
            input_file,
            engine='h5netcdf',
            group=freq_key
        )
        survey.add_frequency(freq_key, ds)
        logging.debug(f"  Loaded {freq_key}: ({ds.sizes['ping']} pings)")

    logging.debug(f"HDF5 file loaded: {input_file}")

    return survey


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)8s] %(filename)<path>)s'
    )

    logging.info(f"Testing sidescanpy/io/hdf5/reader.py")

    from pathlib import Path 
    input_file = Path('./tests/output/survey.h5')

    logging.info(f"Reading survey from hdf5: {input_file}")

    if not input_file.exists():
        logging.error(f"File not found: {input_file}")
        logging.info("Try running writer.py to create test file")

        exit(1)
    
    survey = from_hdf5(input_file)

    logging.info(f"Survey loaded with {survey.n_frequencies} frequencies")