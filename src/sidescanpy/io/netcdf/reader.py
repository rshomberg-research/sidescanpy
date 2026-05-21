'''
Read MultiFrequencySideScanSurvey from netCDF files

2026 rshomberg@uri.edu
'''

import logging 
from pathlib import Path 

import xarray as xr 

from sidescanpy.core.survey import MultiFrequencySideScanSurvey, SideScanFrequencyChannel 

# def _group_netcdf_filenames_by_freq(input_dir: Path) -> dict[str,list[Path]]:
#     '''Pull files from a directory and group by frequency based on filename
    
#     Files should be saved (based on writer) using grouping of
#     'freq_xxxx_hz_surveyName.nc'. Often surveyName represents only a partial
#     survey where multiple files represent the full 
#     '''
#     ...

def from_netcdf(input_dir: Path) -> MultiFrequencySideScanSurvey:
    '''Read survey from directory of netCDF files (one per frequency)'''

    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    logging.debug(f"Reading survey from netCDF directory: {input_dir}")

    nc_files = sorted(input_dir.glob('freq_*/*.nc'))  # FIXME: magic str
    if not nc_files:
        raise FileNotFoundError(f"No freq_*/*.nc files found in {input_dir}")
    
    logging.info(f"Found {len(nc_files)} netCDF files")

    survey = MultiFrequencySideScanSurvey(
        # FIXME: this should really be some kind of SurveyMetaData object
        source_file=input_dir.name,
        source_format='netcdf',
        survey_name=''
    )

    for nc_file in nc_files:
        freq_key = nc_file.parent.name

        ds = xr.open_dataset(nc_file, engine='netcdf4')

        survey.add_frequency(freq_key, ds)
        logging.debug(f"Loaded {freq_key}: {nc_file} ({ds.sizes['ping']} pings)")

    logging.debug(f"{survey.n_frequencies} netCDF files loaded: {input_dir}")
    return survey

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s [%(levelname)8s] %(filename)<path>)s'
    )

    logging.info(f"Testing sidescanpy/io/netcdf/reader.py")

    input_dir = Path("./tests/output/netcdf")

    if not input_dir.exists():
        logging.error(f"Directory not found: {input_dir}")
        logging.info(f"Try running writer.py first to create test files")
        exit(1)

    survey = from_netcdf(input_dir)

    logging.info(f"Loaded with {survey.n_frequencies} frequencies")