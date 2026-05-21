'''Write MultiFrequencySideScanSurvey to netCDF (one file per frequency)

2026 rshomberg@uri.edu'''

import logging 
from pathlib import Path 

import xarray as xr 
from sidescanpy.core.survey import MultiFrequencySideScanSurvey, SideScanFrequencyChannel

# FIXME: the way this needs to work is that a single frequency channel is
# outputted to as a netcdf

def _frequency_channel_to_netcdf(survey: SideScanFrequencyChannel, output_file: Path) -> Path:
    '''Convert a single frequency channel to a netcdf file'''

    # output_file.parent.mkdir(parents=False, exist_ok=True)
    ds: xr.Dataset = survey.dataset # FIXME: figure out this ide warning
    ds.to_netcdf(output_file, engine='netcdf4')
    logging.debug(f"Saved: {output_file} ({ds.sizes['ping']}) pings")
    
    return output_file

def to_netcdf(survey: MultiFrequencySideScanSurvey, output_dir: Path) -> list[Path]:
    '''Write survey to netCDF (one per frequency)'''

    fname_base: str = survey.survey_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.debug(f"Writing survey to netCDFs: {output_dir}")
    logging.debug(f"  {survey.n_frequencies} frequencies")

    output_files = []
    for freq_key in survey.frequencies:
        freq_dir = output_dir / freq_key
        freq_dir.mkdir(exist_ok=True, parents=True)
        freq_survey = survey.get_frequency(freq_key)    # FIXME: fix naming convention (from get_frequency)

        output_file = output_dir / f'{freq_key}/{fname_base}.nc'
        _frequency_channel_to_netcdf(freq_survey, output_file)

        # logging.debug(f"Wrote {freq_key}: {output_file} ({ds.sizes['ping']} pings)")
        output_files.append(output_file)

    logging.debug(f"netCDF export ({len(output_files)} files)")
    return output_files

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)8s] %(filename)<path>)s'
    )

    logging.info(f"Testing sidescanpy/io/netcdf/writer.py")

    from pathlib import Path 
    import pickle 

    from sidescanpy.io.jsf import converter as jsf_converter

    # Load test data
    pkl_file = Path("./tests/data/temp_jsf_data.pkl") 

    if not pkl_file.exists():
        logging.error(f"File not found: {pkl_file}")
        exit(1)

    with open(pkl_file, 'rb') as f:
        messages = pickle.load(f)

    survey = jsf_converter.to_survey(messages, pkl_file)

    output_dir = Path("./tests/output/netcdf")
    output_files = to_netcdf(survey, output_dir)

    logging.info(f"Done! Output files:")
    for f in output_files:
        logging.info(f"  {f}")