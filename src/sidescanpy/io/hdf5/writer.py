'''
Write MultiFrequencySideScanSurvey to HDF5

2026 rshomberg@uri.edu
'''

import logging
from pathlib import Path

from sidescanpy.core.survey import MultiFrequencySideScanSurvey


def to_hdf5(survey: MultiFrequencySideScanSurvey, output_file: Path) -> Path:
    '''Write survey to single HDF5 file with multiple frequencies'''

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing survey to HDF5 file: {output_file}")

    for freq_key in survey.freq_channels:
        freq_survey = survey.get_frequency(freq_key)    # FIXME: improve naming scheme
        ds = freq_survey.dataset
        logging.debug(f"Writing {freq_key} to {output_file}")
        # each frequency becomes a group in HDF5
        ds.to_netcdf(
            output_file,
            mode='a',    # TODO: check
            group=freq_key,
            engine='h5netcdf'
        )

        logging.info(f"  Wrote {freq_key} ({ds.sizes['ping']} pings)")

    logging.info(f"HDF5 file complete: {output_file}")
    return output_file

if __name__=='__main__':

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)8s] %(message)s',
    )

    logging.info("Testing io/hdf5/writer.py")

    import pickle

    from sidescanpy.io.jsf import converter 

    # Load test data
    pkl_file = Path('./tests/data/temp_jsf_data.pkl')

    with open(pkl_file, 'rb') as f:
        messages = pickle.load(f)

    survey = converter.to_survey(messages, str(pkl_file))

    output_file = Path('./tests/output/survey.h5')
    output_file = to_hdf5(survey, output_file)

    logging.info(f"Done. Saved to {output_file}")