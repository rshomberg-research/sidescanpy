'''
JSF to netCDF Converter

2026 rshomberg@uri.edu
'''

import logging
from pathlib import Path 
import xarray as xr 
import numpy as np 

from sidescanpy.converters.base import SonarConverter 
from sidescanpy.converters.jsf import reader
from sidescanpy.io.jsf import messages 

class JSFConverter(SonarConverter):
    '''Convert JSF file to standardized netCDF format'''

    def read(self, filepath: Path) -> dict:
        '''Read JSF file into dict'''
        logging.debug(f"Reading JSF file: {filepath}")
        data = reader.read_jsf_file(filepath)   # FIXME: does not exist
        return data
    
    def to_netcdf(self, data: dict, output_path: Path) -> xr.Dataset:
        '''convert JSF data to netCDF dataset'''
        logging.debug(f"Converting to netCDF: {output_path}")

        # FIXME: include correct format
        # Example structure - customize based on your data
        dataset = xr.Dataset(
            {
                'intensity': (['ping', 'sample'], data.get('intensity', np.array([]))),
                'latitude': (['ping'], data.get('latitude', np.array([]))),
                'longitude': (['ping'], data.get('longitude', np.array([]))),
                'depth': (['ping'], data.get('depth', np.array([]))),
            },
            attrs={
                'source': 'JSF',
                'version': '1.0',
            }
        )
        
        return dataset


# TODO: delete old code
# import numpy as np
# import xarray as xr
# from sidescanpy.core.types import SidescanPing, SidescanPingSeries
# from sidescanpy.data.auxiliary import AuxiliaryData
# from .messages import JSFMessageBody

# # FIXME: this is not used at all

# def decode_samples(payload: bytes) -> np.ndarray:
#     """Stub: decode payload into numpy array of intensity samples"""
#     return np.frombuffer(payload, dtype=np.int16)

# def sonar_to_core(sonar_msgs: list[JSFMessageBody]) -> SidescanPingSeries:
#     pings = []
#     for msg in sonar_msgs:
#         samples = decode_samples(msg.payload)
#         pings.append(SidescanPing(
#             time=msg.header.timestamp,
#             intensity=samples,
#             slant_range=msg.header.size,        # stub field
#             sample_interval=1e-6                # placeholder
#         ))
#     return SidescanPingSeries(pings)

# def gps_to_aux(gps_msgs: list[JSFMessageBody]) -> AuxiliaryData:
#     fields = {"latitude": None, "longitude": None}
#     return AuxiliaryData.from_messages("gps", gps_msgs, fields)

# def dvl_to_aux(dvl_msgs: list[JSFMessageBody]) -> AuxiliaryData:
#     fields = {"altitude": None, "vx": None, "vy": None, "vz": None}
#     return AuxiliaryData.from_messages("dvl", dvl_msgs, fields)
