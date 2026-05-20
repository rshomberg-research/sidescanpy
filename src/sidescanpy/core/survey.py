'''
Core Utilities for Side Scan Sonar Surveys

Two class levels
- SideScanSurvey: Single frequency wrapper around. xr.Dataset
- MultiFrequencySideScanSurvey: container for multiple frequency channels

2026 rshomberg@uri.edu
'''

import logging

import xarray as xr 
import numpy as np 

from pathlib import Path

from dataclasses import dataclass, field 
from typing import Optional, Any
from datetime import datetime 

class SideScanSurvey:
    '''Wrapper for single frequency xr.Dataset'''

    def __init__(self, dataset: xr.Dataset, freq_key: str):
        
        self._validate_schema(dataset)
        self._dataset = dataset 
        self.freq_key = freq_key

    
    @staticmethod
    def _validate_schema(ds: xr.Dataset) -> None:
        '''Validates that a dataset has the correct structure'''
        # TODO: use xr.Accessor?

        required_dims = {'ping', 'channel'}
        if not required_dims.issubset(set(ds.dims)):
            raise ValueError(f"Missing dimensions: {required_dims}")
        
        required_vars = {'samples', 'ping_number', 'timestamp', 'latitude', 'longitude', 'heading', 'depth'}
        if not required_vars.issubset(set(ds.data_vars)):
            raise ValueError(f"Missing variables: {required_vars}")
        
    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset 
    
    @property
    def frequency(self) -> float:
        '''Base frequency in Hz'''
        # TODO: check
        return float(self._dataset.attrs.get('transmit_freq_start', 0))
    
    @property
    def n_pings(self) -> int:
        '''Number of pings'''
        return int(self._dataset.sizes['ping']) # TODO: check
    
    @property
    def n_channels(self) -> int:
        '''Number of channels (generally 0=port, 1=starboard)'''
        return int(self._dataset.sizes['channel'])
    
    @property
    def n_samples(self) -> int:
        ''''Number of samples per ping per channel'''
        return int(self._dataset.sizes.get('sample',0))
    
    def __repr__(self) -> str:
        return f"SideScanSurvey(freq_key='{self.freq_key}', pings={self.n_pings}, channels={self.n_channels})"


class MultiFrequencySideScanSurvey:
    '''
    Container for multi-frequency survey
    
    This is currently contained as a dict of surveys. In this case the surveys
    should share a platform and path. However, the frequency channels cannot be
    stored in the same xr.Dataset because the n_samples is not the same for the
    different frequencies. There will be a few downstream effects that will be
    different on a per sample basis as well. The ping numbers also do not match,
    but the timestamps do. 

    As far as I can tell the only solution appears to be a replication of data
    for survey path. I would like to confirm that is the same though. It may be
    possible to store per ping data in this class and per (ping,sample) data in
    the SideScanSurvey class (renamed to FrequencyChannel?)
    '''

    def __init__(
            self,
            source_file     : str, 
            source_format   : str,
            survey_name     : str,
    ):
        '''Initialize an empty survey'''

        self.source_file    = source_file
        self.source_format  = source_format
        self.survey_name    = survey_name
        self._freq_channels : dict[str, SideScanSurvey] = {}

    def add_frequency(self, freq_key: str, dataset: xr.Dataset) -> None:
        '''Add a frequency dataset to the survey'''

        if freq_key in self._freq_channels:
            raise ValueError(f"Frequency ({freq_key}) already in survey")
        
        self._freq_channels[freq_key] = SideScanSurvey(dataset, freq_key)

    def get_frequency(self, freq_key: str) -> SideScanSurvey:
        '''Get SideScanSurvey for a specific frequency'''
        if freq_key not in self._freq_channels:
            raise KeyError(f"Frequency ({freq_key}) not found. Available: {self.frequencies}")
        return self._freq_channels[freq_key]
    
    @property
    def frequencies(self) -> list[str]:
        return sorted(self._freq_channels.keys())
    
    @property
    def freq_channels(self) -> dict[str, SideScanSurvey]:
        return dict(self._freq_channels)
    
    @property
    def n_frequencies(self) -> int:
        return len(self._freq_channels)
    
    @property
    def frequency_hz_list(self) -> list[float]:
        return sorted([s.frequency for s in self._freq_channels.values()])
    
    def __repr__(self) -> str:
        return f"MultiFrequencySideScanSurvey(n_frequencies={self.n_frequencies}, source='{self.source_file}')"
    

if __name__=='__main__':

    logging.basicConfig(
        level=logging.DEBUG
    )

    # Create dummy dataset
    ds = xr.Dataset({
        'samples': (['ping', 'channel', 'sample'], np.random.randn(10, 2, 100)),
        'ping_number': (['ping', 'channel'], np.tile(np.arange(10)[:, None], (1, 2))),
        'timestamp': (['ping', 'channel'], np.tile(np.arange(10)[:, None], (1, 2)) * 0.1),
        'latitude': (['ping', 'channel'], np.full((10, 2), 41.55)),
        'longitude': (['ping', 'channel'], np.full((10, 2), -71.42)),
        'heading': (['ping', 'channel'], np.full((10, 2), 180.0)),
        'depth': (['ping', 'channel'], np.full((10, 2), 25.0)),
    }, attrs={'transmit_freq_start': 1495000})

    # Test single frequency
    single = SideScanSurvey(ds, 'freq_14950_hz')
    print(single)
    print(f"Frequency: {single.frequency:,.0f} Hz")

    # Test multi-frequency
    survey = MultiFrequencySideScanSurvey(source_file='test.jsf', source_format='jsf', survey_name='test')
    survey.add_frequency('freq_1495000_hz', ds)
    print(survey)
    print(f"Frequencies: {survey.frequency_hz_list}")