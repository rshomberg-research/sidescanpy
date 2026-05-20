'''Unit tests for sidescanpy/io/jsf/converter.py

2026 rshomberg@uri.edu'''

import pytest 
import numpy as np 
from pathlib import Path 
import pickle

from sidescanpy.io.jsf import converter

@pytest.mark.unit
class TestConverter:
    '''Tests for JSF converter module'''

    @pytest.fixture
    def sample_messages(self):
        '''Load test messages'''
        pkl_file = Path('./tests/data/temp_jsf_data.pkl')
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)

    def test_to_survey_creates_survey(self, sample_messages):
        '''Test that to_survey creates a valid survey'''
        survey = converter.to_survey(sample_messages, 'test.jsf')

        assert survey is not None
        assert survey.n_frequencies > 0 

    def test_to_survey_groups_frequencies(self, sample_messages):
        '''Test that frequencies are correctly grouped'''
        survey = converter.to_survey(sample_messages, "test.jsf")

        freqs = list(survey.frequencies)
        assert len(freqs) >= 1
        assert all(f.startswith('freq_') for f in freqs)

    def test_dataset_has_required_variables(self, sample_messages):
        '''Test that dataset has all required variables'''
        survey = converter.to_survey(sample_messages, 'test.jsf')

        freq_key = list(survey.frequencies)[0]
        ds = survey.get_frequency(freq_key).dataset

        required_vars = {   # TODO: check required vars
            'samples', 'ping_number', 'timestamp',
            'latitude', 'longitude', 'heading', 'depth'
        }

        assert required_vars.issubset(set(ds.data_vars))

    def test_dataset_dimensions(self, sample_messages):
        '''Test that dataset has correct dimensions'''
        survey = converter.to_survey(sample_messages, "test.jsf")

        freq_key = list(survey.frequencies)[0]
        ds = survey.get_frequency(freq_key).dataset

        # Should have a 3D array: [ping, channel, sample]
        assert 'ping' in ds.sizes
        assert 'channel' in ds.sizes
        assert 'sample' in ds.sizes 
        assert ds.sizes['channel'] == 2