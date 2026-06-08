"""Fixtures for integration tests"""

import pytest

from pathlib import Path 

@pytest.fixture
def small_jsf_file():
    return Path('tests/data/jsf/sample.jsf')